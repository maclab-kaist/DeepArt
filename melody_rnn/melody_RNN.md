[&larr; Back to Index](../index.md)

# 신경망 모델을 사용한 멜로디 시퀀스 학습 

이번 튜토리얼에서는 기존에 데이터로 주어진 곡들과 비슷한 느낌을 주는 멜로디를 생성하는 모델을 만들어보고자 합니다.

멜로디의 생성을 모델링함에 있어서 크게 두가지 어프로치가 있을 수 있습니다. 


1) 첫번째로는 실제 음악 이론과 작곡가의 멜로디 창작 스타일에 대한 분석을 토대로 직접 작곡 규칙을 설계하는 방법입니다.  

2) 두번째로는 악곡 데이터에서 통계적인 분석을 통해 생성 모델을 학습하는 방법입니다. 기본적으로 시퀀셜한 데이터에서 패턴을 찾아내는 작업이기 때문에 다양한 기계학습 방법론이 쓰일 수 있습니다. 


이번 튜토리얼에서는 딥러닝의 토대가 되는 인공신경망 아키텍쳐 중 시퀀스 학습을 위해 만들어진 RNN 구조를 활용해서 접근해보고자 합니다.


## 라이브러리 준비

먼저 필요한 라이브러리를 불러옵니다. 

미디 파일을 다루는 라이브러리로는 여러가지가 있지만 이 튜토리얼에서는 music21을 사용하고자 합니다.

[http://web.mit.edu/music21/](http://web.mit.edu/music21/)

그 외 Numpy, OS단에서의 로컬 path를 핸들링하기 위해 os, 데이터 저장을 위해 pickle을 임포트합니다.



```python
import music21
import numpy as np
import os
import pickle
```

## 데이터 Preprocessing (1) - Extracting note information from midi file.

우리는 미디 파일에서 음악 정보를 받아온 후 이를 바탕으로 패턴을 학습하는 모델을 만들 예정입니다. 

가장 먼저 필요한 작업은 미디 데이터셋을 준비하는 일입니다.

오픈되어 있는 미디 데이터셋에는 여러가지가 있지만, 단순한 포크송 형태의 기본적인 미디 데이터셋으로 Nottingham Database를 사용하고자 합니다.

1200개의 영국/미국 포크곡으로 이루어져 있는 데이터셋으로, 모든 미디파일은 멜로디, 화성의 두개의 트랙으로 구성되어 있습니다.

또한, 얼마전 음악 AI 스타트업 JukeDeck에서 이 데이터셋을 한번 더 정리(clean up)해서 Github에 공유했기 때문에 저희는 이를 활용하겠습니다.

[https://github.com/jukedeck/nottingham-dataset](https://github.com/jukedeck/nottingham-dataset)

이를 다운 받아서 로컬 하드드라이브의 적당한 위치에 저장합니다.

그런 다음, 데이터가 저장된 폴더를 변수로 지정해 놓습니다.



```python
data_path = '~/dataset/nottingham-dataset/MIDI/melody/'
```

먼저 미디 파일 하나를 가지고 테스트를 하기 위해 파일명 하나를 변수로 받아옵니다.


```python
file_name = 'jigs273.mid'
```

이제 music21 라이브러리에 이미 정의되어 있는 함수를 이용해서 미디 파일 내부의 음악 정보를 해석합니다.


```python
midi_obj = music21.converter.parse(data_path + file_name)
```


```python
midi_obj
```




    <music21.stream.Score 0x10a7efd50>



music21은 라이브러리 내에 정의해놓은 특정한 오브젝트 형태로 정보를 저장합니다. 

이 오브젝트안에는 특정 구조로 음악 정보들이 저장되어 있습니다.

이 구조를 이용해서 우리가 원하는 노트의 음정 / 노트의 시작지점 두가지 정보에 대한 Array를 받아오겠습니다.


```python
def create_mel_data_each_file(midi_obj):
    mel_data = dict()
    
    print(len(midi_obj.flat.getElementsByClass(music21.chord.Chord)))
    for n in midi_obj.flat.getElementsByClass(music21.chord.Chord):
        if n.offset not in mel_data:
            mel_data[n.offset] = dict()
        mel_data[n.offset]['offset'] = n.offset
        for p in n.pitches:
            mel_data[n.offset]['note'] = p.midi
    print(len(midi_obj.flat.getElementsByClass(music21.note.Note)))

    for n in midi_obj.flat.getElementsByClass(music21.note.Note):
        if n.offset not in mel_data:
            mel_data[n.offset] = dict()
        mel_data[n.offset]['offset'] = n.offset
        prev_p = 0
        for p in n.pitches:
            if prev_p < p.midi:
                mel_data[n.offset]['note'] = p.midi
            prev_p = p.midi    
    
    return mel_data
```

우리가 가지고 있는 데이터에 이미 멜로디만 따로 정리되어 있기 때문에 이 작업이 매우 간단하게 이루어졌습니다.

여러분이 작업하기 원하시는 데이터셋이 있다면 거기서 원하는 정보만 추출하는 작업이 까다로울 수도 있습니다. 보통 어느 트랙에 어떤 악기 파트가 배정되어 있는지도 정리가 되어 있지 않는 경우도 많으니까요.


```python
mel_data = create_mel_data_each_file(midi_obj)
```

    0
    165


한곡에서 165개의 멜로디 노트 정보를 받아오는 것을 확인했습니다.

자, 이제 우리 데이터셋의 모든 곡에 대해서 노트 정보를 받아와봅시다.

먼저, 폴더안의 모든 미디 파일명을 리스트로 만듭니다.



```python
data_path = '~/dataset/nottingham-dataset/MIDI/melody/'

file_list = []

for file_name in os.listdir(os.path.expanduser(data_path)):
    if file_name.endswith('.mid') or file_name.endswith('.midi'):
        file_list.append(data_path + file_name)
```

파일 리스트를 확인해봅시다.


```python
file_list 
```




    ['~/dataset/nottingham-dataset/MIDI/melody/ashover1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover16.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover17.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover18.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover19.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover20.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover21.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover22.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover23.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover24.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover25.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover26.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover27.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover28.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover29.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover30.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover31.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover32.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover33.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover34.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover35.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover36.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover37.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover38.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover39.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover40.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover41.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover42.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover43.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover44.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover45.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover46.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/ashover9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps16.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps17.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps18.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps19.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps20.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps21.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps22.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps23.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps24.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps25.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps26.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps27.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps28.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps29.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps30.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps31.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps32.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps33.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps34.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps35.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps36.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps37.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps38.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps39.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps40.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps41.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps42.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps43.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps44.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps45.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps46.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps47.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps48.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps49.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps50.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps51.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps52.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps53.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps54.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps55.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps56.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps57.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps58.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps59.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps60.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps61.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps62.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps63.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps64.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps65.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/hpps9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs100.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs101.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs102.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs103.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs104.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs105.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs106.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs107.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs108.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs109.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs110.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs111.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs112.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs113.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs114.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs115.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs116.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs117.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs118.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs119.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs120.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs121.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs122.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs123.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs124.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs125.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs126.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs127.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs128.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs129.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs130.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs131.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs132.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs133.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs134.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs135.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs136.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs137.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs138.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs139.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs140.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs141.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs142.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs143.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs144.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs145.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs146.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs147.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs148.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs149.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs150.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs151.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs152.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs153.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs154.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs155.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs156.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs157.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs158.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs159.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs16.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs160.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs161.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs162.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs163.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs164.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs165.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs166.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs167.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs168.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs169.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs17.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs170.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs171.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs172.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs173.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs174.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs175.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs176.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs177.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs178.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs179.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs18.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs180.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs181.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs182.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs183.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs184.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs185.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs186.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs187.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs188.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs189.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs19.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs190.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs191.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs192.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs193.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs194.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs195.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs196.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs197.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs198.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs199.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs20.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs200.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs201.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs202.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs203.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs204.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs205.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs206.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs207.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs208.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs209.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs21.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs210.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs211.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs212.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs213.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs214.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs215.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs216.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs217.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs218.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs219.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs22.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs220.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs221.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs222.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs223.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs224.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs225.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs226.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs227.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs228.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs229.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs23.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs230.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs231.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs232.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs233.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs234.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs235.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs236.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs237.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs238.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs239.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs24.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs240.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs241.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs242.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs243.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs244.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs245.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs246.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs247.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs248.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs249.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs25.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs250.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs251.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs252.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs253.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs254.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs255.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs256.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs257.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs258.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs259.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs26.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs260.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs261.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs262.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs263.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs264.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs265.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs266.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs267.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs268.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs269.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs27.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs270.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs271.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs272.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs273.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs274.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs275.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs276.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs277.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs278.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs279.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs28.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs280.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs281.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs282.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs283.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs284.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs285.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs286.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs287.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs288.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs289.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs29.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs290.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs291.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs292.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs293.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs294.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs295.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs296.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs297.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs298.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs299.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs30.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs300.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs301.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs302.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs303.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs304.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs305.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs306.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs307.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs308.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs309.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs31.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs310.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs311.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs312.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs313.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs314.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs315.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs316.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs317.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs318.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs319.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs32.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs320.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs321.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs322.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs323.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs324.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs325.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs326.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs327.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs328.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs329.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs33.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs330.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs331.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs332.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs333.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs334.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs335.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs336.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs337.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs338.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs339.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs34.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs340.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs35.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs36.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs37.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs38.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs39.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs40.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs41.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs42.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs43.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs44.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs45.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs46.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs47.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs48.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs49.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs50.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs51.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs52.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs53.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs54.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs55.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs56.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs57.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs58.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs59.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs60.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs61.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs62.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs63.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs64.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs65.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs66.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs67.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs68.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs69.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs70.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs71.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs72.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs73.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs74.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs75.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs76.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs77.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs78.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs79.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs80.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs81.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs82.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs83.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs84.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs85.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs86.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs87.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs88.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs89.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs90.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs91.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs92.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs93.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs94.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs95.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs96.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs97.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs98.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/jigs99.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris16.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris17.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris18.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris19.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris20.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris21.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris22.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris23.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris24.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris25.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris26.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris27.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris28.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris29.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris30.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris31.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/morris9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/playford9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c16.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c17.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c18.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c19.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c20.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c21.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c22.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c23.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c24.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c25.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c26.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c27.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c28.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c29.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c30.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c31.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c32.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c33.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c34.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c35.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c36.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c37.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c38.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c39.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c40.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c41.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c42.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c43.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c44.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c45.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c46.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c47.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c48.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c49.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c50.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c51.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c52.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c53.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c54.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c55.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c56.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c57.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c58.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c59.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c60.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c61.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c62.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c63.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c64.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c65.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c66.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c67.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c68.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c69.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c70.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c71.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c72.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c73.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c74.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c75.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c76.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c77.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c78.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c79.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c80.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c81.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsa-c9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g16.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g17.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g18.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g19.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g20.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g21.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g22.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g23.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g24.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g25.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g26.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g27.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g28.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g29.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g30.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g31.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g32.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g33.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g34.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g35.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g36.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g37.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g38.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g39.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g40.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g41.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g42.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g43.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g44.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g45.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g46.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g47.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g48.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g49.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g50.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g51.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g52.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g53.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g54.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g55.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g56.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g57.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g58.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g59.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g60.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g61.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g62.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g63.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g64.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g65.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g66.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g67.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g68.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g69.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g70.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g71.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g72.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g73.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g74.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g75.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g76.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g77.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g78.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g79.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g80.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g81.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g82.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g83.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g84.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsd-g9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l16.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l17.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l18.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l19.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l20.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l21.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l22.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l23.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l24.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l25.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l26.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l27.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l28.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l29.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l30.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l31.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l32.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l33.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l34.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l35.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l36.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l37.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l38.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l39.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l40.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l41.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l42.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l43.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l44.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l45.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l46.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l47.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l51.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l52.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l53.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l54.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l55.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l56.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l57.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l58.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l59.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l60.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l61.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l62.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l63.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l64.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l65.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l66.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l67.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l68.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l69.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l70.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l71.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l72.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l73.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l74.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l75.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l76.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l77.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l78.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l79.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l80.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l81.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l82.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l83.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l84.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l85.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l86.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l87.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l88.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l89.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l90.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l91.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l92.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsh-l93.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q16.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q17.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q18.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q19.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q20.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q21.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q22.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q23.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q24.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q25.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q26.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q27.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q28.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q29.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q30.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q31.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q32.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q33.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q34.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q35.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q36.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q37.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q38.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q39.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q40.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q41.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q42.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q43.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q44.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q45.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q46.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q47.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q48.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q49.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q50.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q51.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q52.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q53.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q54.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q55.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q56.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q57.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q58.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q59.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q60.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q61.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q62.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q63.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q64.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q65.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q66.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q67.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q68.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q69.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q70.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q71.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q72.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q73.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q74.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q75.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q76.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q77.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q78.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q79.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q80.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsm-q9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t16.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t17.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t18.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t19.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t20.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t21.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t22.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t23.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t24.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t25.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t26.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t27.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t28.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t29.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t30.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t31.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t32.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t33.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t34.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t35.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t36.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t37.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t38.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t39.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t40.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t41.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t42.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t43.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t44.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t45.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t46.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t47.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t48.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t49.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t50.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t51.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t52.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t53.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t54.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t55.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t56.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t57.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t58.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t59.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t60.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t61.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t62.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t63.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t64.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t65.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t66.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t67.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t68.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t69.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t70.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t71.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t72.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t73.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t74.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t75.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t76.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t77.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t78.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t79.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t80.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t81.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t82.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t83.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t84.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t85.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t86.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t87.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t88.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t89.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t90.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t91.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsr-t92.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z16.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z17.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z18.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z19.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z20.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z21.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z22.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z23.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z24.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z25.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z26.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z27.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z28.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z29.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z30.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z31.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z32.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z33.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z34.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/reelsu-z9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/slip1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/slip10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/slip11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/slip2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/slip3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/slip4.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/slip5.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/slip6.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/slip7.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/slip8.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/slip9.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes1.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes10.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes11.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes12.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes13.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes14.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes15.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes16.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes17.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes18.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes19.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes2.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes20.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes21.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes22.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes23.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes24.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes25.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes26.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes27.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes28.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes29.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes3.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes30.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes31.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes32.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes33.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes34.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes35.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes36.mid',
     '~/dataset/nottingham-dataset/MIDI/melody/waltzes37.mid',
     ...]



이제 모든 파일들의 path를 받아놓았으니, 루프를 돌면서 모든 파일에 대해서 멜로디 노트 정보를 저장해봅시다.


```python
mel_arr_list = []

for file_name in file_list:
    print(file_name)
    midi_obj = music21.converter.parse(file_name)
    mel_data = create_mel_data_each_file(midi_obj)

    mel_arr = []
    for key in sorted(mel_data.iterkeys()):
        mel_arr.append(mel_data[key])
    
    mel_arr_list.append(mel_arr)
```

    ~/dataset/nottingham-dataset/MIDI/melody/ashover1.mid
    0
    68
    ~/dataset/nottingham-dataset/MIDI/melody/ashover10.mid
    0
    423
    ~/dataset/nottingham-dataset/MIDI/melody/ashover11.mid
    0
    169
    ~/dataset/nottingham-dataset/MIDI/melody/ashover12.mid
    0
    239
    ~/dataset/nottingham-dataset/MIDI/melody/ashover13.mid
    0
    266
    ~/dataset/nottingham-dataset/MIDI/melody/ashover14.mid
    0
    132
    ~/dataset/nottingham-dataset/MIDI/melody/ashover15.mid
    8
    173
    ~/dataset/nottingham-dataset/MIDI/melody/ashover16.mid
    0
    147
    ~/dataset/nottingham-dataset/MIDI/melody/ashover17.mid
    0
    374
    ~/dataset/nottingham-dataset/MIDI/melody/ashover18.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/ashover19.mid
    68
    107
    ~/dataset/nottingham-dataset/MIDI/melody/ashover2.mid
    0
    214
    ~/dataset/nottingham-dataset/MIDI/melody/ashover20.mid
    0
    305
    ~/dataset/nottingham-dataset/MIDI/melody/ashover21.mid
    0
    208
    ~/dataset/nottingham-dataset/MIDI/melody/ashover22.mid
    0
    96
    ~/dataset/nottingham-dataset/MIDI/melody/ashover23.mid
    0
    238
    ~/dataset/nottingham-dataset/MIDI/melody/ashover24.mid
    0
    252
    ~/dataset/nottingham-dataset/MIDI/melody/ashover25.mid
    0
    129
    ~/dataset/nottingham-dataset/MIDI/melody/ashover26.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/ashover27.mid
    0
    171
    ~/dataset/nottingham-dataset/MIDI/melody/ashover28.mid
    0
    122
    ~/dataset/nottingham-dataset/MIDI/melody/ashover29.mid
    0
    189
    ~/dataset/nottingham-dataset/MIDI/melody/ashover3.mid
    0
    287
    ~/dataset/nottingham-dataset/MIDI/melody/ashover30.mid
    0
    136
    ~/dataset/nottingham-dataset/MIDI/melody/ashover31.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/ashover32.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/ashover33.mid
    0
    118
    ~/dataset/nottingham-dataset/MIDI/melody/ashover34.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/ashover35.mid
    0
    202
    ~/dataset/nottingham-dataset/MIDI/melody/ashover36.mid
    0
    177
    ~/dataset/nottingham-dataset/MIDI/melody/ashover37.mid
    0
    181
    ~/dataset/nottingham-dataset/MIDI/melody/ashover38.mid
    0
    576
    ~/dataset/nottingham-dataset/MIDI/melody/ashover39.mid
    0
    136
    ~/dataset/nottingham-dataset/MIDI/melody/ashover4.mid
    0
    349
    ~/dataset/nottingham-dataset/MIDI/melody/ashover40.mid
    0
    206
    ~/dataset/nottingham-dataset/MIDI/melody/ashover41.mid
    0
    155
    ~/dataset/nottingham-dataset/MIDI/melody/ashover42.mid
    0
    245
    ~/dataset/nottingham-dataset/MIDI/melody/ashover43.mid
    185
    0
    ~/dataset/nottingham-dataset/MIDI/melody/ashover44.mid
    0
    136
    ~/dataset/nottingham-dataset/MIDI/melody/ashover45.mid
    0
    273
    ~/dataset/nottingham-dataset/MIDI/melody/ashover46.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/ashover5.mid
    0
    285
    ~/dataset/nottingham-dataset/MIDI/melody/ashover6.mid
    0
    183
    ~/dataset/nottingham-dataset/MIDI/melody/ashover7.mid
    0
    214
    ~/dataset/nottingham-dataset/MIDI/melody/ashover8.mid
    0
    129
    ~/dataset/nottingham-dataset/MIDI/melody/ashover9.mid
    144
    0
    ~/dataset/nottingham-dataset/MIDI/melody/hpps1.mid
    0
    202
    ~/dataset/nottingham-dataset/MIDI/melody/hpps10.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/hpps11.mid
    0
    52
    ~/dataset/nottingham-dataset/MIDI/melody/hpps12.mid
    0
    138
    ~/dataset/nottingham-dataset/MIDI/melody/hpps13.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/hpps14.mid
    0
    128
    ~/dataset/nottingham-dataset/MIDI/melody/hpps15.mid
    0
    112
    ~/dataset/nottingham-dataset/MIDI/melody/hpps16.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/hpps17.mid
    0
    61
    ~/dataset/nottingham-dataset/MIDI/melody/hpps18.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/hpps19.mid
    0
    390
    ~/dataset/nottingham-dataset/MIDI/melody/hpps2.mid
    0
    200
    ~/dataset/nottingham-dataset/MIDI/melody/hpps20.mid
    0
    138
    ~/dataset/nottingham-dataset/MIDI/melody/hpps21.mid
    0
    122
    ~/dataset/nottingham-dataset/MIDI/melody/hpps22.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/hpps23.mid
    0
    62
    ~/dataset/nottingham-dataset/MIDI/melody/hpps24.mid
    0
    78
    ~/dataset/nottingham-dataset/MIDI/melody/hpps25.mid
    0
    78
    ~/dataset/nottingham-dataset/MIDI/melody/hpps26.mid
    0
    52
    ~/dataset/nottingham-dataset/MIDI/melody/hpps27.mid
    0
    52
    ~/dataset/nottingham-dataset/MIDI/melody/hpps28.mid
    0
    96
    ~/dataset/nottingham-dataset/MIDI/melody/hpps29.mid
    0
    100
    ~/dataset/nottingham-dataset/MIDI/melody/hpps3.mid
    0
    381
    ~/dataset/nottingham-dataset/MIDI/melody/hpps30.mid
    0
    188
    ~/dataset/nottingham-dataset/MIDI/melody/hpps31.mid
    0
    200
    ~/dataset/nottingham-dataset/MIDI/melody/hpps32.mid
    0
    62
    ~/dataset/nottingham-dataset/MIDI/melody/hpps33.mid
    0
    118
    ~/dataset/nottingham-dataset/MIDI/melody/hpps34.mid
    0
    110
    ~/dataset/nottingham-dataset/MIDI/melody/hpps35.mid
    0
    60
    ~/dataset/nottingham-dataset/MIDI/melody/hpps36.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/hpps37.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/hpps38.mid
    0
    114
    ~/dataset/nottingham-dataset/MIDI/melody/hpps39.mid
    0
    164
    ~/dataset/nottingham-dataset/MIDI/melody/hpps4.mid
    0
    103
    ~/dataset/nottingham-dataset/MIDI/melody/hpps40.mid
    0
    58
    ~/dataset/nottingham-dataset/MIDI/melody/hpps41.mid
    0
    108
    ~/dataset/nottingham-dataset/MIDI/melody/hpps42.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/hpps43.mid
    0
    112
    ~/dataset/nottingham-dataset/MIDI/melody/hpps44.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/hpps45.mid
    0
    374
    ~/dataset/nottingham-dataset/MIDI/melody/hpps46.mid
    0
    114
    ~/dataset/nottingham-dataset/MIDI/melody/hpps47.mid
    0
    122
    ~/dataset/nottingham-dataset/MIDI/melody/hpps48.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/hpps49.mid
    0
    112
    ~/dataset/nottingham-dataset/MIDI/melody/hpps5.mid
    0
    97
    ~/dataset/nottingham-dataset/MIDI/melody/hpps50.mid
    0
    66
    ~/dataset/nottingham-dataset/MIDI/melody/hpps51.mid
    0
    54
    ~/dataset/nottingham-dataset/MIDI/melody/hpps52.mid
    0
    188
    ~/dataset/nottingham-dataset/MIDI/melody/hpps53.mid
    0
    124
    ~/dataset/nottingham-dataset/MIDI/melody/hpps54.mid
    0
    104
    ~/dataset/nottingham-dataset/MIDI/melody/hpps55.mid
    0
    124
    ~/dataset/nottingham-dataset/MIDI/melody/hpps56.mid
    0
    114
    ~/dataset/nottingham-dataset/MIDI/melody/hpps57.mid
    0
    118
    ~/dataset/nottingham-dataset/MIDI/melody/hpps58.mid
    0
    106
    ~/dataset/nottingham-dataset/MIDI/melody/hpps59.mid
    0
    45
    ~/dataset/nottingham-dataset/MIDI/melody/hpps6.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/hpps60.mid
    0
    108
    ~/dataset/nottingham-dataset/MIDI/melody/hpps61.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/hpps62.mid
    0
    96
    ~/dataset/nottingham-dataset/MIDI/melody/hpps63.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/hpps64.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/hpps65.mid
    0
    192
    ~/dataset/nottingham-dataset/MIDI/melody/hpps7.mid
    0
    227
    ~/dataset/nottingham-dataset/MIDI/melody/hpps8.mid
    0
    112
    ~/dataset/nottingham-dataset/MIDI/melody/hpps9.mid
    0
    104
    ~/dataset/nottingham-dataset/MIDI/melody/jigs1.mid
    0
    171
    ~/dataset/nottingham-dataset/MIDI/melody/jigs10.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/jigs100.mid
    0
    175
    ~/dataset/nottingham-dataset/MIDI/melody/jigs101.mid
    0
    269
    ~/dataset/nottingham-dataset/MIDI/melody/jigs102.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/jigs103.mid
    0
    232
    ~/dataset/nottingham-dataset/MIDI/melody/jigs104.mid
    0
    170
    ~/dataset/nottingham-dataset/MIDI/melody/jigs105.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/jigs106.mid
    0
    257
    ~/dataset/nottingham-dataset/MIDI/melody/jigs107.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/jigs108.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs109.mid
    0
    492
    ~/dataset/nottingham-dataset/MIDI/melody/jigs11.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/jigs110.mid
    0
    2724
    ~/dataset/nottingham-dataset/MIDI/melody/jigs111.mid
    0
    502
    ~/dataset/nottingham-dataset/MIDI/melody/jigs112.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs113.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/jigs114.mid
    0
    170
    ~/dataset/nottingham-dataset/MIDI/melody/jigs115.mid
    0
    209
    ~/dataset/nottingham-dataset/MIDI/melody/jigs116.mid
    0
    160
    ~/dataset/nottingham-dataset/MIDI/melody/jigs117.mid
    0
    230
    ~/dataset/nottingham-dataset/MIDI/melody/jigs118.mid
    0
    166
    ~/dataset/nottingham-dataset/MIDI/melody/jigs119.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/jigs12.mid
    0
    112
    ~/dataset/nottingham-dataset/MIDI/melody/jigs120.mid
    0
    131
    ~/dataset/nottingham-dataset/MIDI/melody/jigs121.mid
    0
    264
    ~/dataset/nottingham-dataset/MIDI/melody/jigs122.mid
    0
    176
    ~/dataset/nottingham-dataset/MIDI/melody/jigs123.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/jigs124.mid
    0
    146
    ~/dataset/nottingham-dataset/MIDI/melody/jigs125.mid
    0
    452
    ~/dataset/nottingham-dataset/MIDI/melody/jigs126.mid
    0
    142
    ~/dataset/nottingham-dataset/MIDI/melody/jigs127.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/jigs128.mid
    0
    147
    ~/dataset/nottingham-dataset/MIDI/melody/jigs129.mid
    0
    280
    ~/dataset/nottingham-dataset/MIDI/melody/jigs13.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/jigs130.mid
    0
    400
    ~/dataset/nottingham-dataset/MIDI/melody/jigs131.mid
    0
    186
    ~/dataset/nottingham-dataset/MIDI/melody/jigs132.mid
    0
    132
    ~/dataset/nottingham-dataset/MIDI/melody/jigs133.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/jigs134.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/jigs135.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/jigs136.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/jigs137.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/jigs138.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/jigs139.mid
    0
    173
    ~/dataset/nottingham-dataset/MIDI/melody/jigs14.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/jigs140.mid
    0
    238
    ~/dataset/nottingham-dataset/MIDI/melody/jigs141.mid
    0
    190
    ~/dataset/nottingham-dataset/MIDI/melody/jigs142.mid
    0
    192
    ~/dataset/nottingham-dataset/MIDI/melody/jigs143.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/jigs144.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/jigs145.mid
    0
    169
    ~/dataset/nottingham-dataset/MIDI/melody/jigs146.mid
    0
    206
    ~/dataset/nottingham-dataset/MIDI/melody/jigs147.mid
    0
    172
    ~/dataset/nottingham-dataset/MIDI/melody/jigs148.mid
    0
    56
    ~/dataset/nottingham-dataset/MIDI/melody/jigs149.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs15.mid
    0
    138
    ~/dataset/nottingham-dataset/MIDI/melody/jigs150.mid
    0
    157
    ~/dataset/nottingham-dataset/MIDI/melody/jigs151.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs152.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/jigs153.mid
    0
    149
    ~/dataset/nottingham-dataset/MIDI/melody/jigs154.mid
    0
    140
    ~/dataset/nottingham-dataset/MIDI/melody/jigs155.mid
    0
    143
    ~/dataset/nottingham-dataset/MIDI/melody/jigs156.mid
    0
    143
    ~/dataset/nottingham-dataset/MIDI/melody/jigs157.mid
    0
    176
    ~/dataset/nottingham-dataset/MIDI/melody/jigs158.mid
    0
    188
    ~/dataset/nottingham-dataset/MIDI/melody/jigs159.mid
    0
    132
    ~/dataset/nottingham-dataset/MIDI/melody/jigs16.mid
    0
    76
    ~/dataset/nottingham-dataset/MIDI/melody/jigs160.mid
    0
    252
    ~/dataset/nottingham-dataset/MIDI/melody/jigs161.mid
    0
    128
    ~/dataset/nottingham-dataset/MIDI/melody/jigs162.mid
    0
    204
    ~/dataset/nottingham-dataset/MIDI/melody/jigs163.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/jigs164.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/jigs165.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/jigs166.mid
    0
    164
    ~/dataset/nottingham-dataset/MIDI/melody/jigs167.mid
    0
    170
    ~/dataset/nottingham-dataset/MIDI/melody/jigs168.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/jigs169.mid
    0
    368
    ~/dataset/nottingham-dataset/MIDI/melody/jigs17.mid
    0
    222
    ~/dataset/nottingham-dataset/MIDI/melody/jigs170.mid
    0
    140
    ~/dataset/nottingham-dataset/MIDI/melody/jigs171.mid
    0
    171
    ~/dataset/nottingham-dataset/MIDI/melody/jigs172.mid
    0
    128
    ~/dataset/nottingham-dataset/MIDI/melody/jigs173.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/jigs174.mid
    0
    175
    ~/dataset/nottingham-dataset/MIDI/melody/jigs175.mid
    0
    136
    ~/dataset/nottingham-dataset/MIDI/melody/jigs176.mid
    0
    323
    ~/dataset/nottingham-dataset/MIDI/melody/jigs177.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/jigs178.mid
    0
    172
    ~/dataset/nottingham-dataset/MIDI/melody/jigs179.mid
    0
    136
    ~/dataset/nottingham-dataset/MIDI/melody/jigs18.mid
    0
    168
    ~/dataset/nottingham-dataset/MIDI/melody/jigs180.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/jigs181.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/jigs182.mid
    0
    176
    ~/dataset/nottingham-dataset/MIDI/melody/jigs183.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/jigs184.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/jigs185.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/jigs186.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/jigs187.mid
    0
    552
    ~/dataset/nottingham-dataset/MIDI/melody/jigs188.mid
    0
    125
    ~/dataset/nottingham-dataset/MIDI/melody/jigs189.mid
    0
    532
    ~/dataset/nottingham-dataset/MIDI/melody/jigs19.mid
    0
    164
    ~/dataset/nottingham-dataset/MIDI/melody/jigs190.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/jigs191.mid
    0
    272
    ~/dataset/nottingham-dataset/MIDI/melody/jigs192.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/jigs193.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/jigs194.mid
    0
    164
    ~/dataset/nottingham-dataset/MIDI/melody/jigs195.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs196.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/jigs197.mid
    46
    173
    ~/dataset/nottingham-dataset/MIDI/melody/jigs198.mid
    1
    132
    ~/dataset/nottingham-dataset/MIDI/melody/jigs199.mid
    0
    278
    ~/dataset/nottingham-dataset/MIDI/melody/jigs2.mid
    0
    452
    ~/dataset/nottingham-dataset/MIDI/melody/jigs20.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/jigs200.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/jigs201.mid
    0
    220
    ~/dataset/nottingham-dataset/MIDI/melody/jigs202.mid
    0
    146
    ~/dataset/nottingham-dataset/MIDI/melody/jigs203.mid
    0
    277
    ~/dataset/nottingham-dataset/MIDI/melody/jigs204.mid
    0
    250
    ~/dataset/nottingham-dataset/MIDI/melody/jigs205.mid
    0
    140
    ~/dataset/nottingham-dataset/MIDI/melody/jigs206.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/jigs207.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs208.mid
    0
    472
    ~/dataset/nottingham-dataset/MIDI/melody/jigs209.mid
    0
    480
    ~/dataset/nottingham-dataset/MIDI/melody/jigs21.mid
    0
    164
    ~/dataset/nottingham-dataset/MIDI/melody/jigs210.mid
    0
    64
    ~/dataset/nottingham-dataset/MIDI/melody/jigs211.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/jigs212.mid
    0
    153
    ~/dataset/nottingham-dataset/MIDI/melody/jigs213.mid
    0
    302
    ~/dataset/nottingham-dataset/MIDI/melody/jigs214.mid
    0
    91
    ~/dataset/nottingham-dataset/MIDI/melody/jigs215.mid
    0
    64
    ~/dataset/nottingham-dataset/MIDI/melody/jigs216.mid
    0
    76
    ~/dataset/nottingham-dataset/MIDI/melody/jigs217.mid
    0
    67
    ~/dataset/nottingham-dataset/MIDI/melody/jigs218.mid
    0
    60
    ~/dataset/nottingham-dataset/MIDI/melody/jigs219.mid
    0
    66
    ~/dataset/nottingham-dataset/MIDI/melody/jigs22.mid
    0
    528
    ~/dataset/nottingham-dataset/MIDI/melody/jigs220.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/jigs221.mid
    0
    276
    ~/dataset/nottingham-dataset/MIDI/melody/jigs222.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs223.mid
    0
    451
    ~/dataset/nottingham-dataset/MIDI/melody/jigs224.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/jigs225.mid
    0
    301
    ~/dataset/nottingham-dataset/MIDI/melody/jigs226.mid
    0
    164
    ~/dataset/nottingham-dataset/MIDI/melody/jigs227.mid
    0
    292
    ~/dataset/nottingham-dataset/MIDI/melody/jigs228.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/jigs229.mid
    0
    146
    ~/dataset/nottingham-dataset/MIDI/melody/jigs23.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/jigs230.mid
    0
    188
    ~/dataset/nottingham-dataset/MIDI/melody/jigs231.mid
    0
    186
    ~/dataset/nottingham-dataset/MIDI/melody/jigs232.mid
    0
    166
    ~/dataset/nottingham-dataset/MIDI/melody/jigs233.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/jigs234.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/jigs235.mid
    0
    322
    ~/dataset/nottingham-dataset/MIDI/melody/jigs236.mid
    0
    216
    ~/dataset/nottingham-dataset/MIDI/melody/jigs237.mid
    0
    82
    ~/dataset/nottingham-dataset/MIDI/melody/jigs238.mid
    0
    146
    ~/dataset/nottingham-dataset/MIDI/melody/jigs239.mid
    0
    109
    ~/dataset/nottingham-dataset/MIDI/melody/jigs24.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/jigs240.mid
    0
    149
    ~/dataset/nottingham-dataset/MIDI/melody/jigs241.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/jigs242.mid
    0
    190
    ~/dataset/nottingham-dataset/MIDI/melody/jigs243.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/jigs244.mid
    0
    620
    ~/dataset/nottingham-dataset/MIDI/melody/jigs245.mid
    0
    142
    ~/dataset/nottingham-dataset/MIDI/melody/jigs246.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/jigs247.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/jigs248.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/jigs249.mid
    0
    145
    ~/dataset/nottingham-dataset/MIDI/melody/jigs25.mid
    0
    524
    ~/dataset/nottingham-dataset/MIDI/melody/jigs250.mid
    0
    114
    ~/dataset/nottingham-dataset/MIDI/melody/jigs251.mid
    0
    468
    ~/dataset/nottingham-dataset/MIDI/melody/jigs252.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/jigs253.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/jigs254.mid
    0
    138
    ~/dataset/nottingham-dataset/MIDI/melody/jigs255.mid
    0
    186
    ~/dataset/nottingham-dataset/MIDI/melody/jigs256.mid
    0
    289
    ~/dataset/nottingham-dataset/MIDI/melody/jigs257.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs258.mid
    0
    169
    ~/dataset/nottingham-dataset/MIDI/melody/jigs259.mid
    0
    161
    ~/dataset/nottingham-dataset/MIDI/melody/jigs26.mid
    0
    114
    ~/dataset/nottingham-dataset/MIDI/melody/jigs260.mid
    0
    155
    ~/dataset/nottingham-dataset/MIDI/melody/jigs261.mid
    0
    118
    ~/dataset/nottingham-dataset/MIDI/melody/jigs262.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/jigs263.mid
    0
    176
    ~/dataset/nottingham-dataset/MIDI/melody/jigs264.mid
    0
    170
    ~/dataset/nottingham-dataset/MIDI/melody/jigs265.mid
    0
    146
    ~/dataset/nottingham-dataset/MIDI/melody/jigs266.mid
    0
    122
    ~/dataset/nottingham-dataset/MIDI/melody/jigs267.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/jigs268.mid
    0
    172
    ~/dataset/nottingham-dataset/MIDI/melody/jigs269.mid
    0
    159
    ~/dataset/nottingham-dataset/MIDI/melody/jigs27.mid
    0
    322
    ~/dataset/nottingham-dataset/MIDI/melody/jigs270.mid
    0
    155
    ~/dataset/nottingham-dataset/MIDI/melody/jigs271.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/jigs272.mid
    0
    165
    ~/dataset/nottingham-dataset/MIDI/melody/jigs273.mid
    0
    165
    ~/dataset/nottingham-dataset/MIDI/melody/jigs274.mid
    0
    146
    ~/dataset/nottingham-dataset/MIDI/melody/jigs275.mid
    0
    278
    ~/dataset/nottingham-dataset/MIDI/melody/jigs276.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/jigs277.mid
    0
    171
    ~/dataset/nottingham-dataset/MIDI/melody/jigs278.mid
    0
    97
    ~/dataset/nottingham-dataset/MIDI/melody/jigs279.mid
    0
    384
    ~/dataset/nottingham-dataset/MIDI/melody/jigs28.mid
    0
    326
    ~/dataset/nottingham-dataset/MIDI/melody/jigs280.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs281.mid
    0
    252
    ~/dataset/nottingham-dataset/MIDI/melody/jigs282.mid
    0
    138
    ~/dataset/nottingham-dataset/MIDI/melody/jigs283.mid
    0
    138
    ~/dataset/nottingham-dataset/MIDI/melody/jigs284.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/jigs285.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/jigs286.mid
    0
    142
    ~/dataset/nottingham-dataset/MIDI/melody/jigs287.mid
    0
    185
    ~/dataset/nottingham-dataset/MIDI/melody/jigs288.mid
    0
    142
    ~/dataset/nottingham-dataset/MIDI/melody/jigs289.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/jigs29.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/jigs290.mid
    0
    146
    ~/dataset/nottingham-dataset/MIDI/melody/jigs291.mid
    0
    170
    ~/dataset/nottingham-dataset/MIDI/melody/jigs292.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/jigs293.mid
    0
    235
    ~/dataset/nottingham-dataset/MIDI/melody/jigs294.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs295.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/jigs296.mid
    0
    132
    ~/dataset/nottingham-dataset/MIDI/melody/jigs297.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/jigs298.mid
    0
    318
    ~/dataset/nottingham-dataset/MIDI/melody/jigs299.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/jigs3.mid
    0
    268
    ~/dataset/nottingham-dataset/MIDI/melody/jigs30.mid
    0
    183
    ~/dataset/nottingham-dataset/MIDI/melody/jigs300.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/jigs301.mid
    0
    232
    ~/dataset/nottingham-dataset/MIDI/melody/jigs302.mid
    0
    160
    ~/dataset/nottingham-dataset/MIDI/melody/jigs303.mid
    0
    139
    ~/dataset/nottingham-dataset/MIDI/melody/jigs304.mid
    0
    101
    ~/dataset/nottingham-dataset/MIDI/melody/jigs305.mid
    0
    142
    ~/dataset/nottingham-dataset/MIDI/melody/jigs306.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/jigs307.mid
    0
    440
    ~/dataset/nottingham-dataset/MIDI/melody/jigs308.mid
    0
    175
    ~/dataset/nottingham-dataset/MIDI/melody/jigs309.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/jigs31.mid
    0
    179
    ~/dataset/nottingham-dataset/MIDI/melody/jigs310.mid
    0
    272
    ~/dataset/nottingham-dataset/MIDI/melody/jigs311.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/jigs312.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/jigs313.mid
    0
    166
    ~/dataset/nottingham-dataset/MIDI/melody/jigs314.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/jigs315.mid
    0
    140
    ~/dataset/nottingham-dataset/MIDI/melody/jigs316.mid
    0
    227
    ~/dataset/nottingham-dataset/MIDI/melody/jigs317.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/jigs318.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/jigs319.mid
    0
    198
    ~/dataset/nottingham-dataset/MIDI/melody/jigs32.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/jigs320.mid
    0
    185
    ~/dataset/nottingham-dataset/MIDI/melody/jigs321.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/jigs322.mid
    0
    220
    ~/dataset/nottingham-dataset/MIDI/melody/jigs323.mid
    0
    122
    ~/dataset/nottingham-dataset/MIDI/melody/jigs324.mid
    0
    109
    ~/dataset/nottingham-dataset/MIDI/melody/jigs325.mid
    0
    604
    ~/dataset/nottingham-dataset/MIDI/melody/jigs326.mid
    0
    131
    ~/dataset/nottingham-dataset/MIDI/melody/jigs327.mid
    0
    128
    ~/dataset/nottingham-dataset/MIDI/melody/jigs328.mid
    0
    188
    ~/dataset/nottingham-dataset/MIDI/melody/jigs329.mid
    0
    416
    ~/dataset/nottingham-dataset/MIDI/melody/jigs33.mid
    0
    169
    ~/dataset/nottingham-dataset/MIDI/melody/jigs330.mid
    0
    160
    ~/dataset/nottingham-dataset/MIDI/melody/jigs331.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs332.mid
    0
    134
    ~/dataset/nottingham-dataset/MIDI/melody/jigs333.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/jigs334.mid
    0
    160
    ~/dataset/nottingham-dataset/MIDI/melody/jigs335.mid
    0
    230
    ~/dataset/nottingham-dataset/MIDI/melody/jigs336.mid
    0
    181
    ~/dataset/nottingham-dataset/MIDI/melody/jigs337.mid
    0
    246
    ~/dataset/nottingham-dataset/MIDI/melody/jigs338.mid
    0
    507
    ~/dataset/nottingham-dataset/MIDI/melody/jigs339.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/jigs34.mid
    0
    168
    ~/dataset/nottingham-dataset/MIDI/melody/jigs340.mid
    0
    117
    ~/dataset/nottingham-dataset/MIDI/melody/jigs35.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/jigs36.mid
    0
    160
    ~/dataset/nottingham-dataset/MIDI/melody/jigs37.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/jigs38.mid
    0
    234
    ~/dataset/nottingham-dataset/MIDI/melody/jigs39.mid
    0
    143
    ~/dataset/nottingham-dataset/MIDI/melody/jigs4.mid
    0
    271
    ~/dataset/nottingham-dataset/MIDI/melody/jigs40.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/jigs41.mid
    0
    176
    ~/dataset/nottingham-dataset/MIDI/melody/jigs42.mid
    0
    142
    ~/dataset/nottingham-dataset/MIDI/melody/jigs43.mid
    0
    164
    ~/dataset/nottingham-dataset/MIDI/melody/jigs44.mid
    0
    147
    ~/dataset/nottingham-dataset/MIDI/melody/jigs45.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/jigs46.mid
    0
    226
    ~/dataset/nottingham-dataset/MIDI/melody/jigs47.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/jigs48.mid
    0
    192
    ~/dataset/nottingham-dataset/MIDI/melody/jigs49.mid
    0
    185
    ~/dataset/nottingham-dataset/MIDI/melody/jigs5.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/jigs50.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/jigs51.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/jigs52.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/jigs53.mid
    0
    170
    ~/dataset/nottingham-dataset/MIDI/melody/jigs54.mid
    0
    186
    ~/dataset/nottingham-dataset/MIDI/melody/jigs55.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/jigs56.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/jigs57.mid
    0
    137
    ~/dataset/nottingham-dataset/MIDI/melody/jigs58.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/jigs59.mid
    0
    135
    ~/dataset/nottingham-dataset/MIDI/melody/jigs6.mid
    0
    449
    ~/dataset/nottingham-dataset/MIDI/melody/jigs60.mid
    0
    68
    ~/dataset/nottingham-dataset/MIDI/melody/jigs61.mid
    0
    145
    ~/dataset/nottingham-dataset/MIDI/melody/jigs62.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/jigs63.mid
    0
    197
    ~/dataset/nottingham-dataset/MIDI/melody/jigs64.mid
    0
    190
    ~/dataset/nottingham-dataset/MIDI/melody/jigs65.mid
    0
    323
    ~/dataset/nottingham-dataset/MIDI/melody/jigs66.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/jigs67.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/jigs68.mid
    0
    168
    ~/dataset/nottingham-dataset/MIDI/melody/jigs69.mid
    0
    168
    ~/dataset/nottingham-dataset/MIDI/melody/jigs7.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/jigs70.mid
    0
    176
    ~/dataset/nottingham-dataset/MIDI/melody/jigs71.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/jigs72.mid
    0
    134
    ~/dataset/nottingham-dataset/MIDI/melody/jigs73.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/jigs74.mid
    0
    276
    ~/dataset/nottingham-dataset/MIDI/melody/jigs75.mid
    0
    468
    ~/dataset/nottingham-dataset/MIDI/melody/jigs76.mid
    0
    140
    ~/dataset/nottingham-dataset/MIDI/melody/jigs77.mid
    0
    234
    ~/dataset/nottingham-dataset/MIDI/melody/jigs78.mid
    0
    136
    ~/dataset/nottingham-dataset/MIDI/melody/jigs79.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/jigs8.mid
    0
    732
    ~/dataset/nottingham-dataset/MIDI/melody/jigs80.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/jigs81.mid
    0
    938
    ~/dataset/nottingham-dataset/MIDI/melody/jigs82.mid
    0
    160
    ~/dataset/nottingham-dataset/MIDI/melody/jigs83.mid
    0
    122
    ~/dataset/nottingham-dataset/MIDI/melody/jigs84.mid
    0
    112
    ~/dataset/nottingham-dataset/MIDI/melody/jigs85.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/jigs86.mid
    0
    149
    ~/dataset/nottingham-dataset/MIDI/melody/jigs87.mid
    0
    212
    ~/dataset/nottingham-dataset/MIDI/melody/jigs88.mid
    0
    168
    ~/dataset/nottingham-dataset/MIDI/melody/jigs89.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/jigs9.mid
    0
    226
    ~/dataset/nottingham-dataset/MIDI/melody/jigs90.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/jigs91.mid
    0
    73
    ~/dataset/nottingham-dataset/MIDI/melody/jigs92.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/jigs93.mid
    0
    134
    ~/dataset/nottingham-dataset/MIDI/melody/jigs94.mid
    0
    145
    ~/dataset/nottingham-dataset/MIDI/melody/jigs95.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/jigs96.mid
    0
    236
    ~/dataset/nottingham-dataset/MIDI/melody/jigs97.mid
    0
    202
    ~/dataset/nottingham-dataset/MIDI/melody/jigs98.mid
    0
    166
    ~/dataset/nottingham-dataset/MIDI/melody/jigs99.mid
    0
    124
    ~/dataset/nottingham-dataset/MIDI/melody/morris1.mid
    0
    190
    ~/dataset/nottingham-dataset/MIDI/melody/morris10.mid
    0
    682
    ~/dataset/nottingham-dataset/MIDI/melody/morris11.mid
    0
    528
    ~/dataset/nottingham-dataset/MIDI/melody/morris12.mid
    0
    50
    ~/dataset/nottingham-dataset/MIDI/melody/morris13.mid
    0
    122
    ~/dataset/nottingham-dataset/MIDI/melody/morris14.mid
    0
    1104
    ~/dataset/nottingham-dataset/MIDI/melody/morris15.mid
    0
    228
    ~/dataset/nottingham-dataset/MIDI/melody/morris16.mid
    0
    832
    ~/dataset/nottingham-dataset/MIDI/melody/morris17.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/morris18.mid
    0
    86
    ~/dataset/nottingham-dataset/MIDI/melody/morris19.mid
    0
    28
    ~/dataset/nottingham-dataset/MIDI/melody/morris2.mid
    0
    289
    ~/dataset/nottingham-dataset/MIDI/melody/morris20.mid
    0
    374
    ~/dataset/nottingham-dataset/MIDI/melody/morris21.mid
    0
    673
    ~/dataset/nottingham-dataset/MIDI/melody/morris22.mid
    0
    262
    ~/dataset/nottingham-dataset/MIDI/melody/morris23.mid
    0
    157
    ~/dataset/nottingham-dataset/MIDI/melody/morris24.mid
    0
    367
    ~/dataset/nottingham-dataset/MIDI/melody/morris25.mid
    0
    57
    ~/dataset/nottingham-dataset/MIDI/melody/morris26.mid
    0
    121
    ~/dataset/nottingham-dataset/MIDI/melody/morris27.mid
    0
    66
    ~/dataset/nottingham-dataset/MIDI/melody/morris28.mid
    0
    121
    ~/dataset/nottingham-dataset/MIDI/melody/morris29.mid
    0
    50
    ~/dataset/nottingham-dataset/MIDI/melody/morris3.mid
    0
    697
    ~/dataset/nottingham-dataset/MIDI/melody/morris30.mid
    0
    475
    ~/dataset/nottingham-dataset/MIDI/melody/morris31.mid
    0
    1242
    ~/dataset/nottingham-dataset/MIDI/melody/morris4.mid
    0
    89
    ~/dataset/nottingham-dataset/MIDI/melody/morris5.mid
    0
    105
    ~/dataset/nottingham-dataset/MIDI/melody/morris6.mid
    0
    121
    ~/dataset/nottingham-dataset/MIDI/melody/morris7.mid
    0
    374
    ~/dataset/nottingham-dataset/MIDI/melody/morris8.mid
    0
    541
    ~/dataset/nottingham-dataset/MIDI/melody/morris9.mid
    0
    702
    ~/dataset/nottingham-dataset/MIDI/melody/playford1.mid
    0
    94
    ~/dataset/nottingham-dataset/MIDI/melody/playford10.mid
    0
    64
    ~/dataset/nottingham-dataset/MIDI/melody/playford11.mid
    0
    136
    ~/dataset/nottingham-dataset/MIDI/melody/playford12.mid
    0
    102
    ~/dataset/nottingham-dataset/MIDI/melody/playford13.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/playford14.mid
    0
    63
    ~/dataset/nottingham-dataset/MIDI/melody/playford15.mid
    0
    260
    ~/dataset/nottingham-dataset/MIDI/melody/playford2.mid
    0
    304
    ~/dataset/nottingham-dataset/MIDI/melody/playford3.mid
    0
    332
    ~/dataset/nottingham-dataset/MIDI/melody/playford4.mid
    0
    182
    ~/dataset/nottingham-dataset/MIDI/melody/playford5.mid
    0
    205
    ~/dataset/nottingham-dataset/MIDI/melody/playford6.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/playford7.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/playford8.mid
    0
    274
    ~/dataset/nottingham-dataset/MIDI/melody/playford9.mid
    0
    68
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c1.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c10.mid
    0
    142
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c11.mid
    0
    114
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c12.mid
    0
    226
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c13.mid
    0
    224
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c14.mid
    0
    134
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c15.mid
    0
    252
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c16.mid
    0
    182
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c17.mid
    0
    118
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c18.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c19.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c2.mid
    0
    109
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c20.mid
    0
    272
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c21.mid
    0
    194
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c22.mid
    0
    309
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c23.mid
    0
    199
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c24.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c25.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c26.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c27.mid
    0
    202
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c28.mid
    0
    232
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c29.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c3.mid
    0
    182
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c30.mid
    32
    197
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c31.mid
    0
    48
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c32.mid
    0
    62
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c33.mid
    0
    111
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c34.mid
    0
    140
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c35.mid
    0
    172
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c36.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c37.mid
    0
    121
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c38.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c39.mid
    0
    172
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c4.mid
    0
    215
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c40.mid
    0
    94
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c41.mid
    0
    102
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c42.mid
    0
    124
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c43.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c44.mid
    0
    210
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c45.mid
    0
    192
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c46.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c47.mid
    0
    346
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c48.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c49.mid
    0
    74
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c5.mid
    0
    136
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c50.mid
    0
    143
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c51.mid
    0
    73
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c52.mid
    0
    222
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c53.mid
    0
    92
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c54.mid
    0
    188
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c55.mid
    0
    222
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c56.mid
    0
    224
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c57.mid
    0
    198
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c58.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c59.mid
    0
    132
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c6.mid
    0
    98
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c60.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c61.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c62.mid
    0
    113
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c63.mid
    0
    472
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c64.mid
    0
    236
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c65.mid
    0
    318
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c66.mid
    40
    115
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c67.mid
    0
    168
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c68.mid
    0
    133
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c69.mid
    0
    370
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c7.mid
    0
    190
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c70.mid
    0
    43
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c71.mid
    0
    664
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c72.mid
    0
    310
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c73.mid
    0
    296
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c74.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c75.mid
    0
    230
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c76.mid
    0
    112
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c77.mid
    0
    206
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c78.mid
    0
    202
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c79.mid
    0
    296
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c8.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c80.mid
    0
    332
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c81.mid
    0
    102
    ~/dataset/nottingham-dataset/MIDI/melody/reelsa-c9.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g1.mid
    68
    356
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g10.mid
    0
    109
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g11.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g12.mid
    0
    258
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g13.mid
    0
    224
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g14.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g15.mid
    0
    312
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g16.mid
    0
    137
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g17.mid
    0
    234
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g18.mid
    0
    137
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g19.mid
    0
    179
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g2.mid
    0
    165
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g20.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g21.mid
    0
    139
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g22.mid
    0
    198
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g23.mid
    0
    216
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g24.mid
    0
    190
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g25.mid
    0
    222
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g26.mid
    0
    216
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g27.mid
    0
    121
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g28.mid
    20
    199
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g29.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g3.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g30.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g31.mid
    0
    102
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g32.mid
    0
    198
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g33.mid
    0
    89
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g34.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g35.mid
    0
    110
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g36.mid
    0
    134
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g37.mid
    0
    210
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g38.mid
    0
    177
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g39.mid
    0
    218
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g4.mid
    0
    232
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g40.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g41.mid
    42
    191
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g42.mid
    0
    240
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g43.mid
    0
    132
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g44.mid
    0
    108
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g45.mid
    0
    240
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g46.mid
    0
    236
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g47.mid
    0
    196
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g48.mid
    0
    186
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g49.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g5.mid
    0
    407
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g50.mid
    0
    214
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g51.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g52.mid
    0
    114
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g53.mid
    0
    228
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g54.mid
    0
    200
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g55.mid
    0
    224
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g56.mid
    0
    128
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g57.mid
    0
    87
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g58.mid
    0
    72
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g59.mid
    0
    287
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g6.mid
    0
    168
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g60.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g61.mid
    0
    199
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g62.mid
    0
    103
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g63.mid
    0
    95
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g64.mid
    0
    89
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g65.mid
    0
    226
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g66.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g67.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g68.mid
    0
    240
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g69.mid
    0
    168
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g7.mid
    0
    328
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g70.mid
    0
    214
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g71.mid
    0
    246
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g72.mid
    0
    138
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g73.mid
    0
    140
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g74.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g75.mid
    0
    162
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g76.mid
    0
    318
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g77.mid
    0
    322
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g78.mid
    0
    142
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g79.mid
    0
    186
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g8.mid
    0
    242
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g80.mid
    0
    247
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g81.mid
    0
    102
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g82.mid
    0
    186
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g83.mid
    0
    230
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g84.mid
    0
    198
    ~/dataset/nottingham-dataset/MIDI/melody/reelsd-g9.mid
    8
    174
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l1.mid
    0
    236
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l10.mid
    0
    143
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l11.mid
    0
    117
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l12.mid
    0
    100
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l13.mid
    0
    81
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l14.mid
    0
    46
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l15.mid
    0
    456
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l16.mid
    0
    190
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l17.mid
    0
    212
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l18.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l19.mid
    0
    288
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l2.mid
    0
    79
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l20.mid
    0
    75
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l21.mid
    0
    227
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l22.mid
    0
    164
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l23.mid
    0
    140
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l24.mid
    0
    202
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l25.mid
    0
    165
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l26.mid
    0
    124
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l27.mid
    0
    141
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l28.mid
    0
    346
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l29.mid
    0
    124
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l3.mid
    0
    106
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l30.mid
    0
    194
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l31.mid
    0
    214
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l32.mid
    0
    182
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l33.mid
    0
    149
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l34.mid
    0
    164
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l35.mid
    0
    322
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l36.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l37.mid
    0
    222
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l38.mid
    0
    112
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l39.mid
    0
    122
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l4.mid
    0
    103
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l40.mid
    0
    321
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l41.mid
    0
    160
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l42.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l43.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l44.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l45.mid
    0
    316
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l46.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l47.mid
    0
    198
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l5.mid
    0
    357
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l51.mid
    0
    191
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l52.mid
    0
    111
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l53.mid
    0
    95
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l54.mid
    0
    140
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l55.mid
    1
    79
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l56.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l57.mid
    4
    202
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l58.mid
    0
    176
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l59.mid
    0
    256
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l6.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l60.mid
    0
    274
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l61.mid
    0
    220
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l62.mid
    0
    90
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l63.mid
    0
    179
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l64.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l65.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l66.mid
    0
    209
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l67.mid
    0
    228
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l68.mid
    0
    237
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l69.mid
    0
    236
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l7.mid
    0
    91
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l70.mid
    0
    237
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l71.mid
    0
    236
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l72.mid
    0
    276
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l73.mid
    0
    218
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l74.mid
    0
    144
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l75.mid
    0
    413
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l76.mid
    0
    212
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l77.mid
    0
    242
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l78.mid
    0
    188
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l79.mid
    0
    402
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l8.mid
    0
    136
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l80.mid
    2
    331
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l81.mid
    0
    98
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l82.mid
    0
    149
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l83.mid
    0
    194
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l84.mid
    0
    114
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l85.mid
    0
    333
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l86.mid
    0
    115
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l87.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l88.mid
    14
    152
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l89.mid
    0
    288
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l9.mid
    0
    108
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l90.mid
    0
    137
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l91.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l92.mid
    0
    192
    ~/dataset/nottingham-dataset/MIDI/melody/reelsh-l93.mid
    0
    106
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q1.mid
    0
    110
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q10.mid
    0
    240
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q11.mid
    0
    234
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q12.mid
    0
    224
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q13.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q14.mid
    0
    97
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q15.mid
    0
    122
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q16.mid
    0
    111
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q17.mid
    0
    314
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q18.mid
    0
    238
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q19.mid
    0
    143
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q2.mid
    0
    77
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q20.mid
    0
    123
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q21.mid
    0
    77
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q22.mid
    0
    226
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q23.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q24.mid
    0
    309
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q25.mid
    0
    140
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q26.mid
    0
    346
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q27.mid
    0
    208
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q28.mid
    0
    244
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q29.mid
    0
    98
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q3.mid
    0
    218
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q30.mid
    0
    186
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q31.mid
    0
    218
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q32.mid
    0
    194
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q33.mid
    0
    170
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q34.mid
    0
    203
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q35.mid
    0
    58
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q36.mid
    0
    246
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q37.mid
    0
    121
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q38.mid
    0
    258
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q39.mid
    0
    200
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q4.mid
    0
    240
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q40.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q41.mid
    0
    223
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q42.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q43.mid
    0
    111
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q44.mid
    0
    218
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q45.mid
    0
    62
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q46.mid
    0
    197
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q47.mid
    0
    54
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q48.mid
    0
    78
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q49.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q5.mid
    0
    112
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q50.mid
    0
    98
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q51.mid
    0
    118
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q52.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q53.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q54.mid
    0
    166
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q55.mid
    0
    108
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q56.mid
    0
    828
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q57.mid
    0
    218
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q58.mid
    0
    91
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q59.mid
    0
    67
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q6.mid
    0
    111
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q60.mid
    0
    149
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q61.mid
    0
    134
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q62.mid
    0
    149
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q63.mid
    0
    234
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q64.mid
    0
    124
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q65.mid
    0
    142
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q66.mid
    0
    182
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q67.mid
    0
    177
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q68.mid
    0
    456
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q69.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q7.mid
    0
    92
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q70.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q71.mid
    0
    170
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q72.mid
    0
    214
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q73.mid
    0
    119
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q74.mid
    0
    207
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q75.mid
    0
    227
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q76.mid
    0
    198
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q77.mid
    0
    192
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q78.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q79.mid
    0
    214
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q8.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q80.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/reelsm-q9.mid
    0
    194
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t1.mid
    0
    207
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t10.mid
    6
    182
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t11.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t12.mid
    0
    256
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t13.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t14.mid
    0
    126
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t15.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t16.mid
    0
    76
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t17.mid
    0
    466
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t18.mid
    0
    128
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t19.mid
    0
    108
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t2.mid
    0
    286
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t20.mid
    0
    178
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t21.mid
    0
    222
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t22.mid
    0
    230
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t23.mid
    0
    200
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t24.mid
    0
    505
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t25.mid
    0
    132
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t26.mid
    0
    204
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t27.mid
    0
    132
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t28.mid
    0
    208
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t29.mid
    0
    36
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t3.mid
    0
    195
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t30.mid
    0
    184
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t31.mid
    0
    216
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t32.mid
    0
    340
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t33.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t34.mid
    0
    165
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t35.mid
    0
    186
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t36.mid
    0
    218
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t37.mid
    0
    216
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t38.mid
    0
    250
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t39.mid
    0
    218
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t4.mid
    0
    224
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t40.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t41.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t42.mid
    0
    188
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t43.mid
    0
    620
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t44.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t45.mid
    0
    109
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t46.mid
    0
    394
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t47.mid
    0
    225
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t48.mid
    0
    218
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t49.mid
    0
    216
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t5.mid
    0
    136
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t50.mid
    0
    213
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t51.mid
    0
    244
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t52.mid
    0
    223
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t53.mid
    0
    214
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t54.mid
    0
    228
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t55.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t56.mid
    0
    194
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t57.mid
    0
    204
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t58.mid
    0
    110
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t59.mid
    0
    116
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t6.mid
    0
    81
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t60.mid
    0
    73
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t61.mid
    0
    246
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t62.mid
    0
    240
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t63.mid
    0
    110
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t64.mid
    0
    248
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t65.mid
    0
    206
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t66.mid
    0
    212
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t67.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t68.mid
    0
    196
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t69.mid
    0
    236
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t7.mid
    0
    212
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t70.mid
    0
    232
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t71.mid
    0
    702
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t72.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t73.mid
    0
    192
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t74.mid
    0
    241
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t75.mid
    0
    240
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t76.mid
    0
    304
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t77.mid
    0
    131
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t78.mid
    0
    255
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t79.mid
    0
    154
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t8.mid
    0
    726
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t80.mid
    0
    146
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t81.mid
    0
    210
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t82.mid
    0
    242
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t83.mid
    0
    208
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t84.mid
    0
    238
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t85.mid
    0
    152
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t86.mid
    0
    134
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t87.mid
    0
    236
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t88.mid
    0
    114
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t89.mid
    0
    77
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t9.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t90.mid
    0
    230
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t91.mid
    0
    182
    ~/dataset/nottingham-dataset/MIDI/melody/reelsr-t92.mid
    0
    123
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z1.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z10.mid
    0
    199
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z11.mid
    0
    296
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z12.mid
    0
    290
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z13.mid
    0
    303
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z14.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z15.mid
    2
    154
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z16.mid
    0
    95
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z17.mid
    0
    230
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z18.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z19.mid
    0
    160
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z2.mid
    0
    146
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z20.mid
    0
    174
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z21.mid
    0
    186
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z22.mid
    0
    237
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z23.mid
    0
    208
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z24.mid
    0
    357
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z25.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z26.mid
    0
    150
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z27.mid
    0
    114
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z28.mid
    0
    792
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z29.mid
    0
    202
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z3.mid
    6
    168
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z30.mid
    0
    185
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z31.mid
    0
    158
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z32.mid
    0
    111
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z33.mid
    0
    259
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z34.mid
    0
    206
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z4.mid
    0
    172
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z5.mid
    0
    244
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z6.mid
    4
    164
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z7.mid
    0
    236
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z8.mid
    0
    288
    ~/dataset/nottingham-dataset/MIDI/melody/reelsu-z9.mid
    0
    164
    ~/dataset/nottingham-dataset/MIDI/melody/slip1.mid
    0
    136
    ~/dataset/nottingham-dataset/MIDI/melody/slip10.mid
    0
    161
    ~/dataset/nottingham-dataset/MIDI/melody/slip11.mid
    0
    51
    ~/dataset/nottingham-dataset/MIDI/melody/slip2.mid
    0
    92
    ~/dataset/nottingham-dataset/MIDI/melody/slip3.mid
    0
    188
    ~/dataset/nottingham-dataset/MIDI/melody/slip4.mid
    0
    120
    ~/dataset/nottingham-dataset/MIDI/melody/slip5.mid
    0
    415
    ~/dataset/nottingham-dataset/MIDI/melody/slip6.mid
    0
    210
    ~/dataset/nottingham-dataset/MIDI/melody/slip7.mid
    0
    336
    ~/dataset/nottingham-dataset/MIDI/melody/slip8.mid
    0
    113
    ~/dataset/nottingham-dataset/MIDI/melody/slip9.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes1.mid
    0
    186
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes10.mid
    0
    90
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes11.mid
    0
    122
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes12.mid
    0
    101
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes13.mid
    0
    169
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes14.mid
    0
    91
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes15.mid
    0
    192
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes16.mid
    0
    148
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes17.mid
    0
    85
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes18.mid
    0
    210
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes19.mid
    0
    346
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes2.mid
    0
    141
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes20.mid
    0
    206
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes21.mid
    0
    61
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes22.mid
    0
    98
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes23.mid
    0
    50
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes24.mid
    0
    139
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes25.mid
    0
    81
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes26.mid
    0
    97
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes27.mid
    0
    100
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes28.mid
    0
    156
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes29.mid
    0
    108
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes3.mid
    0
    88
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes30.mid
    0
    229
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes31.mid
    0
    94
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes32.mid
    0
    63
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes33.mid
    0
    108
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes34.mid
    0
    70
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes35.mid
    0
    179
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes36.mid
    0
    78
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes37.mid
    0
    76
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes38.mid
    0
    180
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes39.mid
    0
    83
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes4.mid
    0
    31
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes40.mid
    0
    114
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes41.mid
    0
    86
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes42.mid
    0
    130
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes43.mid
    0
    192
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes44.mid
    0
    88
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes45.mid
    0
    73
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes46.mid
    0
    83
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes47.mid
    0
    78
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes48.mid
    0
    94
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes49.mid
    0
    96
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes5.mid
    0
    70
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes50.mid
    0
    90
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes51.mid
    0
    129
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes52.mid
    0
    80
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes6.mid
    0
    170
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes7.mid
    0
    172
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes8.mid
    0
    92
    ~/dataset/nottingham-dataset/MIDI/melody/waltzes9.mid
    0
    164
    ~/dataset/nottingham-dataset/MIDI/melody/xmas1.mid
    0
    48
    ~/dataset/nottingham-dataset/MIDI/melody/xmas10.mid
    0
    110
    ~/dataset/nottingham-dataset/MIDI/melody/xmas11.mid
    0
    76
    ~/dataset/nottingham-dataset/MIDI/melody/xmas12.mid
    0
    81
    ~/dataset/nottingham-dataset/MIDI/melody/xmas13.mid
    0
    62
    ~/dataset/nottingham-dataset/MIDI/melody/xmas2.mid
    0
    108
    ~/dataset/nottingham-dataset/MIDI/melody/xmas3.mid
    0
    54
    ~/dataset/nottingham-dataset/MIDI/melody/xmas4.mid
    0
    68
    ~/dataset/nottingham-dataset/MIDI/melody/xmas5.mid
    0
    118
    ~/dataset/nottingham-dataset/MIDI/melody/xmas6.mid
    0
    37
    ~/dataset/nottingham-dataset/MIDI/melody/xmas7.mid
    0
    17
    ~/dataset/nottingham-dataset/MIDI/melody/xmas8.mid
    0
    119
    ~/dataset/nottingham-dataset/MIDI/melody/xmas9.mid
    0
    114


전처리해놓은 데이터를 파일로 저장해놓으면 향후에 매번 전처리 작업을 다시 하지 않고 데이터를 불러오기만 할 수 있기 때문에, 적당한 폴더에 파일로 저장해놓겠습니다.


```python
preprocessed_dir = "./preprocessed_data/"
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

with open(preprocessed_dir + "mel_arr_list.p", "wb") as fp:   
    pickle.dump(mel_arr_list, fp)
```

우리가 전처리한 데이터를 한번 확인해봅시다. 

리스트로 만들어 놓았기 때문에 첫번째 곡의(인덱스 0번) 정보를 한번 확인해보겠습니다.


```python
mel_arr_list[0]
```




    [{'note': 76, 'offset': 2.0},
     {'note': 74, 'offset': 3.0},
     {'note': 71, 'offset': 5.0},
     {'note': 69, 'offset': 6.0},
     {'note': 71, 'offset': 7.5},
     {'note': 72, 'offset': 8.0},
     {'note': 71, 'offset': 9.0},
     {'note': 67, 'offset': 11.0},
     {'note': 69, 'offset': 12.0},
     {'note': 76, 'offset': 14.0},
     {'note': 74, 'offset': 15.0},
     {'note': 71, 'offset': 17.0},
     {'note': 69, 'offset': 18.0},
     {'note': 71, 'offset': 19.5},
     {'note': 72, 'offset': 20.0},
     {'note': 71, 'offset': 21.0},
     {'note': 65, 'offset': 22.0},
     {'note': 67, 'offset': 23.0},
     {'note': 76, 'offset': 25.0},
     {'note': 74, 'offset': 26.0},
     {'note': 71, 'offset': 28.0},
     {'note': 69, 'offset': 29.0},
     {'note': 71, 'offset': 30.5},
     {'note': 72, 'offset': 31.0},
     {'note': 71, 'offset': 32.0},
     {'note': 67, 'offset': 34.0},
     {'note': 69, 'offset': 35.0},
     {'note': 76, 'offset': 37.0},
     {'note': 74, 'offset': 38.0},
     {'note': 71, 'offset': 40.0},
     {'note': 69, 'offset': 41.0},
     {'note': 71, 'offset': 42.5},
     {'note': 72, 'offset': 43.0},
     {'note': 71, 'offset': 44.0},
     {'note': 65, 'offset': 45.0},
     {'note': 67, 'offset': 46.0},
     {'note': 76, 'offset': 48.0},
     {'note': 79, 'offset': 49.0},
     {'note': 76, 'offset': 51.0},
     {'note': 77, 'offset': 52.0},
     {'note': 74, 'offset': 54.0},
     {'note': 72, 'offset': 55.0},
     {'note': 69, 'offset': 57.0},
     {'note': 65, 'offset': 58.0},
     {'note': 76, 'offset': 60.0},
     {'note': 79, 'offset': 61.0},
     {'note': 76, 'offset': 63.0},
     {'note': 77, 'offset': 64.0},
     {'note': 74, 'offset': 66.0},
     {'note': 72, 'offset': 67.0},
     {'note': 69, 'offset': 68.0},
     {'note': 67, 'offset': 69.0},
     {'note': 76, 'offset': 71.0},
     {'note': 79, 'offset': 72.0},
     {'note': 76, 'offset': 74.0},
     {'note': 77, 'offset': 75.0},
     {'note': 74, 'offset': 77.0},
     {'note': 72, 'offset': 78.0},
     {'note': 69, 'offset': 80.0},
     {'note': 65, 'offset': 81.0},
     {'note': 76, 'offset': 83.0},
     {'note': 79, 'offset': 84.0},
     {'note': 76, 'offset': 86.0},
     {'note': 77, 'offset': 87.0},
     {'note': 74, 'offset': 89.0},
     {'note': 72, 'offset': 90.0},
     {'note': 69, 'offset': 91.0},
     {'note': 67, 'offset': 92.0}]



## 데이터 Preprocessing (2) - Convert into sequential melodic form.

우리가 원하는 것은 신경망으로 하여금 멜로디 시퀀스의 진행 특성을 학습하도록 하는 것입니다.

그리하여 새로운 멜로디 인풋이 들어왔을 때, 그 뒤에 이어지는 음의 시퀀스가 생성되도록 하는 것입니다.


우리가 사용할 RNN이라는 구조는 시간의 흐름을 반영한 네트워크 구조인데, 각 시간 스텝마다 하나의 인풋을 받아서 이전까지의 시간 스텝을 통과하면서 전달받아온 정보와 함께 프로세싱한 후 아웃풋을 생성하는 모델입니다. (이때, 다음 시간 스텝에 넘겨줄 정보도 생성한 후 전달합니다.)

이를 통해 어떤 순서에 따라 어떤 입력이 들어올 때 어떤 출력값을 내어 놓는지에 대한 논리를 학습하도록 되어 있습니다.

이를 멜로디 학습에 적용하기 위한 개요를 적어봅시다.

(1) 순서 : 멜로디는 기본적으로 순차적인 구조로 되어 있기 때문에 그대로 적용.


(2) 입/출력 시퀀스 : 

- 멜로디 정보에 대한 표현
  
  a. 개별 노트의 음높이(상대 높이) + 길이
  
  b. 이어지는 개별 노트들 사이의 변화량 (피치 변화량 + 박자 거리)
  
  c. 마디 단위의 멜로디 정보(예를 들어 한 마디를 32개의 박자로 쪼갠 후 음이 있는 곳에만 바이너리로 표현)
 
 
 - 한번에 얼만큼의 시간 스텝을 보고 다음 스텝을 생성할지
  
  a. Char-RNN 구조 : 한번에 한개의 인풋이 들어가고 아웃풋으로 바로 다음 타입스텝의 입력값을 생성
  
  b. seq-to-seq 구조 : 한번에 한 마디에 해당하는 노트 시퀀스가 인풋으로 들어가고, 아웃풋으로 다음 마디의 노트 시퀀스를 생성.





이번 튜토리얼에서는, 개별 노트들 사이의 변화량을 하나의 엘리먼트로 하는 시퀀스를 학습해보고자 합니다. 학습 구조는 하나의 인풋에 대해서 바로 다음 시간 스텝의 엘리먼트를 예측하는 Char-RNN 구조를 사용하겠습니다.

### 멜로디 커브 데이터 준비

1단계에서 준비한 노트 정보를 가지고 인접한 노트들 간의 차이 정보에 대한 시퀀셜 데이터를 준비해보겠습니다.


```python
def create_curve_seq(mel_arr):
    curve_seq = []
    for idx in range(1, len(mel_arr)):
        curr_p_diff = mel_arr[idx]['note'] - mel_arr[idx-1]['note']
        curr_t_diff = mel_arr[idx]['offset'] - mel_arr[idx-1]['offset']
        curve_seq.append((curr_p_diff, curr_t_diff))
    return curve_seq
```


```python
def create_longer_curve_seq(mel_arr):
    curve_seq = []
    for idx in range(2, len(mel_zrr)):
        curr_p_diff = [mel_arr[idx]['note'] - mel_arr[idx-1]['note'], mel_arr[idx-1]['note'] - mel_arr[idx-2]['note']]
        curr_p_diff = [mel_arr[idx]['offset'] - mel_arr[idx-1]['offset'], mel_arr[idx-1]['offset'] - mel_arr[idx-2]['offset']]
        curve_seq.append((curr_p_diff, curr_t_diff))
    return curve_seq

```


```python
curve_seq_list = []
for mel_arr in mel_arr_list:
    curve_seq_list.append(create_curve_seq(mel_arr))

print(len(curve_seq_list))
# print(curve_seq_list)
```

    1034



```python
# flatten
curve_corpus= sum(curve_seq_list, [])
print(len(curve_corpus))

# flattened seq
mel_seq = curve_corpus
# print(curve_corpus)
```

    195174


### One-hot encoding을 위한 딕셔너리 준비

신경망 학습을 잘하기 위해서는 보통 one-hot encoding을 해서 데이터를 준비합니다. 이를 위해 모든 유니크한 멜로디 커브 엘리먼트에 대한 딕셔너리를 구성하고, 각각에 인덱스를 부여합니다.


```python
# prepare the complete set, and pairs of data and indices
def get_corpus_data(curve_corpus):
    curve_corpus_set = set(curve_corpus) 
    val_indices = dict((v, i) for i, v in enumerate(curve_corpus_set))
    indices_val = dict((i, v) for i, v in enumerate(curve_corpus_set))

    return curve_corpus_set, val_indices, indices_val

```


```python
mel_set, mel_v_i, mel_i_v = get_corpus_data(curve_corpus)
```

이 딕셔너리의 사이즈를 확인해봅시다.


```python
vocab_size = len(mel_set)
print(vocab_size)
```

    422


이제, 데이터에 대한 기초 준비가 끝났습니다. 

학습 모델에 이 데이터를 어떤 구조로 넣어야 할지 알아봅시다.


### 모델의 인풋/아웃풋 구조 준비

우리가 사용할 Char-RNN구조는 일정 길이의 시퀀스 단위로 인풋을 처리합니다. 

즉, 일정 길이의 인풋 시퀀스를 받아서 시퀀스 내의 각 스텝이 다음 스텝의 인풋과 같은 엘리먼트를 아웃풋으로 출력하도록 학습되게 하는 것입니다.

Char-RNN은 한 스텝의 인풋 씩 받아서 다음 스텝을 생성하는 모델이지만, 학습할 때는 일정 길이의 시퀀스 단위로 학습을 하도록 합니다. 


* 사실 원래 개념상으로는 데이터 전체에 대해서 RNN이 backpropagation하면서 학습이 이루어져야 합니다.하지만 메모리의 문제 때문에 그렇게 할 수 없으므로 일정 길이의 묶음으로 나누어서 학습을 하기 위해서 이런 방식을 사용합니다. (물론, Sequence-to-sequence 모델에서는 우리의 목적에 따른 길이의 데이터 시퀀스로 학습을 하겠죠.)


따라서, 학습할 때에는 적당한 길이로 잘라서 학습하고, 나중에 생성할 때에는 한번에 한 엘리먼트씩 생성됩니다. (생성된 아웃풋을 다음 스텝의 인풋으로 사용.)





```python
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
from text_loader import TextLoader
```


```python
num_layers  = 3
hidden_size = 512
batch_size  = 1
max_length  = 1000

```


```python
X = tf.placeholder(tf.int32, [None, None])
x_one_hot = tf.one_hot(X, vocab_size)

cells = [rnn.BasicLSTMCell(hidden_size) for _ in range(num_layers)]
cells = rnn.MultiRNNCell(cells, state_is_tuple=True)

initial_state = cells.zero_state(batch_size, tf.float32)
outputs, states = tf.nn.dynamic_rnn(cells, x_one_hot, 
                                    initial_state=initial_state, dtype=tf.float32)

outputs = tf.reshape(outputs, [-1, hidden_size])
logits = layers.fully_connected(outputs, vocab_size,
                                activation_fn=None)
y_softmax = tf.nn.softmax(logits)
pred = tf.argmax(y_softmax, axis=1)
pred = tf.reshape(pred, [batch_size, -1])
```

학습을 진행해봅시다.


```python
sentence = list()
sentence += input_sequence[:10].tolist()

print("Input seed data:", "".join([input_sequence_v_i[char] for char in sentence]))

saver = tf.train.Saver()
sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "checkpoints/char-rnn_30000")
    
    # 매 이터레이션에 글자 하나씩 생성 (일정 sequence length 만큼)
    pred_char, state = sess.run([pred, states], feed_dict={X:[sentence[:]]})
    for i in range(max_length):
        # 이전 스텝에 갖고 있는 state 값을 다음 스텝에 넣어줌
        pred_char, state = sess.run([pred, states], 
            feed_dict={X:[[sentence[-1]]], initial_state: state})
        sentence.append(pred_char[0][-1])
        
sentence = [loader.words[char] for char in sentence]
print("".join(sentence))
```


```python

```

유저의 인풋을 받아서 넣어줄 때에는 기존 커브 데이터에 없는 커브가 들어올 수 있기 때문에, 최대한 비슷한 커브로 대체하는 작업이 필요합니다.


```python
def truncate(f, n):
    s = '%.12f' % f
    i, p, d = s.partition('.')
    return float('.'.join([i, (d+'0'*n)[:n]]))
```


```python
def euclidean_dist(a, b):
    return np.linalg.norm(np.array(a)-np.array(b))

def find_similar_curve(query_curve, mel_set):
    list_mel_set = list(mel_set)
    min_dist = 10000 # to act like positive infinity
    found_curve_idx = -1 # just to initialize
    for idx, curve in enumerate(list_mel_set):
        if np.array_equal(query_curve, curve):
            found_curve_idx = idx
            break
        elif euclidean_dist(query_curve, curve) < min_dist:
            min_dist = euclidean_dist(query_curve, curve)
            found_curve_idx = idx
    print(list_mel_set[found_curve_idx])
    return found_curve_idx

```


```python
def get_mel_id_sequence(mel_arr_list, curve_arr_list):
        
    input_seq_list = []
    label_seq_list = []

    for curve_arr in curve_arr_list:
        print(len(curve_arr))
        mel_id_seq = []
        for curve in curve_arr:
            mel_id = mel_v_i[curve]
            mel_id_seq.append(mel_id)

        input_id_seq = np.array(mel_id_seq)[:-sequence_length]
        label_id_seq = np.roll(np.array(mel_id_seq), -sequence_length)[:-sequence_length]
        # label_id_seq = np.array(mel_id_seq)[:-sequence_length]
        print('input_id_seq', input_id_seq)
        print('label_id_seq', label_id_seq)

        input_seq_list.append(input_id_seq)
        label_seq_list.append(label_id_seq)

    return input_seq_list, label_seq_list
```


```python
input_seq_list, label_seq_list = get_mel_id_sequence(mel_arr_list, curve_arr_list)
```


```python
def get_batch_sequence(input_seq_list, label_seq_list, sequence_length):
        
    input_sequences = []
    label_sequences = []

    for idx in range(len(input_seq_list)):
        num_seqs_per_song = max(int((len(input_seq_list[idx]) / sequence_length)) - 1, 0)

        for ns in range(num_seqs_per_song):
            input_sequences.append(input_seq_list[idx][ns * sequence_length:(ns+1) * sequence_length])
            label_sequences.append(label_seq_list[idx][ns * sequence_length:(ns+1) * sequence_length])

    return np.array(input_sequences), np.array(label_sequences)
```


```python
input_sequences, label_sequences = get_batch_sequence(input_seq_list, label_seq_list, args.sequence_length)
```


```python

```
