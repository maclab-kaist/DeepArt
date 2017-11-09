
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

## 데이터 Preprocessing - Extracting note information from midi file.

우리는 미디 파일에서 음악 정보를 받아온 후 이를 바탕으로 패턴을 학습하는 모델을 만들 예정입니다. 

가장 먼저 필요한 작업은 미디 데이터셋을 준비하는 일입니다.

오픈되어 있는 미디 데이터셋에는 여러가지가 있지만, 단순한 포크송 형태의 기본적인 미디 데이터셋으로 Nottingham Database를 사용하고자 합니다.

1200개의 영국/미국 포크곡으로 이루어져 있는 데이터셋으로, 모든 미디파일은 멜로디, 화성의 두개의 트랙으로 구성되어 있습니다.

또한, 얼마전 음악 AI 스타트업 JukeDeck에서 이 데이터셋을 한번 더 정리(clean up)해서 Github에 공유했기 때문에 저희는 이를 활용하겠습니다.

[https://github.com/jukedeck/nottingham-dataset](https://github.com/jukedeck/nottingham-dataset)

이를 다운 받아서 로컬 하드드라이브의 적당한 위치에 저장합니다.

그런 다음, 데이터가 저장된 폴더를 변수로 지정해 놓습니다.



```python
data_path = '~/Google Drive/data/nottingham-dataset/MIDI/melody/'
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




    <music21.stream.Score 0x108ab4390>



music21은 라이브러리 내에 정의해놓은 특정한 오브젝트 형태로 정보를 저장합니다. 

이 오브젝트안에는 특정 구조로 음악 정보들이 저장되어 있습니다.

이 구조를 이용해서 우리가 원하는 노트의 음정 / 노트의 시작지점 두가지 정보에 대한 Array를 받아오겠습니다.


```python
def create_mel_data_each_file(midi_obj):
    mel_data = dict()
    
#     print(len(midi_obj.flat.getElementsByClass(music21.chord.Chord)))
    for n in midi_obj.flat.getElementsByClass(music21.chord.Chord):
        if n.offset not in mel_data:
            mel_data[n.offset] = dict()
        mel_data[n.offset]['offset'] = n.offset
        for p in n.pitches:
            mel_data[n.offset]['note'] = p.midi

#     print(len(midi_obj.flat.getElementsByClass(music21.note.Note)))
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

위 함수에서는 만약 두개 이상의 음이 동시에 발현되는 경우 단순하게 맨 위의 음정만을 멜로디로 치부해서 데이터화하도록 되어있습니다. 


```python
mel_data = create_mel_data_each_file(midi_obj)
```

한곡에서 165개의 멜로디 노트 정보를 받아오는 것을 확인했습니다.

자, 이제 우리 데이터셋의 모든 곡에 대해서 노트 정보를 받아와봅시다.

먼저, 폴더안의 모든 미디 파일명을 리스트로 만듭니다.



```python
file_list = []

for file_name in os.listdir(os.path.expanduser(data_path)):
    if file_name.endswith('.mid') or file_name.endswith('.midi'):
        file_list.append(data_path + file_name)
```

파일 리스트를 확인해봅시다.


```python
# file_list 
```

이제 모든 파일들의 path를 받아놓았으니, 루프를 돌면서 모든 파일에 대해서 멜로디 노트 정보를 저장해봅시다.


```python
mel_arr_list = []

for file_name in file_list:
#     print(file_name)
    midi_obj = music21.converter.parse(file_name)
    mel_data = create_mel_data_each_file(midi_obj)

    mel_arr = []
    for key, value in sorted(mel_data.items()):
        mel_arr.append(mel_data[key])
    
    mel_arr_list.append(mel_arr)
```

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
# mel_arr_list[0]
```
