# Deep Dream

인공 신경망에서 학습된 뉴런의 대다수는 어떠한 추상 또는 패턴을 탐지합니다.</br>
이러한 추상 또는 패턴의 예를 보자면 다음과 같습니다.

![ex_screenshot](./filters.png)

사진출처: https://distill.pub/2017/feature-visualization/</br></br>





# Deep Dream 따라해보기 



https://deepdreamgenerator.com/generator (Deep Style, Thin Style, Deep Dream)
사용방법


예시
http://www.miketyka.com/?s=deepdream
http://barabeke.it/portfolio/deep-dream-generator


# Deep Dream 코딩 해보기 



이 튜토리얼에서는 인공신경망을 이용해 다양한 음원의 종류를 파악하는 분류기를 만들어 보고, 이를 이용해 각 음원의 특징을 2차원 상에 나타내는 시각화를 해 보겠습니다.

이 튜토리얼은 다음과 같은 구조로 이루어져 있습니다.

== toc ==

## 준비

이 튜토리얼을 완료하기 위해선 다음과 같은 파이썬 라이브러리가 필요합니다.

- numpy: 수치 자료를 고속으로 다루기 위한 라이브러리입니다.
- librosa: 음원 파일을 분석하는 데 필요한 라이브러리입니다.
- tensorflow: 인공신경망을 이용해 학습시키는 데 필요한 라이브러리입니다.
- scikit-learn: tSNE를 이용해 자료를 2차원에 시각화하는데 필요한 라이브러리입니다.

== 설치 과정? ==

## 자료 준비와 전처리

인공신경망을 학습시키기 위해선 좋은 품질의 자료가 필요합니다. 마침 구글의 마젠타 프로젝트에서 [NSynth](https://magenta.tensorflow.org/datasets/nsynth)라는 악기 데이터셋을 제공합니다.

### 데이터셋 다운받기

NSynth 데이터셋은 tfrecord, json/wav 두 가지 종류로 다운로드할 수 있습니다. 이번 튜토리얼에서는 wav파일을 직접 전처리하는 과정을 연습해볼 것이므로 json/wav 형태의 자료를 다운로드합니다. 또한, NSynth 데이터셋은 세 부분으로 나뉘어 있습니다. 이번에는 세 부분 모두 필요하므로 전부 다운로드해서 압축을 풀도록 합시다.

각 데이터셋의 의미는 다음과 같습니다.
- Train: 모델을 학습시킬 때 사용하는 부분입니다.
- Valid: 최적의 조건을 찾기 위해 여러 조건에서 모델을 학습하게 되는데, 이 때 이 부분을 이용해 각 모델의 성능을 비교합니다.
- Test: Valid를 이용해 찾은 최고의 모델을 최종적으로 테스트하는데 필요한 부분입니다.

NSynth 데이터셋은 다양한 가상악기에서 높이와 세기를 달리하며 한 음씩 녹음한 데이터셋입니다. 우리가 분류하고 싶은 것은 악기의 종류(Instrument Families)입니다. [NSynth 데이터셋의 설명](https://magenta.tensorflow.org/datasets/nsynth#instrument-families)에 따르면 11개의 종류가 있고, 각 음원은 단 하나의 종류에만 해당한다고 합니다.

세 파일의 압축을 풀면 다음과 같은 파일 구조가 나오게 됩니다.

```
- train
  | audio
  | examples.json
- valid
  | audio
  | examples.json
- test
  | audio
  | examples.json
```

압축 해제 방식 등의 이유로 폴더 구조가 다르다면 위와 같이 맞춰줍니다. `audio` 폴더 안에는 `wav` 파일들이 있고, 그 위에는 `examples.json` 파일이 있습니다. `wav`파일은 압축되지 않은 음원 파일이고, `json`은 해당 음원 파일에 관한 추가적인 정보를 답고 있는 파일입니다. `wav` 파일의 이름은 `bass_synthetic_035-025-030.wav`와 같은 식으로 `음원종류_생성방식_악기번호_높이_세기.wav`로 이루어져 있습니다. 우리는 음원 종류만 필요하므로 `json` 파일을 따로 분석할 필요는 없습니다.

### 데이터셋 정보 정리하기

데이터가 준비되었으면 먼저 각 파일의 경로와 악기 종류를 정리해둔 파일을 만들어 보겠습니다. `example.json`에도 우리가 필요한 정보가 정리되어 있지만 모든 데이터셋이 그런 것을 제공하지는 않으므로 연습해보도록 하겠습니다.

먼저 악기 종류를 정리해서 저장해 둡니다.

`labels.txt` 파일을 만들고 다음과 같이 내용을 작성한 후 저장합니다.

```bass
brass
flute
guitar
keyboard
mallet
organ
reed
string
synth_lead
vocal
```

<h2><a name="exactline"><span style="color:black; text-decoration:none">Linking Directly to a Specific Location</span></a></h2>


레이블은 일반적으로 어떤 학습 모델이 찾아야 할 답(이 경우에는 악기의 종류)을 말합니다.

`gather_information.py` 파일을 만듭니다. 이 파일에서는 다음과 같은 작업들을 하겠습니다.

- audio 폴더에서 `wav` 파일 목록을 얻어오기
- 각 파일 이름에서 앞 부분을 잘라서 어떤 종류의 악기인지 파악하기
- train, valid, test별로 음원 파일 이름과 악기 종류를 모아서 저장하기

그럼 실제로 파일을 작성해 보겠습니다. 다음 코드를 참고해서 파일을 작성해 주세요.

```python
import os # 파일 목록을 구할 때 필요한 패키지

def gather_information(part): # part 인자에는 'train', 'valid', 'test' 등을 넣어줍니다.
  label_list = open('labels.txt').read().strip().split('\n') # labels.txt 파일을 읽어서 각 줄을 기준으로 나눕니다.
  files = [] # wav 파일의 목록
  labels = [] # 각 파일의 label
  all_files = os.listdir(part + '/audio') # 각 part 아래 'audio' 안에 있는 파일의 목록을 불러옵니다.
  for f in all_files: # all_files에 있는 각 파일들에 대해서
    if f[-4:] == '.wav': # 파일이 '.wav'로 끝나면
      files.append(f[:-4]) # files 목록에 추가하고,
      label = f.split('_')[0] # 파일의 가장 첫 단어를 잘라냅니다.
      if label == 'synth': # synth_lead의 경우는 두 단어로 이루어져 있으므로
        label = 'synth_lead' # 첫 단어가 synth인 경우에는 label 이름을 synth_lead로 바꿔줍니다.
      labels.append(label_list.index(label)) # 이제 label_list에서 label의 위치를 찾아서 labels에 추가합니다.
  file_out = open(part + '_samples.txt', 'w') # part+'_samples.txt' 에 파일 목록을 적어줍니다.
  for f in files:
    file_out.write(f + '\n')
  file_out.close()
  label_out = open(part + '_labels.txt', 'w') # label도 저장합니다.
  for l in labels:
    label_out.write(str(l) + '\n')
  label_out.close()

if __name__ == '__main__' :
  gather_information('train') # 위 함수를 'train', 'valid', 'test'에 대해 실행합니다.
  gather_information('valid')
  gather_information('test')
```

파일의 각 부분을 좀 더 자세히 보겠습니다.

```python
def gather_information(part):
```

정보를 모으는 기능을 독립된 함수로 만들었습니다. 'train', 'valid', 'test'에 대해 동일한 작업을 반복해야 하므로 이렇게 동일한 작업을 함수로 만들면 코드가 간결하고 이해하기 쉬워집니다. part에는 'train', 'valid', 'test' 중 하나가 들어올 수 있습니다. 만약 폴더명이 다르다면 해당 폴더명을 넣어도 됩니다.

```python
labels_list = open('labels.txt').read().strip().split('\n')
```

우리는 악기의 종류를 'bass', 'brass' 처럼 문자로 기억하지만, 인공지능 모델을 학습시키는 등 연산 작업을 할 때는 숫자로 변환시켜서 사용합니다. 예를 들면 'bass'는 `0`, 'brass'는 `1`같은 식으로 사용합니다. `labels_list`는 우리가 사용할 악기의 종류를 담고 있는 리스트입니다. 'labels.txt'의 내용을 읽어서, 줄바꿈('\n') 기준으로 잘라내어 리스트를 만듭니다.(`strip()` 함수는 문자열 양 끝의 불필요한 공백문자를 잘라내 줍니다.) 그 결과는 다음과 같은 리스트가 됩니다.

```python
['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
```

이제 이 리스트에서 각 악기 종류의 위치가 해당 악기 종류를 나타내는 숫자가 됩니다.

```python
all_files = os.listdir(part + '/audio')
```
`os.listdir`은 주어진 경로 아래에 있는 파일 목록을 반환합니다. 우리는 'train/audio'와 같이 `part + '/audio'` 아래에 음원 파일들이 있으므로 all_files에는 음원 파일들의 목록이 저장됩니다.

```python
for f in all_files:
  if f[-4:] == '.wav':
```

`all_files`에 들어있는 각 파일에 대해서 정보를 처리하고 파일 목록과 레이블 목록에 추가합니다. 그런데, 'audio' 디렉토리 아래에 있는 파일 중 '.wav'파일이 아닌 파일도 있을 수 있으므로 (NSynth 데이터셋 자체에는 없지만 시스템에서 숨겨진 파일을 생성할 수 있습니다.) 파일 이름의 맨 뒤 네 글자가 '.wav'로 끝나는 경우에만 작업을 진행하도록 합니다.

```python
files.append(f[:-4])
```

files에 현재 파일을 추가합니다. 이 때, 다음 작업의 편의를 위해 뒤의 네 글자('.wav')를 제외하고 저장합니다.

```python
label = f.split('_')[0]
```

`split` 함수는 문자열을 인자를 기준으로 나눈 리스트를 반환합니다. 예를 들어 다음과 같은 경우,

```python
words = 'bass_electronic_044-046-075'.split('_')
```

`words`에는 `['bass', 'electronic', '044-046-075']`가 저장됩니다. 즉 `f.split('_')[0]`은 파일명을 '_'를 기준으로 잘랐을 때의 첫 번째 단어가 됩니다. 대부분의 경우 이 첫 단어가 악기의 종류를 나타냅니다.

```python
if label == 'synth':
  label = 'synth_lead'
```

하지만 레이블 중 'synth_lead'는 두 단어로 이루어져 첫 단어만 잘랐을 때는 'synth'가 됩니다. 이런 경우 'synth'를 'synth_lead'로 바꿔줍니다.

```python
labels.append(label_list.index(label))
```

`index()` 함수를 이용해 `label_list`에서 `label`이 몇 번째에 있는지를 알아낼 수 있습니다. 위에서 확인한 대로 'bass'의 경우 `0`, 'brass'의 경우 `1`이 반환되어 `labels`에 추가되게 됩니다.

```python
file_out = open(part + '_samples.txt', 'w')
for f in files:
  file_out.write(f + '\n')
file_out.close()
```

`samples`와 `labels`를 파일에 저장합니다. 이 때, 한 줄에 한 샘플이 들어가도록 샘플 파일 명에 줄바꿈('\n')을 붙여줍니다.

이제 `gather_information.py` 파일을 파이썬으로 실행하면 6개의 `txt` 파일이 생긴 것을 확인할 수 있습니다.

### 스펙트럼 추출하기

이제 'wav'파일을 읽고 스펙트럼을 추출해 보도록 하겠습니다.

`extract_spectrum.py` 파일을 만듭니다. 이 파일에서는 다음과 같은 일들을 할 것입니다.

- 오디오 파일 목록 불러오기
- 각 파일을 읽어서 스펙트럼으로 변환
- 정규화를 위한 통계 누적
- 변환된 파일을 저장하기

파일의 내용은 다음과 같습니다.

```python
import numpy # 수치 연산에 이용
import librosa # 음원 파일을 읽고 분석하는 데 이용
import os # 디렉토리 생성 등 시스템 관련 작업
import os.path # 특정 경로가 존재하는지 파악하기 위해 필요

sequence_length = 251
feature_dimension = 513

def extract_spectrum(part):
  sample_files = open(part + '_samples.txt').read().strip().split('\n') # 샘플 목록을 읽어옵니다.
  if part == 'train': # 'train'인 경우에는 평균과 표준편차를 구해야 합니다.
    data_sum = numpy.zeros((sequence_length, feature_dimension)) # 합계를 저장할 변수를 만듭니다.
    data_squared_sum = numpy.zeros((sequence_length, feature_dimension)) # 제곱의 합을 저장할 변수입니다.
  if not os.path.exists(part+'/spectrum/'): # 'spectrum' 디렉토리가 존재하지 않으면 만들어 줍니다.
    os.mkdir(part+'/spectrum/')
  for f in sample_files:
    print('%d/%d: %s'%(sample_files.index(f), len(sample_files), f)) # 현재 진행상황을 출력합니다.
    y, sr = librosa.load(part+'/audio/'+f+'.wav', sr=16000) # librosa를 이용해 샘플 파일을 읽습니다.
    D = librosa.stft(y, n_fft=1024, hop_length=256).T # short-time Fourier transform을 합니다.
    mag, phase = librosa.magphase(D) # phase 정보를 제외하고, 세기만 얻습니다.
    S = numpy.log(1 + mag * 1000) # 로그형태로 변환합니다.
    if part == 'train': # 'train'인 경우 합계와 제곱의 합을 누적합니다.
      data_sum += S
      data_squared_sum += S ** 2
    numpy.save(part+'/spectrum/'+f+'.npy', S) # 현재 샘플의 스펙트럼을 저장합니다.
  if part == 'train': # 모든 파일의 변환이 끝난 후에, 'train'인 경우 평균과 표준편차를 저장합니다.
    data_mean = data_sum / len(sample_files)
    data_std = (data_squared_sum / len(sample_files) - data_mean ** 2) ** 0.5
    numpy.save('data_mean.npy', data_mean)
    numpy.save('data_std.npy', data_std)

if __name__ == '__main__':
  for part in ['train', 'valid', 'test']:
    extract_spectrum(part)

```

스펙트럼은 어떤 신호를 소리의 높이에 따라 분해했을 때, 각 높이에 해당하는 성분이 얼마나 강한지 나타냅니다. 이 때, 신호의 높이를 주파수라고 하고, 낮은 소리는 낮은 주파수 성분이 강하게, 높은 소리는 높은 주파수의 성분이 강하게 나옵니다. 이를 통해 어떤 소리의 특징을 보다 알기 쉽게 표현할 수 있습니다. 스펙트럼을 짧은 시간마다 반복해서 추출한 것을 STFT(short-time Fourier transform)이라고 합니다.

많은 경우에 음원 자체(raw wave)를 이용하는 것보다 스펙트럼과 같이 분석된 정보를 이용하는 것이 기계학습에 더 효율이 좋다고 알려져 있습니다. 이 연습에서는 매 스펙트럼마다 1024 개의 샘플(기록)을 사용하고, 다음 스펙트럼은 256 샘플만큼 뒤에서 다시 추출합니다. 즉, 모든 스펙트럼은 자신과 인접한 스펙트럼과 768개의 샘플을 공유하게 됩니다.

샘플링 레이트(samping rate)는 소리를 녹음할 때 얼마나 자주 기록하는지를 의미합니다. NSynth 데이터셋의 경우 16,000 Hz로 일초에 16,000번 기록했다는 의미입니다. 전체 샘플 길이는 4초이니 각 샘플은 총 16,000 * 4 = 64,000개의 샘플을 가지고 있습니다.

전체 64,000개의 샘플을 256개씩 넘어가면서 스펙트럼을 추출하기 때문에 방식에 따라 250개 내외의 스펙트럼이 나오게 됩니다. 여기서 사용한 `librosa.stft`의 경우 총 251개가 나옵니다.

```python
sequence_length = 251
feature_dimension = 513
```

스펙트럼에 1024개의 샘플을 넣었기 때문에 결과로 나오는 한 스펙트럼은 513개의 값을 가집니다. 한 파일에서 STFT를 추출하면 (251, 513)의 크기를 가진 행렬이 나오게 됩니다. 위에서는 미리 두 숫자를 정의했습니다.

```python
if part == 'train': # 'train'인 경우에는 평균과 표준편차를 구해야 합니다.
    data_sum = numpy.zeros((sequence_length, feature_dimension))
    data_squared_sum = numpy.zeros((sequence_length, feature_dimension))
```

'train' 셋의 경우 정규화(normalization)를 위해 평균(mean)과 표준편차(standard deviation, std)를 구해야 합니다. 정규화는 자료의 전체 혹은 일부의 평균과 표준편차를 일정한 값으로 조정해 주는 것으로 일반적으로 평균은 0으로, 표준편차는 1로 변환해줍니다. 자료가 정규화되면 학습에 사용하는 다양한 함수들이 더 효과적인 범위에서 작동하게 됩니다.

모든 자료를 동시에 로드할 수 있다면 `numpy.mean(), numpy.std()` 함수를 통해 간단히 평균과 표준편차를 구할 수 있지만
이 경우는 자료가 매우 크므로 직접 통계적인 계산을 해야 합니다. 평균은 `전체 데이터의 합 / 전체 데이터의 수`로 구할 수 있고, 표준편차는 `(전체 데이터의 제곱의 평균 - 전체 데이터의 평균의 제곱)의 제곱근`으로 구할 수 있습니다. 전체 데이터의 수는 이미 알고 있으므로 합과 제곱의 합을 저장할 변수를 만들어 줍니다.

```python
print('%d/%d: %s'%(sample_files.index(f), len(sample_files), f))
```

프로그램이 실행되는 중에 진행 상황에 관해 적절한 정보를 주도록 만들면 어떤 문제가 생겼을 때 쉽게 대처할 수도 있고, 실행 시간이 매우 긴 프로그램의 경우 남은 시간을 추측할 수도 있습니다. 위의 코드는 예를 들어 현재 어떤 파일을 처리하고 있고, 전체의 몇번째 파일인지를 출력해줍니다. 만약 특정 파일을 처리하다 에러가 났다면 마지막으로 처리하던 파일의 이름을 알 수 있습니다.

```python
y, sr = librosa.load(part+'/audio/'+f+'.wav', sr=16000)
```

`librosa.load()` 함수는 음원 파일을 읽어 샘플을 리스트로 반환해 줍니다. 샘플링 레이트는 위에서 언급한 대로 16,000을 사용합니다.

```python
D = librosa.stft(y, n_fft=1024, hop_length=256).T
```

`librosa.stft()` 함수를 이용해 스펙트럼을 얻습니다. `librosa`는 `(feature_dim, sequence_length)` 형태로 반환하므로 `.T`를 이용해 `(sequence_length, feature_dim)`의 형태로 뒤집어 줍니다.

```python
mag, phase = librosa.magphase(D)
S = numpy.log(1 + mag * 1000)
```

스펙트럼은 복소수로 되어 있습니다. 이 중에 주로 사용되는 부분은 실수 부분으로 어떤 주파수 영역의 세기(magnitude)를 의미합니다. 복소수 부분은 페이즈(phase)라고 합니다. 예제 코드에서는 `librosa.magphase()` 함수를 이용해 두 부분을 분리했습니다.

사람이 소리의 크기를 인지할 때, 소리의 크기가 커질수록 크기에 대한 민감도가 낮아집니다. 이를 좀 더 자세히 설명하면, 소리가 작을 때에는 에너지가 조금만 증가해도 소리가 커졌다고 인식하지만 소리가 클 때에는 에너지가 훨씬 더 많이 증가해야 소리가 비슷한만큼 커졌다고 인식하게 됩니다. 이를 수학적으로 로그함수(logarithm)으로 나타낼 수 있습니다. 이에 따라 `log`함수를 이용해 스펙트럼의 세기를 변환해주는 것이 좋습니다.

```python
if part == 'train':
  data_sum += S
  data_squared_sum += S ** 2
numpy.save(part+'/spectrum/'+f+'.npy', S)
```

`'train'`인 경우 `data_sum`에는 `S`값을, `date_squared_sum`에는 `S`값의 제곱을 더해줍니다. 모든 계산이 끝나면 현재 샘플의 스펙트럼 값을 '.npy' 파일로 저장해 줍니다.

```python
if part == 'train': # 모든 파일의 변환이 끝난 후에, 'train'인 경우 평균과 표준편차를 저장합니다.
  data_mean = data_sum / len(sample_files)
  data_std = (data_squared_sum / len(sample_files) - data_mean ** 2) ** 0.5
  numpy.save('data_mean.npy', data_mean)
  numpy.save('data_std.npy', data_std)
```

모든 샘플을 처리한 후에 평균과 표준편차를

파일 작성 후 파이썬을 이용해 실행하면 각 부분 폴더 아래에 'spectrum' 폴더가 생성되고, 각 음원파일의 스펙트럼이 '.npy' 파일로 저장됩니다.

## 모델 생성하고 학습시키기

그럼 이제 모델을 만들고, 수집된 데이터를 이용해서 학습을 시켜보겠습니다.

먼저 모델을 정의하는 파일을 만들어 보겠습니다.

`model.py` 파일을 만들고, 다음과 같이 적어주세요.

```python
import tensorflow as tf

def get_model(X, is_training, sequence_length, spectrum_size, n_labels):
  last_layer, last_minus_one_layer = get_model_and_activation(X, is_training, sequence_length, spectrum_size, n_labels)
  return last_layer

def get_last_minus_one(X, is_training, sequence_length, spectrum_size, n_labels):
  last_layer, last_minus_one_layer = get_model_and_activation(X, is_training, sequence_length, spectrum_size, n_labels)
  return last_minus_one_layer

def get_model_and_activation(X, is_training, sequence_length, spectrum_size, n_labels):
  conv1 = tf.contrib.layers.conv2d(X, 64, (3, spectrum_size), padding='VALID', normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params={"is_training":is_training})
  pool1 = tf.contrib.layers.max_pool2d(conv1, (3, 1), stride=3)
  conv2 = tf.contrib.layers.conv2d(pool1, 64, (3, 1), normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params={"is_training":is_training})
  pool2 = tf.contrib.layers.max_pool2d(conv2, (3, 1), stride=3)
  conv3 = tf.contrib.layers.conv2d(pool2, 128, (3, 1), normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params={"is_training":is_training})
  pool3 = tf.contrib.layers.max_pool2d(conv3, (3, 1), stride=3)
  conv4 = tf.contrib.layers.conv2d(pool3, 128, (3, 1), normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params={"is_training":is_training})
  pool4 = tf.contrib.layers.max_pool2d(conv4, (3, 1), stride=3)
  flatten = tf.contrib.layers.flatten(pool4)
  fc1 = tf.contrib.layers.fully_connected(flatten, 256)
  fc2 = tf.contrib.layers.fully_connected(fc1, 256)
  fc3 = tf.contrib.layers.fully_connected(fc2, n_labels)
  return fc3, fc2
```

우리가 사용할 모델의 각 레이어와 레이어별 출력 사이즈는 다음과 같습니다.

```
input: (sequence_length * spectrum_size), 1
convolution1: (3 * spectrum_size), 64
maxpool1: (3 * 1)
convolution2: (3 * 1), 64
maxpool2: (3 * 1)
convolution3: (3 * 1), 128
maxpool3: (3 * 1)
convolution4: (3 * 1), 128
maxpool4: (3 * 1)
flatten: (n)
fc1: 256
fc2: 256
fc3: n_labels
```

입력의 경우 스펙트럼의 길이 * 스펙트럼의 크기가 들어옵니다. 위에서 정의한 대로라면 `251 * 513`의 데이터가 들어오게 됩니다. 뒤의 1은 채널 개수인데, 이미지의 경우 RGB 채널당 하나씩 총 3개의 데이터가 들어오지만 오디오의 경우는 하나입니다. 실제로는 한 번에 여러 샘플 데이터를 동시에 사용하고, 이를 배치(batch)라고 합니다. 실제 입력 데이터는 여기서 정의된 것 * 배치 크기가 됩니다.

convolution1 레이어의 경우 필터 크기가 (3 * spectrum_size)이고, 64개의 필터를 사용합니다. 추가적인 패딩을 넣지 않았기 때문에(`padding='VALID'`) 시간 축은 (3 - 1)만큼, 주파수 축은 (spectrum_size - 1)만큼 작아지게 됩니다. 결국 `(sequence_length - 2, 1)`의 크기가 됩니다. 이는 의도적으로 데이터를 1차원으로 만들기 위한 것입니다. 여기서 필터 크기를 더 작게 하면 2차원 컨볼루션을 이용하게 됩니다.

추가적으로 모든 컨볼루션 레이어는 배치 정규화(batch normalization)을 적용합니다. 배치 정규화는 학습중인지 테스트중인지에 대한 정보가 필요한데, 이를 `is_training` 패러미터를 이용해 전달하게 됩니다.

maxpool1 레이어는 (3, 1)로 맥스 풀링을 합니다. 맥스 풀링은 주어진 데이터에서 일정한 개수의 값마다 가장 높은 값 한개를 선택해 출력하는 기법입니다. 이 경우 시간축에서는 3개, 주파수 축에서는 1개마다 풀링을 하므로 데이터가 대략 1/3로 줄게 됩니다. 주파수 축의 경우 이미 길이가 1이 되었기 때문에 추가로 맥스 풀링을 할 필요가 없으므로 1이 됩니다. 이 예제에서는 모든 컨볼루션 레이어 다음에 동일한 형태의 맥스풀링 레이어를 적용합니다.

convolution2 레이어는 (3 * 1)의 크기를 가진 필터를 적용합니다. 패딩을 따로 지정하지 않기 때문에 기본적으로 'SAME'이 적용되는데, 이는 입력과 출력 크기가 같도록 (필터 크기 - 1) / 2만큼씩 앞뒤로 패딩을 붙여줍니다.

convolution3, 4 레이어는 이전과 같지만 필터 개수를 2배로 늘렸습니다.

flatten 레이어는 입력되는 데이터의 모양을 1차원으로 만들어 줍니다.

fc1, 2, 3는 fully-connected 레이어이고, 각각 256, 256, n_labels의 출력을 가집니다.


다음은 본격적으로 학습에 들어가기 전에 자료를 읽어서 제공해주는 역할을 하는 파일을 작성해 보겠습니다.

`data_provider.py` 파일을 만들고, 다음과 같은 항목들을 작성해 주세요.

```python
import numpy
import random

n_labels = 11
batch_size = 32
sequence_length = 251
feature_dimension = 513

def prepare_data():
  global train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels, data_mean, data_std

  train_samples = open('train_samples.txt').read().strip().split('\n')
  train_labels = [int(label) for label in open('train_labels.txt').read().strip().split('\n')]

  valid_samples = open('valid_samples.txt').read().strip().split('\n')
  valid_labels = [int(label) for label in open('valid_labels.txt').read().strip().split('\n')]

  test_samples = open('test_samples.txt').read().strip().split('\n')
  test_labels = [int(label) for label in open('test_labels.txt').read().strip().split('\n')]

  data_mean = numpy.load('data_mean.npy')
  data_std = numpy.load('data_std.npy')

def get_random_sample(part):
  global train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels, data_mean, data_std

  if part == 'train':
    samples = train_samples
    labels = train_labels
  elif part == 'valid':
    samples = valid_samples
    labels = valid_labels
  elif part == 'test':
    samples = test_samples
    labels = test_labels
  else :
    print('Please use train, valid, or test for the part name')

  i = random.randrange(len(samples))
  spectrum = numpy.load(part+'/spectrum/'+samples[i]+'.npy')
  spectrum = (spectrum - data_mean) / (data_std + 0.0001)
  return spectrum, labels[i]

def get_sample_at(part, i):
  global train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels, data_mean, data_std

  if part == 'train':
    samples = train_samples
    labels = train_labels
  elif part == 'valid':
    samples = valid_samples
    labels = valid_labels
  elif part == 'test':
    samples = test_samples
    labels = test_labels
  else :
    print('Please use train, valid, or test for the part name')

  spectrum = numpy.load(part+'/spectrum/'+samples[i]+'.npy')
  spectrum = (spectrum - data_mean) / (data_std + 0.0001)
  return spectrum, labels[i]

def get_random_batch(part):
  X = numpy.zeros((batch_size, sequence_length, feature_dimension, 1))
  Y = numpy.zeros((batch_size,))
  for b in range(batch_size):
    s, l = get_random_sample(part)
    X[b, :, :, 0] = s[:sequence_length, :feature_dimension]
    Y[b] = l
  return X, Y
```

이 파일은 데이터의 목록를 읽어서 필요할 때마다 무작위로 샘플과 레이블을 읽어서 배치를 만들어 제공해주는 역할을 합니다.

```python
def prepare_data():
  global train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels, data_mean, data_std
```

`prepare_data` 함수는 샘플 파일 목록을 불러오는 역할을 합니다. 이 때, 각 샘플 파일 목록은 다른 함수에서도 사용해야 하기 때문에 위와 같이 전역 변수(global)로 지정해 줍니다.

```python
train_samples = open('train_samples.txt').read().strip().split('\n')
train_labels  = [int(label) for label in open('train_labels.txt').read().strip().split('\n')]
```

위에서 만들어둔 세가지의 샘플 파일과 레이블 파일을 읽어서 각각 리스트로 저장합니다.

```python
data_mean = numpy.load('data_mean.npy')
data_std = numpy.load('data_std.npy')
```

평균과 표준편차도 읽어옵니다.

```python
def get_random_sample(part):
  global train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels, data_mean, data_std

  if part == 'train':
    samples = train_samples
    labels = train_labels
  elif part == 'valid':
    samples = valid_samples
    labels = valid_labels
  elif part == 'test':
    samples = test_samples
    labels = test_labels
  else :
    print('Please use train, valid, or test for the part name')
```

`get_random_sample` 함수는 랜덤하게 하나의 샘플과 레이블 값을 반환합니다. 먼저 어떤 부분을 사용할 것인지에 따라서 `samples`, `labels` 리스트를 결정해 줍니다.

```python
i = random.randrange(len(samples))
spectrum = numpy.load(part+'/spectrum/'+samples[i]+'.npy')
spectrum = (spectrum - data_mean) / (data_std + 0.0001)
return spectrum, labels[i]
```

이후에 0부터 샘플의 개수 중에서 무작위로 하나의 숫자를 고른 후에, 이 번호에 해당하는 샘플 파일을 읽어옵니다. 평균을 빼고 표준편차로 나워서 0-1 평준화를 거쳐서 레이블과 함께 반환해 줍니다.

```python
def get_random_batch(part):
  X = numpy.zeros((batch_size, sequence_length, feature_dimension, 1))
  Y = numpy.zeros((batch_size,))
  for b in batch_size:
    s, l = get_random_sample(part)
    X[b, :, :, 0] = s[:sequence_length, :feature_dimension]
    Y[b] = l
  return X, Y
```

마지막으로 `get_random_batch` 함수는 배치의 크기만큼 `get_random_sample` 함수를 호출해 이를 하나의 배열로 묶어서 반환해줍니다.

이제 본격적으로 학습을 하는 부분을 작성해 보겠습니다.

`train.py` 파일을 만들고, 상단에 다음과 같은 항목들을 작성해 주세요.

```python
import tensorflow as tf
import model
import data_provider

n_labels = data_provider.n_labels
batch_size = data_provider.batch_size
sequence_length = data_provider.sequence_length
feature_dimension = data_provider.feature_dimension

def train():
  data_provider.prepare_data()

  with tf.Graph().as_default():
    X = tf.placeholder(tf.float32, shape=(batch_size, sequence_length, feature_dimension, 1))
    Y = tf.placeholder(tf.int32, shape=(batch_size,))
    phase_train = tf.placeholder(tf.bool)

    output = model.get_model(X, phase_train, sequence_length, feature_dimension, n_labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=Y)
    loss = tf.reduce_mean(cross_entropy)
    evaluation = tf.reduce_sum(tf.cast(tf.nn.in_top_k(output, Y, 1), tf.float32))

    optimizer = tf.train.AdadeltaOptimizer(0.01)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    accumulated_loss = 0

    for step in range(100000):
      x, y = data_provider.get_random_batch('train')
      _, loss_value = sess.run([train_op, loss], feed_dict={phase_train:True, X:x, Y:y})
      accumulated_loss += loss_value
      if (step + 1) % 10 == 0:
        print('step %d, loss = %.5f'%(step+1, accumulated_loss / 10))
        accumulated_loss = 0
      if (step + 1) % 100 == 0:
        correct = 0;
        for i in range(10):
          x, y = data_provider.get_random_batch('valid')
          corr = sess.run([evaluation], feed_dict={phase_train:False, X:x, Y:y})
          correct += corr[0]
        print('step %d, valid accuracy = %.2f'%(step+1, 100 * correct / 10 / batch_size))

    saver = tf.Saver(tf.all_variables())
    saver.save(sess, 'model.ckpt')

if __name__ == '__main__':
  train()
```

그러면 한 부분씩 보도록 하겠습니다.

```python
import tensorflow as tf
import model
import data_provider

n_labels = data_provider.n_labels
batch_size = data_provider.batch_size
sequence_length = data_provider.sequence_length
feature_dimension = data_provider.feature_dimension
```

`tensorflow`와 이전에 만들었던 `model, data_provider`를 불러옵니다. 레이블의 개수, 배치 크기, 데이터의 크기 등은 data_provider에서 가져오도록 합니다.

```python
def train():
  data_provider.prepare_data()
```

`data_provider`의 `prepare_data` 함수를 호출해 자료 목록을 읽어옵니다.

```python
  with tf.Graph().as_default():
    X = tf.placeholder(tf.float32, shape=(batch_size, sequence_length, feature_dimension, 1))
    Y = tf.placeholder(tf.int32, shape=(batch_size,))
    phase_train = tf.placeholder(tf.bool)
```

TensorFlow에서 모델을 만들고 연산을 하기 위해서는 `Graph`객체를 만들어야 합니다. 먼저 그래프를 만든 후 학습에 필요한 `X`(입력값), `Y`(레이블), `phase_train`(학습중 여부)의 `placeholder`를 만듭니다. placeholder는 실제로 값이 들어있는 변수는 아니고, Tensorflow의 연산 모델을 구성할 때 어떤 자리에 어떤 크기와 형태를 가진 데이터가 들어갈 것인지 표시해 두기 위한 변수입니다. 입력값이나 레이블 값은 후에 실제로 훈련할 때 계속 값을 바꿔가면서 훈련해야 하므로 placeholder를 만들어 둡니다.

```python
    output = model.get_model(X, phase_train)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=Y)
    loss = tf.reduce_mean(cross_entropy)
    evaluation = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, Y, 1), tf.float32))
```

만들어 둔 `model.py`의 `get_model`을 이용해 `output` 노드를 얻습니다. 이 노드는 `get_model` 함수에서 마지막 레이어인 `fc3`의 출력으로 주어진 입력이 각 레이블에 속하는 악기의 소리일 확률을 나타냅니다. Tensorflow에서 이렇게 실제 연산 전에 어떤 연산의 입력 혹은 결과 등 어떤 값이 들어갈 위치를 나타내는 것들을 노드(node)라고 부릅니다. placeholder도 노드의 일종입니다. 이렇게 다양한 노드를 지정해 두고 `Session.run` 함수를 통해 원하는 노드에 어떤 값이 들어가게 되는지 연산하게 됩니다.

위 코드에서는 `output`뿐 아니라 모델의 출력 확률과 실제 레이블의 차이를 나타내는 크로스엔트로피(cross_entropy)가 있습니다. 실제로는 각 배치에서 이 값의 평균을 취한 `loss` 노드의 값을 사용합니다.

또한 출력 확률이 제일 높은 악기가 실제 레이블과 일치하는지를 판단하는 `evaluation` 노드도 정의했습니다.

```python
    optimizer = tf.train.AdadeltaOptimizer(0.01)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
```

마지막 준비과정의 첫 번째는 위에서 정의한 `loss`, 즉 연산 결과와 실제 레이블의 차이를 최소로 줄여주도록 모델을 학습시키는 `optimizer`를 만드는 것입니다. 여기서는 Adadelta 기법을 사용합니다.

그 이후에는 실제로 각 노드를 연산하는데 필요한 세션(Session)을 생성하고, 연산에 필요한 변수들의 값을 초기화해줍니다.


```python
    accumulated_loss = 0
    for step in range(100000):
      x, y = data_provider.get_random_batch('train')
      _, loss_value = sess.run([train_op, loss], feed_dict={phase_train:True, X:x, Y:y})
      accumulated_loss += loss_value
```

연산과 학습은 위의 코드에서 일어납니다. `for` 구문을 이용해 각 단계를 십만번 반복하도록 지정했습니다.
각 단계에서 `get_randome_batch` 함수를 이용해 데이터를 불러오고, `train_op`와 `loss`값을 구하도록 연산합니다. `train_op` 연산의 경우 실제로 어떤 값을 출력하는 것은 아니고, `loss`가 작아지도록 모델을 조금씩 변화시키게 됩니다. 연산한 `loss` 값은 `accumulated_loss`에 저장합니다.

```
      if (step + 1) % 10 == 0:
        print('step %d, loss = %.5f'%(step+1, accumulated_loss / 10))
        accumulated_loss = 0
      if (step + 1) % 100 == 0:
        correct = 0;
        for i in range(10):
          x, y = data_provider.get_random_batch('valid')
          corr = sess.run([evaluation], feed_dict={phase_train:False, X:x, Y:y})
          correct += corr[0]
        print('step %d, valid accuracy = %.2f'%(step+1, 100 * correct / 10 / batch_size))
```

매 10번째 단계에서는 누적된 `loss`값을 출력해 학습의 진행 정도를 확인할 수 있습니다.

매 100번재 단계에서는 'valid' 데이터를 이용해 검증하도록 합니다. 이 때는 `loss` 대신에 정확도인 `evaluation` 노드를 사용합니다.

```python
    saver = tf.train.Saver(tf.all_variables())
    saver.save(sess, 'model.ckpt')
```


학습이 끝나면 학습된 모델의 변수들을 `'model.ckpt'`에 저장해 둡니다.

## t-SNE

학습이 끝나면 t-SNE 기법을 이용해 샘플을 분석한 결과가 어떻게 분포되는지 보고자 합니다. t-SNE는 다차원 데이터의 분포를 더 적은 차원에서 효과적으로 보여주는 기법입니다. 예를 들어, 우리가 학습시킨 모델은 주어진 입력값을 여러 레이어를 거쳐 연산한 후에 최종적으로 256차원의 데이터로 변환합니다(fc2). 여기서 fc3 레이어를 거쳐 11개의 악기에 대한 확률 값을 얻어냅니다. 이 256차원의 데이터는 각 입력 데이터에서 악기를 결정하는 데 중요한 데이터를 추출한 것이라고 볼 수 있습니다. 그렇지만 사람이 인지하기에는 너무 높은 차원이기 때문에 t-SNE를 이용해 2차원이나 3차원으로 그 분포를 나타내면 보다 쉽게 데이터의 분포를 확인할 수 있습니다.

먼저 학습된 모델에서 테스트 데이터를 이용해 fc2 레이어의 값을 추출해 보겠습니다.

activation.py 파일을 만들고 다음 내용을 적어주세요.

```python
import tensorflow as tf
import data_provider
import model

n_labels = data_provider.n_labels
data_provider.batch_size = 1
batch_size = 1
sequence_length = data_provider.sequence_length
feature_dimension = data_provider.feature_dimension

def get_activation():
  data_provider.prepare_data()
  n_samples = len(data_provider.test_samples)

  with tf.Graph().as_default():
    X = tf.placeholder(tf.float32, shape=(batch_size, sequence_length, feature_dimension, 1))
    phase_train = tf.placeholder(tf.bool)

    activations = numpy.zeros((n_samples, 256))

    output = model.get_last_minus_one(X, phase_train, sequence_length, feature_dimension, n_labels)

    sess = tf.Session()
    saver = tf.Saver(tf.all_variables())
    saver.restore(sess, 'model.ckpt')

    for step in range(n_samples):
      x, y = data_provider.get_sample_at('test', step)
      activation = sess.run(output, feed_dict={phase_train:False, X:x})
      activations[step, :] = activation

    numpy.save('activation.npy', activations)
```

tsne.py 파일을 다음과 같이 만들어 주세요.

```python
import numpy

act = numpy.load('activation.npy')

from sklearn.decomposition import PCA
pca = PCA(n_components=50)
act_pca = pca.fit_transform(act)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
act_tsne = tsne.fit_transform(act_pca)

tsne = TSNE(n_components=3)
act_tsne_3d = tsne.fit_transform(act_pca)


numpy.save('act_pca.npy', act_pca)
numpy.save('act_tsne.npy', act_tsne)
numpy.save('act_tsne_3d.npy', act_tsne_3d)
```
