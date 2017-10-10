# 음원 분류 및 2차원 시각화

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
