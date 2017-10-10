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