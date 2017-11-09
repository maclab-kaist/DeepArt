<h2><a name="d1" style="color:black; text-decoration:none;">Deep Dream 소개</a></h2>


인공 신경망에서 학습된 뉴런의 대다수는 추상성의 계층에 따라서 어떠한 추상 또는 패턴을 탐지합니다.<br/>
이러한 추상 또는 패턴의 예를 보자면 다음과 같습니다.

<p align="center"> 
<img src="./filters.png">
</p>

<p align="center" style="color:#337ab7; font-size: 0.7em;">사진출처: https://distill.pub/2017/feature-visualization/</p>

위의 사진은 이미지 분류를 위하여 학습된 필터를 계층에 따라서 시각화 한 것으로 Edges부터 Objects까지 추상성의 변화과정을 보여줍니다. 예를 들어, 인공신경망의 하위 층(윗 두줄)에서는 외곽선 또는 질감 같은 낮은 수준의 패턴을 학습하고, 상위 층(아래 두 줄)에서는 꽃, 건물 그리고 눈과 같은 상위 수준의 물체를 학습합니다.<br/>

우리는 이러한 필터를 이용하여 다음과 같은 매우 혼란스러운 작품을 만들어 볼 수 있습니다. 

<p align="center"> 
<img src="https://cdn-images-1.medium.com/max/800/1*PKIwmv_VuRKnaECuKkIvtQ.gif">
</p>
<p align="center" style="color:#337ab7; font-size: 0.7em;">GIF출처: https://cdn-images-1.medium.com/max/800/1*PKIwmv_VuRKnaECuKkIvtQ.gif</p>

위와 같은 그림을 Deep Dream이라 부르며, 인공 신경망의 학습에 사용된 데이터가 강아지, 고양이와 같은 동물을 많이 포함하였기에, 어류, 개, 눈알의 모양으로 왜곡되거나, 우주에서 온 듯한 기이한 색채를 뿌리는 형태로 이미지가 변환되는 성향이 강합니다. 하여 코스믹 호러 이미지라거나 LSD에 의한 환각체험 같다는 평도 많이 있습니다. (from <a href="http://nuriwiki.net/wiki/index.php/Google_Deep_Dream" target="_blank">누리위키</a>)

이러한 꿈꾸는듯한 이미지는 다음의 방법으로 생성됩니다. <br/>

- Seed Image: 변환하고자 하는 이미지를 선택합니다.
- Layer Selection: 변환하고자 하는 추상성의 레벨을 선택합니다. (층이 높을수록 더욱 추상적, 하위일수록 점선면 패턴과 같이 더욱 직관적인 패턴이 Seed Image에 덧씌워집니다.)
- Activation Maximization: 해당 층에서의 평균 뉴론 반응수치를 최대화하고자 Seed Image를 계속해서 수정해줍니다.
- Hallucinated Image: Seed Image가 선택한 층의 뉴론 반응수치를 최대화하도록 변환됩니다. 

**본 튜토리얼**에서는 다음 섹션에서 이러한 Deep Dream을 기성 플랫폼을 이용하여 손쉽게 사용할 수 있는 방법에 대하여 소개하고, 더 나아가 마지막 섹션에서 직접 코딩을 통하여 해당 작업을 수행해 봅니다.

<h2><a name="d2" style="color:black; text-decoration:none;">Deep Dream 나만의 그림 만들기</a></h2>

Deep dream을 이용하여 나만의 추상적인 그림을 만들어 보고자 합니다.<br/><br/>

<a href="https://deepdreamgenerator.com/generator" style="text-decoration:none;transition: color ease 0.7s;" target="_blank">"DEEP DREAM GENERATOR"</a>라는 기성 사이트를 이용하여 우리는 가지고 있는 이미지에 원하는 추상성을 손쉽게 덧입힐 수 있습니다.

먼저, 로그인을 한 후 (페이스북, 구글+, 트위터 계정으로 손쉽게 로그인 할 수 있습니다), 
<p align="center"> 
<img src="./login.png">
</p>



https://deepdreamgenerator.com/generator (Deep Style, Thin Style, Deep Dream)
사용방법


예시
http://www.miketyka.com/?s=deepdream
http://barabeke.it/portfolio/deep-dream-generator


<h2><a name="d3" style="color:black; text-decoration:none;">Deep Dream 코딩 해보기</a></h2>


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

