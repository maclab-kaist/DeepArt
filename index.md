딥러닝을 이용한 예술 튜토리얼
=============================

이 페이지는 [KAIST MAC랩](http://mac.kaist.ac.kr) 에서 만들었습니다.

<h2>0. 시작하기</h2>

1. 프로젝트에 관하여

2.1 [Python 설치 / 시작하기 (musicinformationretrieval.com)](http://musicinformationretrieval.com/python_basics.html)

2.2 [Python 튜토리얼 (www.learnpython.org)](https://www.learnpython.org/en/Welcome)

3. [Numpy 튜토리얼 (musicinformationretrieval.com)](http://musicinformationretrieval.com/numpy_basics.html)

4. [jupyter notebook 튜토리얼 (musicinformationretrieval.com)](http://musicinformationretrieval.com/get_good_at_ipython.html)

5. [jupyter notebook audio 튜토리얼 (musicinformationretrieval.com)](http://musicinformationretrieval.com/ipython_audio.html)

<h2>1. Deep-Dream <span style="font-size:0.6em">(by 이종필, <a href="https://jongpillee.github.io/" style="color:#337ab7;text-decoration:none;transition: color ease 0.7s;" target="_blank">개인블로그</a>)</span></h2>

1. <a href="deep_dream/deep-dream.html#d1">Deep Dream 소개</a>

2. <a href="deep_dream/deep-dream.html#d2">Deep Dream 나만의 그림 만들기</a>

3. <a href="deep_dream/deep-dream.html#d3">Deep Dream 코딩 해보기</a>


<h2>2. Style-Transfer <span style="font-size:0.6em">(by 권태균)</span></h2>

1. <a href="deep_dream/deep-dream.html#d1">Style transfer 소개</a>

2. <a href="deep_dream/deep-dream.html#d2">이미지 읽어들이기</a>

3. <a href="deep_dream/deep-dream.html#d3">모델 읽어오기</a>

4. <a href="deep_dream/deep-dream.html#d4">Style loss, Content loss 정의하기</a>

5. <a href="deep_dream/deep-dream.html#d4">Optimization 하기</a>

<h2>3. Sound Classification and t-SNE <span style="font-size:0.6em">(by 김근형)</span></h2>

1. [데이터 준비하고 전처리하기](classify_tsne/classify_tsne_1_prepare_data.md)

2. [스펙트로그램 추출하기](classify_tsne/classify_tsne_2_extract_spectrum.md)

3. [모델 만들고 학습시키기](classify_tsne/classify_tsne_3_train.md)

4. [t-SNE로 분석하기](classify_tsne/classify_tsne_4_tsne.md)

<h2>4. Neural Network를 이용한 자동작곡 <span style="font-size:0.6em">(by 최정)</span></h2>

1. [소개 및 데이터 받아오기](melody_rnn/MelodyRNN_01_introduction.md)[(ipynb파일)](melody_rnn/MelodyRNN_01_introduction.ipynb)

2. [텐서플로우 간단 Review](melody_rnn/MelodyRNN_02_Tensorflow_Quick_Review.md)[(ipynb파일)](melody_rnn/MelodyRNN_02_Tensorflow_Quick_Review.ipynb)

3. [텐서플로우 Basic RNN 모델](melody_rnn/MelodyRNN_03_Tensorflow_Basic_RNN_model.md)[(ipynb파일)](melody_rnn/MelodyRNN_03_Tensorflow_Basic_RNN_model.ipynb)

4. [데이터 구조 결정 및 전처리](melody_rnn/MelodyRNN_04_Data_Preprocessing.md)[(ipynb파일)](melody_rnn/MelodyRNN_04_Data_Preprocessing.ipynb)

5. [모델 정의 및 학습](melody_rnn/MelodyRNN_05_Model_Define_Train.md)[(ipynb파일)](melody_rnn/MelodyRNN_05_Model_Define_Train.ipynb)

6. [학습된 모델을 사용한 멜로디 예측](melody_rnn/MelodyRNN_06_Model_Prediction.md)[(ipynb파일)](melody_rnn/MelodyRNN_06_Model_Prediction.ipynb)

**워크샵 참여인원은 다음 파일을 다운로드 받으세요.**  
**(나중에 DL4A_workshop/checkpoint 폴더에 모델 파일을 넣으세요)**

- [Tutorial workshop용 코드 다운로드](melody_rnn/DL4A_workshop.zip)
- [Tutorial workshop용 trained model 다운로드](melody_rnn/model.zip)




