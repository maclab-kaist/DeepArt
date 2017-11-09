
## 학습된 모델을 활용한 멜로디 prediction

자, 이제 드디어 이전 장에서 학습한 멜로디 모델을 사용해서 주어진 인풋에 대해 예측값을 출력해볼 차례입니다.

이 모델을 사용해서 우리는 구글 마젠타 프로젝트에서 선보였던 AI_DUET같은 어플리케이션을 시도해볼 수 있습니다.

[https://experiments.withgoogle.com/ai/ai-duet](https://experiments.withgoogle.com/ai/ai-duet)

즉, 유저에게 인풋을 받은 후 이에 이어지는 멜로디를(학습된 데이터에 기반해서) AI가 생성해서 피드백하는 형태의 어플리케이션입니다.

언제나처럼, 필요한 라이브러리를 로드해봅시다.



```python
import sys
import tensorflow as tf
import pickle
import numpy as np
```

지면을 중복해서 사용하지 않기 위해 지난 챕터들에서 정의했던 데이터 관련 함수들과 모델 구조 정의를 간단한 파이썬 파일로 따로 만들어 두었습니다.(mel_op.py / model.py)

이 파일을 로드하면 정의된 모든 함수를 사용할 수 있습니다.


```python
from mel_op import *
from model import model_RNN
```

현재는 유저의 미디 인풋을 받는 모듈 작업이 되어 있지 않기 때문에, 간단히 우리의 인풋 데이터 중 하나를 골라서 인풋으로 넣어보겠습니다.

이를 위해 저장했던 데이터를 다시 불러오겠습니다.


```python
preprocessed_dir = "./preprocessed_data/"

with open(preprocessed_dir + "input_sequences.p", "rb") as fp:   
    input_sequences = pickle.load(fp)

with open(preprocessed_dir + "vocab_size.p", "rb") as fp:   
    vocab_size = pickle.load(fp)

with open(preprocessed_dir + "mel_set.p", "rb") as fp:   
    mel_set = pickle.load(fp)

with open(preprocessed_dir + "mel_i_v.p", "rb") as fp:   
    mel_i_v = pickle.load(fp)
```


```python
## prepare as an batch
def get_input_batch_sequence(input_seq, sequence_length):
    
    input_sequence_batches = []

    num_seqs_per_song = max(int((len(input_seq) / sequence_length)) - 1, 0)

    for ns in range(num_seqs_per_song):
        batch = np.expand_dims(input_seq[ns * sequence_length:(ns+1) * sequence_length], axis=0)
        input_sequence_batches.append(batch)

    return np.array(input_sequence_batches)

def preprocess_user_input(mel_arr):
    curve_seq_list = []
    curve_seq_list.append(create_curve_seq(mel_arr))

    return curve_seq_list

def predict_output(curve_arr, mel_i_v, sequence_length = 8):

    ## prepare user input sequence with existing vocab in melody set
    user_input_sequence = []
    for curve in curve_arr:
        similar_curve = find_similar_curve(curve, mel_set)
        user_input_sequence.append(similar_curve)

    print(user_input_sequence)
    
    ## pad zeros to the user input sequence
    if len(user_input_sequence) < sequence_length:
        user_input_sequence += [0] * (sequence_length - len(user_input_sequence))

    input_sequence_as_batches = get_input_batch_sequence(user_input_sequence, sequence_length)

    with tf.Session() as sess:
        model = model_RNN(sess, 
                         batch_size=1, 
                         learning_rate=0.001,
                         num_layers = 3,
                         num_vocab = vocab_size,
                         hidden_layer_units = 64,
                         sequence_length = 8,
                         data_dir='generation_model/preprocessed_data/')

    output_sequence = model.predict(np.array(input_sequence_as_batches), mel_i_v)

    return output_sequence
```


```python
sequence_length = 8
user_input_file = '~/Google Drive/data/nottingham-dataset/MIDI/melody/ashover1.mid'
midi_obj = music21.converter.parse(user_input_file)
mel_data = create_mel_data_each_file(midi_obj)
mel_arr = []
for key in sorted(mel_data.keys()):
    mel_arr.append(mel_data[key])

curve_arr = create_curve_seq(mel_arr)
output_sequence = predict_output(curve_arr, mel_i_v, sequence_length)
```


```python
print(output_sequence)
```

    [(-2, 1.0), (-2, 1.0), (-2, 1.0), (-2, 1.0)]




### AI-MELODY-DUET 예시

아티스트로써 여러분의 창의력을 또한번 발휘할 수 있는 부분이라고 생각합니다. 일정 길이의 유저 인풋에 대해서 대응하는 멜로디를 AI가 연주하는 컨셉을 어떤 식으로 표현하는가에 따라 작품이 이루어질 수 있을 것입니다.

여기서는 하나의 예시로 좌우로 나뉜 화면에서 유저의 입력 멜로디와 AI가 생성한 멜로디를 번갈아 보여주는 시각화 페이지를 보겠습니다.

