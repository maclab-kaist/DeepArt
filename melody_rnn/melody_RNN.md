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
     ...
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

    ...


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
