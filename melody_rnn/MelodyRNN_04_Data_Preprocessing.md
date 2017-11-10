
## 라이브러리 준비

먼저 필요한 라이브러리를 다시 불러옵니다. 



```python
import music21
import numpy as np
import os
import pickle
```

저장해놓았던 데이터를 다시 불러옵시다. 우리는 멜로디의 음정, 오프셋(시작지점) 데이터를 저장해놓았었습니다.|


```python
preprocessed_dir = "./preprocessed_data/"

with open(preprocessed_dir + "mel_arr_list.p", "rb") as fp:   
    mel_arr_list = pickle.load(fp)

```

## 데이터 Preprocessing - Convert into sequential melodic form.

우리가 원하는 것은 신경망으로 하여금 멜로디 시퀀스의 진행 특성을 학습하도록 하는 것입니다.

그리하여 새로운 멜로디 인풋이 들어왔을 때, 그 뒤에 이어지는 음의 시퀀스가 생성되도록 하는 것입니다.

우리가 사용할 RNN이라는 구조는 시간의 흐름을 반영한 네트워크 구조인데, 각 시간 스텝마다 하나의 인풋을 받아서 이전까지의 시간 스텝을 통과하면서 전달받아온 정보와 함께 프로세싱한 후 아웃풋을 생성하는 모델입니다. (이때, 다음 시간 스텝에 넘겨줄 정보도 생성한 후 전달합니다.)

이를 통해 어떤 순서에 따라 어떤 입력이 들어올 때 어떤 출력값을 내어 놓는지에 대한 논리를 학습하도록 되어 있습니다.

이를 멜로디 학습에 적용하기 위한 개요를 적어봅시다.

(1) 음정  
(2) 음길이  
(3) 화성학 관점에서의 해석  
(4) Sequential Contour   


## 데이터 구조 정의

우리가 사용할 모델은 가장 기본적인 Recurrent Neural Network 구조입니다. 

이 구조는 기본적으로 단일 방향의 정보 흐름을 학습해서 이전 스텝까지의 정보를 바탕으로 다음 스텝의 내용을 출력하도록 학습됩니다.  
즉, 중간의 hidden layer에 지금까지의 정보가 누적 학습됨으로써 다음 스텝의 출력 내용에 이를 반영되는 방식입니다.

기존의 단순한 Markov assumption을 기반으로 한 확률모델과의 차이점은 
1) 정보 처리가 인공신경망 구조를 통해 이루어진다는 점
2) Hidden layer에 바로 전 스텝의 정보 이외에도 누적된 과거의 정보가 누적 학습된다는 점
입니다.

물론, bidirectional한 RNN을 만들어서 두가지 방향의 정보 흐름을 반영할 수도 있고, attention이라는 개념을 추가해서 시퀀스 내 특정 위치에 대해 가중치를 더하는 논리 구조를 학습시키는 등 여러 advanced한 방법론이 쓰일 수도 있지만 이 강좌에서는 우선 가장 기본적인 모델로 학습해보고자 합니다.

또 한가지 언급할 내용은 우리가 pure RNN cell이 아닌 LSTM cell을 사용한다는 점입니다.
Long Short Term Memory의 약자인 LSTM구조를 아주 간략히 설명하자면, hidden layer에 addtive한 함수 구조를 추가해서 유용한 정보를 오랫동안 기억할 수 있는 능력을 부여한 구조라고 할 수 있습니다. 
Original RNN의 단점은 시퀀스가 길어질 수록 정보의 흐름이 흐려지면서 학습이 잘 이루어지지 않는 것이었는데, 이를 보완한 구조라고 보면 되겠습니다.

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

이 LSTM의 수식은 이미 텐서플로우 라이브러리에 모두 구현되어 있으므로 우리는 이를 잘 활용하는 방법을 알아보도록 하겠습니다.



딥러닝 연구자들은 어떤 정보의 흐름을 효과적으로 학습시키는 모델들을 계속해서 연구하고 발전시키고 있습니다. 
이를 활용해서 아티스트로써 의미있는 작업을 하는 일은 어떤 방식이 될 수 있을까요?

다양한 접근방식이 있을 수 있겠으나, 그중 하나는 **'무엇을 학습시킬것인가'**일 것입니다.

즉, 어떤 시퀀셜한 정보를 학습하는 모델을 사용할 때, 어떤 시퀀스를 학습해서 어떤 아웃풋을 얻을까에 대한 고민이 바로 아티스트가 딥러닝 기술을 작품에 접목할 수 있는 접점이 될 수 있는 것입니다.

일단 우리의 목적은 다양한 곡을 학습시켜서 비슷한 곡을 만들어내는 것이라고 정했습니다. 
음악은 시간에 따른 정보의 나열이라고도 볼 수 있는데, 이것을 어떤 식으로 해석해서 데이터 구조를 정의하느냐는 우리의 음악적 지식 및 감각, 그리고 창의력이 투입될 수 있는 부분입니다.

오늘은 아주 간단한 두가지 접근으로 음악적 의미를 갖는 데이터 구조를 정의하고, 이를 실제 모델에 학습시켜보도록 하겠습니다.


### (1) Pianoroll 형식의 데이터

가장 쉽게 접근할 수 있는 데이터 형식입니다. 세로축을 88개의 음정으로 구성하고 가로축을 16분 음표 단위의 시간으로 구성하는 형태입니다.

이런 데이터 구조로 모델을 학습시킨 시도로는 다음 링크의 예제가 있습니다.

http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/



### (2) 노트 시퀀스 형식의 데이터 

역시 쉽게 접근할 수 있는 데이터 형식입니다. 노트의 음정, 음의 시작지점(offset) 한 묶음으로 해서 이 노트들의 시퀀스로 데이터를 정의합니다.

다음 링크는 펫메시니의 연주부분(기타솔로)을 가지고 노트 정보의 시퀀스로 데이터를 정의한 예제입니다. 한가지 더 음악적인 아이디어를 더한 것은, 해당멜로디가 속한 마디의 화성을 고려해서 노트 정보를 단순히 음정이 아닌 코드에서의 역할로 치환하여 데이터를 정의했다는 점입니다.

https://github.com/jisungk/deepjazz

### (3) 멜로디 변화량 벡터 데이터

또 한가지 시도해볼 접근은 각 노트 사이의 변화량(시간 변화, 음정 변화)을 벡터로 정의해서 전체 멜로디를 이런 변화량의 시퀀스로 나타내는 방법입니다. 

멜로디라는 것은 결국 어떤 음의 높낮이의 변화의 흐름이라는 아이디어에서 착안한 접근이라고 볼 수 있습니다.



이외에도 다양한 형태로 음악의 essence를 담아내는 구조를 창안할 수 있습니다. 또한 우리는 지금 단일 멜로디만을 다루고 있지만, polyphonic 정보를 다루기 위해서는 또 다른 형식의 구조가 필요할 것입니다. 

여러분의 몫이 될 수 있습니다.

오늘은 이 중 세번째 음과 음 사이의 변화량을 벡터로 정의한 시퀀스 모델링을 진행해봅시다.

우리는 받아왔던 멜로디 시퀀스 데이터에서 노트를 하나씩 불러와서 바로 이전 노트와의 차이를 벡터로 만드는 작업이 필요합니다. 편의상 이것을 '멜로디 커브'라고 부르겠습니다.


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
```

    1034


1034개의 곡에서 멜로디 커브의 시퀀스를 추출해낸 것을 확인했습니다.

이제 이렇게 추출한 커브들에 ID를 부여해서 신경망이 학습하기 좋은 형식으로 변환하는 작업을 할 차례입니다.



### One-hot encoding을 위한 딕셔너리 준비

신경망 학습을 잘하기 위해서는 보통 one-hot encoding을 해서 데이터를 준비합니다. 이를 위해 모든 유니크한 멜로디 커브 엘리먼트에 대한 딕셔너리를 구성하고, 각각에 ID 인덱스를 부여합니다.

먼저 모든 멜로디 커브 데이터를 flatten합니다.


```python
# flatten
curve_corpus= sum(curve_seq_list, [])
print(len(curve_corpus))

# flattened seq
mel_seq = curve_corpus
```

    195174


전체 시퀀스를 이어붙이니 195174개의 시퀀스가 되었습니다.

이중 중복되지 않는 데이터만 남기는 작업은 python의 내장함수 set을 사용해서 간단히 만들 수 있습니다.


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


중복되는 엘리먼트를 제거하고 422개의 유니크한 벡터들의 딕셔너리가 만들어졌음을 확인할 수 있습니다.

여기서 한가지 추가작업이 필요합니다. 

만약 우리가 유저로부터 어떤 입력을 받아서 이에 이어지는 시퀀스를 생성하도록 하고 싶다면, 새로 받아온 입력을 똑같은 멜로디 커브 형태로 만들어주어야 합니다.

이때 기존 커브 데이터 딕셔너리에 없는 커브가 들어올 수 있기 때문에, 이를 최대한 비슷한 커브로 대체하는 작업이 필요합니다.

이 작업을 하는 함수를 'find_similar_curve'라고 정의합시다. 

이 함수는 새로운 입력에서 멜로디 커브 벡터를 만들어 낸 후 이 커브가 딕셔너리에 없다면 Euclidean distance가 가장 가까운 커브로 대체하도록 하는 합수입니다.


```python
def truncate(f, n):
    s = '%.12f' % f
    i, p, d = s.partition('.')
    return float('.'.join([i, (d+'0'*n)[:n]]))

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

이제 모델에 실제로 넣을 인풋/아웃풋을 만들어주는 함수를 정의할 차례입니다.

다시 언급하자면, 우리가 사용할 모델은 Char-RNN이라고 불리는 단순한 모델로, 한번에 한 캐릭터씩을 인풋으로 받아서 다음 캐릭터를 생성하는 모델입니다. 따라서, 인풋으로 어떤 시퀀스를 준비한 후 이 인풋에 대한 '정답 데이터', 즉 레이블 데이터는 한 스텝 이후의 캐릭터들의 시퀀스라고 볼 수 있습니다.

좀 전에 정의한 멜로디 커브 ID를 사용해서 이 ID의 시퀀스인 인풋데이터를 만들고, 한칸씩 뒤로 밀린 레이블데이터 또한 정의하도록 하겠습니다.



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
sequence_length = 8
input_seq_list, label_seq_list = get_mel_id_sequence(mel_arr_list, curve_seq_list)
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
input_sequences, label_sequences = get_batch_sequence(input_seq_list, label_seq_list, sequence_length)
```


```python
preprocessed_dir = "./preprocessed_data/"

with open(preprocessed_dir + "input_sequences.p", "wb") as fp:   #Pickling
    pickle.dump(input_sequences, fp)

with open(preprocessed_dir + "label_sequences.p", "wb") as fp:   #Pickling
    pickle.dump(label_sequences, fp)

with open(preprocessed_dir + "vocab_size.p", "wb") as fp:   #Pickling
    pickle.dump(vocab_size, fp)

with open(preprocessed_dir + "mel_set.p", "wb") as fp:   #Pickling
    pickle.dump(mel_set, fp)

with open(preprocessed_dir + "mel_i_v.p", "wb") as fp:   #Pickling
    pickle.dump(mel_i_v, fp)

```
