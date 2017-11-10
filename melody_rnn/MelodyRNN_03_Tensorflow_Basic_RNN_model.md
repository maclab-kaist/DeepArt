
## Basic RNN implemented with Tensorflow

우리가 사용할 모델은 가장 기본적인 Recurrent Neural Network 구조입니다.

이 구조는 기본적으로 단일 방향의 정보 흐름을 학습해서 이전 스텝까지의 정보를 바탕으로 다음 스텝의 내용을 출력하도록 학습됩니다.

즉, 중간의 hidden layer에 지금까지의 정보가 누적 학습됨으로써 다음 스텝의 출력 내용에 이를 반영되는 방식입니다.

기존의 단순한 Markov assumption을 기반으로 한 확률모델과의 차이점은 1) 정보 처리가 인공신경망 구조를 통해 이루어진다는 점 2) Hidden layer에 바로 전 스텝의 정보 이외에도 누적된 과거의 정보가 누적 학습된다는 점 입니다. 우리가 사용한 모델은 한번에 하나씩 엘리먼트를 받아들여서 다음번 엘리먼트를 예측하므로써 전체 정보 시퀀스의 흐름을 학습할 수 있는 Char-RNN 구조입니다.

큰 그림에서 비슷하게 느껴지는 마코프 성질 기반 모델들과의 차이점은, 확률 모델이 아닌 어떤 사고체계가 이를 수행한다는 점입니다. 즉, RNN Cell이 어떤 시퀀셜한 흐름에 대한 논리구조를 학습한다는 정도로 러프하게 이해해도 괜찮습니다.

좀 더 깊이 Char-RNN에 대한 이해를 하기 위해서는 다음 웹페이지를 참고하셔도 좋습니다.

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

물론, bidirectional한 RNN을 만들어서 두가지 방향의 정보 흐름을 반영할 수도 있고, attention이라는 개념을 추가해서 시퀀스 내 특정 위치에 대해 가중치를 더하는 논리 구조를 학습시키는 등 여러 advanced한 방법론이 쓰일 수도 있지만 이 강좌에서는 우선 가장 기본적인 모델로 학습해보고자 합니다.

또 한가지 언급할 내용은 우리가 pure RNN cell이 아닌 LSTM cell을 사용한다는 점입니다.
Long Short Term Memory의 약자인 LSTM구조를 아주 간략히 설명하자면, hidden layer에 addtive한 함수 구조를 추가해서 유용한 정보를 오랫동안 기억할 수 있는 능력을 부여한 구조라고 할 수 있습니다. 
Original RNN의 단점은 시퀀스가 길어질 수록 정보의 흐름이 흐려지면서 학습이 잘 이루어지지 않는 것이었는데, 이를 보완한 구조라고 보면 되겠습니다.

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

이 LSTM의 수식은 이미 텐서플로우 라이브러리에 모두 구현되어 있으므로 우리는 이를 잘 활용하는 방법을 알아보도록 하겠습니다.




```python
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
from text_loader import TextLoader
```


```python
num_layers  = 3
hidden_size = 512
batch_size  = 200
max_length  = 30
learning_rate = 0.001

loader = TextLoader("data/hamlet.txt")
vocab_size = len(loader.vocab)

if not os.path.exists("basic_checkpoints/"):
    os.makedirs("basic_checkpoints/")
```


```python
X = tf.placeholder(tf.int32, [None, max_length])
y = tf.placeholder(tf.int32, [None, max_length]) # [N, seqlne]

x_one_hot = tf.one_hot(X, vocab_size)
y_one_hot = tf.one_hot(y, vocab_size)            # [N, seqlen, vocab_size]

cells = [rnn.BasicLSTMCell(hidden_size) for _ in range(num_layers)]
cells = rnn.MultiRNNCell(cells, state_is_tuple=True)

initial_state = cells.zero_state(batch_size, tf.float32)
outputs, states = tf.nn.dynamic_rnn(cells, x_one_hot, 
    initial_state=initial_state, dtype=tf.float32)

outputs = tf.reshape(outputs, [-1, hidden_size]) # [N x seqlen, hidden]
logits = layers.linear(outputs, vocab_size)      # [N x seqlen, vocab_size]
y_flat = tf.reshape(y_one_hot, [-1, vocab_size]) # [N x seqlen, vocab_size]
```


```python
loss_op = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_flat) # [N x seqlen]
loss_op = tf.reduce_mean(loss_op)

opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

y_softmax = tf.nn.softmax(logits)         # [N x seqlen, vocab_size]
pred = tf.argmax(y_softmax, axis=1)       # [N x seqlen]
pred = tf.reshape(pred, [batch_size, -1]) # [N, seqlen]
```


```python
saver = tf.train.Saver()

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(3):
        batch_X, batch_y = loader.next_batch(batch_size, max_length)
        loss, _ = sess.run([loss_op, opt], feed_dict={X: batch_X, y: batch_y})
        
        if (step+1) % 1 == 0:
            print("{:08d} step, loss:{:.4f}".format(step+1, loss))
            
            random = np.random.randint(0, batch_size)
            results = sess.run(pred, feed_dict={X: batch_X})
            words = [loader.words[word] for word in results[random]]
            print("".join(words))
     
        if (step+1) % 1 == 0: 
            saver.save(sess, "basic_checkpoints/char-rnn_"+str(step+1))
```

Prediction을 위해서는 batch_size를 1로 만들어줄 필요가 있습니다.

**주의: 새로운 Jupyter Notebook을 만들어서 다시 정의해주세요.**


```python
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
from text_loader import TextLoader

num_layers  = 3
hidden_size = 512
batch_size  = 1
max_length  = 1000

loader = TextLoader("data/hamlet.txt")
vocab_size = len(loader.vocab)
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


```python
sentence = list()
# 시작 글자 생성
sentence += loader.X[:10].tolist()
print("Start with:", "".join([loader.words[char] for char in sentence]))

saver = tf.train.Saver()
sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "basic_checkpoints/char-rnn_3")
    
    # 매 이터레이션마다 글자 하나씩 생성
    pred_char, state = sess.run([pred, states], feed_dict={X:[sentence[:]]})
    for i in range(max_length):
        # 이전 스텝에 갖고 있는 state 값을 다음 스텝에 넣어줌
        pred_char, state = sess.run([pred, states], 
            feed_dict={X:[[sentence[-1]]], initial_state: state})
        sentence.append(pred_char[0][-1])
        
sentence = [loader.words[char] for char in sentence]
print("".join(sentence))
```

    Start with: <START>	HAMLET
    
    
    INFO:tensorflow:Restoring parameters from basic_checkpoints/char-rnn_3
    <START>	HAMLET
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            


### (optional) 조금 더 텐서플로우로 RNN을 잘 다루기 위한 추가적인 사항들

우리가 실제로 RNN을 정의하고 학습할 때에는 위의 구조보다는 더 다양한 구조로 응용할 필요가 있을 것입니다. 

이를 위해 아주 조금만 더 RNN관련 정의된 함수들의 활용 방법을 알아봅시다.


텐서플로우에서 RNN을 사용하는 방식은 다음과 같습니다.  

- Cell을 정의한다. (BasicLSTMCell 등의 구현된 클래스 사용. 내부에 연산한 후 output값과 state값을 넘겨주는 구조가 정의되어 있음.)



```python
cell = tf.contrib.rnn.BasicLSTMCell(num_hidden_unit,
                                    state_is_tuple=True)
```

- RNN cell 내부에서 받을 hidden state값(previous step으로부터 넘겨받는)에 대한 초기값을 지정하기 위해서 일단 cell모양 그대로 0을 채워놓은 텐서를 저장해놓습니다.


```python
initial_state = cell.zero_state(batch_size, tf.float32)
```

- 정의된 cell들을 가지고 (static 혹은) dynamic rnn 구조를 정의합니다. 여기서 최종 레이어 단계에서의 output sequence과 최종 state 값을 리턴받습니다. (만약 이 스테이트 값을 다음 iteration에서 쓰고자 한다면 받아오고, 아닐 경우 사용하지 않음.)


```python
output, _ = tf.nn.dynamic_rnn(cell, input_tensor, sequence_length,
                              time_major=False, dtype=tf.float32)
```

중요한 부분은 input_tensor로 받아오는 텐서 (즉 인풋)의 shape을 가지고 알아서 time step의 길이를 추정한다는 것입니다. (sequence_length부분을 생략해도 됨.)

    Batch size x time steps x features


이 부분에 유동성을 위해서 time_major라는 argument가 쓰이는데, 보통 [batch_size, num_steps, state_size]의 꼴로 처리를 하지만 이것을 True로 설정하면 [num_steps, batch_size, state_size]의 꼴로 처리합니다. 특정 스텝에서의 결과값을 얻어내는 데에 유용하게 쓰일 수 있습니다.

혹은 이 때 시퀀스의 길이가 일정하다면 static RNN을 사용해도 상관없습니다.(하지만 메모리를 미리 잡는 이슈가 있어서 그냥 dynamic_rnn을 사용하는 것이 좋습니다.)


```python
output, _ = tf.nn.static_rnn(cell, input_tensor,
                             dtype=tf.float32)
```

- 만약 멀티 레이어 RNN을 사용하고 싶다면, RNN셀을 각각 생성한 뒤에 이를 리스트로 묶어서 tf.contrib.rnn.MultiRNNCell안에 인풋으로 넣어주면 됩니다.


```python
rnn_cells = tf.contrib.rnn.MultiRNNCell([cell1, cell2])
```

혹은 다음과 같이 아예 cell을 생성하는 함수를 만들면 더 편리합니다.


```python
# RNN cell layer generating function
def create_rnn_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size,
                                        state_is_tuple = True)
    return cell
 
# 2 layers for hidden layer
multi_cells = tf.contrib.rnn.MultiRNNCell([create_rnn_cell()
                                           for _ in range(2)],
                                           state_is_tuple=True)
```

그리고 정의된 multi_cells를 가지고 static 혹은 dynamic rnn 구조를 정의합니다.


```python
outputs, _= tf.nn.dynamic_rnn(multi_cells, x_data,
                                     dtype=tf.float32)
```

tf.nn.dynamic_rnn을 처리한 output의 dimension은 cell의 크기와 동일하게 됩니다. (cell에 지정한 num_units 만큼의 output의 dimension이 자동으로 결정됩니다.)

즉, [batch_size, sequence_length, input_dim]을 인풋으로 넣으면 [batch_size, sequence_length, num_units]의 output이 나오게 되는 것입니다.

이렇게 얻은 dynamic_rnn의 아웃풋에 대해서 보통 마지막 레이어에 Fully-connected layer를 추가하는데, 이 작업을 한 시퀀스 내의 타임스텝 전체에 대해서 행렬 연산으로 한번에 하기 위해서 batch_size * sequence_length 부분을 flat하게 펴줄 필요가 있습니다.


```python
rnn_output_flat = tf.reshape(outputs, [-1, hidden_size]) # output : [N x sequence_length, hidden_size]
```

여기에 Fully connected layer를 더해줍니다. 이 때 아웃풋의 dimension을 one-hot encoding으로 정의한 vocabulary이 크기로 지정해줍니다.


```python
logits = tf.contrib.layers.fully_connected(rnn_output_flat, num_vocab, None) # output : [N x sequence_length, vocab_size]
```

이렇게 얻은 logits들에 softmax function을 이용해서 vocab들 중에 모델이 예측하는 가능성의 정도를 측정합니다.


```python
y_softmax = tf.nn.softmax(self.logits) # outputs : [N x sequence_length, vocab_size]
```

이렇게 얻은 값을 이용해서 가장 높게 측정한 character를 뽑는 것이 모델의 prediction 값이 됩니다.


```python
pred = tf.argmax(y_softmax, axis=1) # outputs : [N x sequence_length]
```

아까 Fully-connected layer의 계산을 위해서 flatten했던 데이터를 다시 (배치 사이즈, 시퀀스)의 형태로 만들어주면, 한 미니배치에 대한 우리 모델의 예측값이 완성되는 것입니다.


```python
pred = tf.reshape(pred, [batch_size, -1]) # [N, sequence_length]
```

위의 몇 스텝은 prediction을 위한 부분이었지만, 모델의 학습을 위해서는 가장 높은 값을 뽑는 prediction 이전 단계의 logits값을 가지고 Loss를 계산하는 부분이 필요합니다.

텐서플로우에 미리 정의된 Cross-entropy 함수를 사용해서 각 timestep의 Loss를 계산하고, 전체 timestep에 대해서 평균을 낸 값으로 최종 Loss를 얻게 됩니다. (이 값이 BPTT를 통해서 모든 weight parameter 업데이트에 활용됩니다.)


```python
losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_flat, logits=logits)
sequence_loss = tf.reduce_mean(losses)
```

그 외의 optimizer를 사용하는 부분은 앞서 살펴본 기본 모델과 동일합니다. 

Optimizer만 지정해주면, 알아서 내부에 구현된 해당 방법으로 parameter 업데이트를 진행하게 됩니다.


```python
opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(sequence_loss)
```
