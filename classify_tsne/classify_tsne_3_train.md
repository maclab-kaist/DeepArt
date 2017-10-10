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