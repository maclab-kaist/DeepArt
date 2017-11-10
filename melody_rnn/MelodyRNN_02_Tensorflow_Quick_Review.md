
## Tensorflow의 간단 Review

텐서플로우는 딥러닝에 필요한 각종 연산들을 내부적으로 구현해놓은 라이브러리입니다.

텐서플로우에 관한 강의를 이곳에서 모두 다룰 수는 없기 때문에, 온라인 상에 공유되어있는 좋은 강의들의 링크를 제공해드리고, 더불어 간단한 예제코드만을 살펴보도록 하겠습니다.

다음 예제코드는 온라인으로 공유된 튜토리얼 코드의 일부를 가져온 것입니다.

[https://github.com/aymericdamien/TensorFlow-Examples/](https://github.com/aymericdamien/TensorFlow-Examples/)

먼저, 텐서플로우를 import하고, 라이브러리 내부에 간단한 예제를 위해 제공하고 있는 데이터셋(MNIST)를 불러오겠습니다.


```python
# import tensorflow
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
```

    Extracting /tmp/data/train-images-idx3-ubyte.gz
    Extracting /tmp/data/train-labels-idx1-ubyte.gz
    Extracting /tmp/data/t10k-images-idx3-ubyte.gz
    Extracting /tmp/data/t10k-labels-idx1-ubyte.gz


우리가 구현할 모델은 간단한 인공신경망 모델입니다.

어떤 인풋에 대해 Activate/deactivate를 판별하는 뉴런들의 모임을 여러 레이어에 걸쳐서 연결하면 일반적인 함수로 표현하기 어려운 복잡한 논리구조를 근사해서 반영할 수 있다는 원리입니다.

<img src="./img/NN.png">


```python
# Parameters
learning_rate = 0.1
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
```


```python
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}
```


```python
# Create model
def model_simple_nn(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
```


```python
# Construct model
logits = model_simple_nn(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
```

이 예제에서는 텐서플로우 라이브러리에 포함되어 있는 토이 데이터셋을 사용하는데, Input 데이터와 Label 데이터를 미니배치 형태로 이쁘게 뽑을 수 있는 함수가 미리 정의되어 있기 때문에 간편하게 사용할 수 있습니다.

학습의 이터레이션 수만 원하는 만큼 지정해서 테스트해보겠습니다.


```python
num_steps = 3
```


```python
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                             Y: batch_y})
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
```

    Step 1, Minibatch Loss= 9521.5879, Training Accuracy= 0.258
    Step 2, Minibatch Loss= 12462.8232, Training Accuracy= 0.352
    Step 3, Minibatch Loss= 11920.3799, Training Accuracy= 0.492
    Optimization Finished!
    Testing Accuracy: 0.5152

