
## t-SNE

학습이 끝나면 t-SNE 기법을 이용해 샘플을 분석한 결과가 어떻게 분포되는지 보고자 합니다. t-SNE는 다차원 데이터의 분포를 더 적은 차원에서 효과적으로 보여주는 기법입니다. 예를 들어, 우리가 학습시킨 모델은 주어진 입력값을 여러 레이어를 거쳐 연산한 후에 최종적으로 256차원의 데이터로 변환합니다(fc2). 여기서 fc3 레이어를 거쳐 11개의 악기에 대한 확률 값을 얻어냅니다. 이 256차원의 데이터는 각 입력 데이터에서 악기를 결정하는 데 중요한 데이터를 추출한 것이라고 볼 수 있습니다. 그렇지만 사람이 인지하기에는 너무 높은 차원이기 때문에 t-SNE를 이용해 2차원이나 3차원으로 그 분포를 나타내면 보다 쉽게 데이터의 분포를 확인할 수 있습니다.

먼저 학습된 모델에서 테스트 데이터를 이용해 fc2 레이어의 값을 추출해 보겠습니다.

activation.py 파일을 만들고 다음 내용을 적어주세요.

```python
import tensorflow as tf
import data_provider
import model

n_labels = data_provider.n_labels
data_provider.batch_size = 1
batch_size = 1
sequence_length = data_provider.sequence_length
feature_dimension = data_provider.feature_dimension

def get_activation():
  data_provider.prepare_data()
  n_samples = len(data_provider.test_samples)

  with tf.Graph().as_default():
    X = tf.placeholder(tf.float32, shape=(batch_size, sequence_length, feature_dimension, 1))
    phase_train = tf.placeholder(tf.bool)

    activations = numpy.zeros((n_samples, 256))

    output = model.get_last_minus_one(X, phase_train, sequence_length, feature_dimension, n_labels)

    sess = tf.Session()
    saver = tf.Saver(tf.all_variables())
    saver.restore(sess, 'model.ckpt')

    for step in range(n_samples):
      x, y = data_provider.get_sample_at('test', step)
      activation = sess.run(output, feed_dict={phase_train:False, X:x})
      activations[step, :] = activation

    numpy.save('activation.npy', activations)
```

tsne.py 파일을 다음과 같이 만들어 주세요.

```python
import numpy

act = numpy.load('activation.npy')

from sklearn.decomposition import PCA
pca = PCA(n_components=50)
act_pca = pca.fit_transform(act)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
act_tsne = tsne.fit_transform(act_pca)

tsne = TSNE(n_components=3)
act_tsne_3d = tsne.fit_transform(act_pca)


numpy.save('act_pca.npy', act_pca)
numpy.save('act_tsne.npy', act_tsne)
numpy.save('act_tsne_3d.npy', act_tsne_3d)
```
