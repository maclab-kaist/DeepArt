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



<br/><br/>
<h2><a name="d2" style="color:black; text-decoration:none;">Deep Dream 나만의 그림 만들기</a></h2>

Deep dream을 이용하여 나만의 추상적인 그림을 만들어 보고자 합니다.<br/><br/>

<a href="https://deepdreamgenerator.com/generator" style="text-decoration:none;transition: color ease 0.7s;" target="_blank">"DEEP DREAM GENERATOR"</a>라는 기성 사이트를 이용하여 우리는 가지고 있는 이미지에 원하는 추상성을 손쉽게 덧입힐 수 있습니다.

먼저, 로그인을 한 후 (페이스북, 구글 플러스, 트위터 계정으로 손쉽게 로그인 할 수 있습니다), 
<p align="center"> 
<img src="./login.png">
</p><br/>

우측 상단의 "Generate" 버튼을 클릭하시면, 
<p align="center"> 
<img src="./generate.png">
</p><br/>

다음과 같은 페이지로 이동합니다.<br/>
"Deep Style", "Thin Style"와 같은 내용은 다음 튜토리얼에서 다룰 것이기에 본 페이지에서는 바로 "Deep Dream"항목을 클릭합니다. (다음 튜토리얼 Style Transfer 내용을 숙지하신 후 역시 해당사이트에서 바로 이용하실 수도 있습니다.)
<p align="center"> 
<img src="./style.png">
</p><br/>

아래와 같은 화면이 나오는데, "Settings"를 클릭하면 아래와 같이 인공신경망의 다양한 계층들을 선택할 수 있게 나옵니다. (아래의 "Show All layers"를 클릭하시면 더욱 많은 계층을 볼 수가 있습니다.)
<p align="center"> 
<img src="./settings.png">
</p><br/>
<p align="center"> 
<img src="./layers.png">
</p><br/>

원하는 풍의 이미지를 선택하신 후, 맨 밑의 아래 "Generate"버튼을 클릭하시면 최종 결과물을 얻을 수 있습니다.
<p align="center"> 
<img src="./run.png">
</p><br/>

위와 같은 방법을 통하여 우리는 다양한 작품을 만들 수가 있는데,<br/> 아래 링크들은 Deep Dream을 이용한 작가들의 작품 예시입니다.<br/>
<a href="http://www.miketyka.com/?s=deepdream" style="text-decoration:none;transition: color ease 0.7s;" target="_blank">http://www.miketyka.com/?s=deepdream</a><br/>
<a href="http://barabeke.it/portfolio/deep-dream-generator" style="text-decoration:none;transition: color ease 0.7s;" target="_blank">http://barabeke.it/portfolio/deep-dream-generator</a><br/>




<br/><br/>
<h2><a name="d3" style="color:black; text-decoration:none;">Deep Dream 코딩 해보기</a></h2>

위의 두 섹션에서 Deep Dream에 대하여 알아보고,<br/> 
기성 웹 페이지를 이용하여 직접 생성해보는 방법에 대하여 알아보았습니다.<br/><br/>
본 섹션에서는 내용을 좀 더 심화하여 직접 코딩하면서 해당 내용을 수행해보고자 합니다. <br/>
본 내용은 <a href="https://github.com/fchollet/keras/blob/master/examples/deep_dream.py" style="text-decoration:none;transition: color ease 0.7s;" target="_blank">케라스 Deep Dream 튜토리얼</a> 내용을 기반으로 작성하였습니다.

<a href="./deepdream.py" style="text-decoration:none;transition: color ease 0.7s;" target="_blank">해당 코드</a>와 <a href="./sample.jpg" style="text-decoration:none;transition: color ease 0.7s;" target="_blank">샘플 이미지</a>를 다운로드 하신 후,<br/> "python deepdream.py sample.jpg dream"의 커멘드 만으로 직접 코드를 돌려보실 수 있습니다.<br/>

해당 코드를 자세히 들여다 보도록 하겠습니다.<br/><br/>
먼저 아래에 사용된 dependency를 준비해야 합니다. 
```
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import scipy
import argparse

from keras.applications import inception_v3
from keras import backend as K
```
<br/>

이제 학습된 인공신경망을 불러와야 합니다. ImageNet이라는 큰 데이터셋에서 학습한 모델을 아래의 코드로 쉽게 불러올 수 있습니다.
```
# Build the InceptionV3 network with our placeholder.
# The model will be loaded with pre-trained ImageNet weights.
model = inception_v3.InceptionV3(weights='imagenet',
                                 include_top=False)
dream = model.input
print('Model loaded.')
```
<br/>


인공신경망 내의 각 층들이 어떤 이름을 가지고 있는지를 알아보기 위하여, 다음의 작업을 수행하면 층의 이름이 layer_dict에 저장됩니다.
```
# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])
print(layer_dict)
```
<br/>


그러고 나면, 사용할 층을 선택해주어야 합니다.<br/>
단일 층으로 할 수도 있으나 아래와 같이 여러층의 다양한 조합을 해볼 수 있습니다.
```
settings = {
    'features': {
        'mixed2': 0.2,
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed5': 1.5,
    },
}
```
<br/>


이제 부터가 중요한 부분입니다.<br/>
선택한 층의 뉴론 반응 수치를 최대화 하기 위하여 아래의 코드를 사용합니다. coeff라는 변수에 위에서 설정한 선택한 각 층의 0.2, 0.5와 같은 수치를 저장하고, K.sum(K.square(x[:, :, 2: -2, 2: -2]))와 같이 해당 층의 최외각 pixel에 해당하는 수치를 제외한 (가장자리의 데이터까지 포함하면 checker-border artifact라는 현상이 생기는데 이를 피하기 위하여 위와 같이 합니다.) 모든 수치를 제곱하고 더하여 하나의 숫자로 표현합니다. 이 숫자가 Seed Image에 대한 해당 층에서의 반응 수치를 대표하게 됩니다. 우리는 이 수치를 최대화하도록 Seed Image에 해당 수치의 back-propagation값을 계속해서 Seed Image에 덧입힙니다. 

```
# Define the loss.
loss = K.variable(0.)
for layer_name in settings['features']:
    # Add the L2 norm of the features of a layer to the loss.
    assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
    coeff = settings['features'][layer_name]
    x = layer_dict[layer_name].output
    # We avoid border artifacts by only involving non-border pixels in the loss.
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    if K.image_data_format() == 'channels_first':
        loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
    else:
        loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling
```
<br/>

Back-propagation해서 나오는 값을 gradients라고 하는데 이를 정의합니다.
```
# Compute the gradients of the dream wrt the loss.
grads = K.gradients(loss, dream)[0]
# Normalize gradients.
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
```
<br/>

여기에서 dream이란 Seed Image를 말합니다. 이러한 인풋에 대한 해당 층에서의 반응 수치 대표값을 최대화하도록 back-propagation하여 gradients값이 나오도록 정의합니다.

```
# Set up function to retrieve the value
# of the loss and gradients given an input image.
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values
```
<br/>


이렇게 나오는 gradients값을 인풋 Seed Image에 더하면, back-propagation이라는 과정의 원리상, 해당층의 뉴론반응수치를 더욱 높게 만들게 됩니다. 아래의 x += step * grad_values 수식이 gradient값을 인풋인 x에 누적하여 더함을 의미합니다. 이 과정을 gradient ascent라고 부릅니다. 여기에서 step은 gradients값을 조정하는 수치로 아래에서 임의로 지정하게 됩니다.
```
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x
```
<br/>


이러한 gradient ascent 방법에서 원본 이미지를 스케일링하여 다양한 사이즈에서 deep dream하면 더 좋은 결과를 얻을 수 있습니다. 
이에 아래와 같은 resize_img함수를 정의합니다.
```
def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)
```
<br/>


이제 gradient ascent 과정에 쓰이는 파라미터들을 정의합니다.
- step은 Seed Image에 gradients를 더할때 쓰이는 수치입니다.
- num_octave는 위와 같은 multi-scale방법을 사용할 때 몇번의 scaling을 할지 정하는 수치입니다.
- octave_scale은 scale간의 크기 비율을 나타냅니다.
- iterations는 매 scale마다 몇 번의 gradients를 연산할지를 정합니다.
- max_loss는 iteration을 꽉 채우지 않더라도 해당 수치만큼 loss가 달성되면 deep dream이 충분히 이루어졌다고 보고 연산을 중지하도록 하는 수치입니다.
<br/>
```
# Playing with these hyperparameters will also allow you to achieve new effects
step = 0.01  # Gradient ascent step size
num_octave = 3  # Number of scales at which to run gradient ascent
octave_scale = 1.4  # Size ratio between scales
iterations = 20  # Number of ascent steps per scale
max_loss = 10.
```
<br/>


아래의 과정은 다양한 크기를 가지는 Seed Image의 크기에 맞춰서 위에서 정의한 num_octave와 octave_scale에 따라 어떤 중간 scale값을 가져야하는지 구하는 코드입니다. 기본적으로 Seed Image의 크기를 기준으로 num_octave의 개수만큼 octave_scale값을 나누면서 다른 scale의 값을 정의합니다.
```
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])
```
<br/>


마지막으로, 위에서 정의한 모든 변수와 함수들을 가지고 아래의 과정을 수행하면 최종 deep dream된 이미지가 저장됩니다.<br/>
successive_shapes의 개수만큼 연산을 수행하고, 매 수행시에는 resize_img를 통하여 이미지를 해당 스케일의 크기를 갖도록 조정한 후, 
gradient_ascent를 통하여 deep dream을 수행합니다. 여기에서 이미지 resize에서 나오는 손실을 보정해주기 위한 코드가 존재합니다.
그리고 마지막으로 결과를 저장합니다.
```
for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)

save_img(img, fname=result_prefix + '.png')
```
<br/>
<br/>

위와 같이 Deep Dream에 대한 소개, 직접 그림 만들어보기, 코딩 해보기 까지 둘러보았습니다.<br/>
위의 내용에 대하여 추가적으로 궁금하신 점이 있다면 아래의 메일주소로 연락주시면 답변드리도록 하겠습니다. 읽어주셔서 감사합니다~! <br/>
<p align="right"> 
<b>이종필</b><br/>
<b>카이스트 문화기술대학원</b><br/>
<b>jongpillee.brian@gmail.com</b>
</p><br/>


