
## 모델 정의 및 학습

이번 챕터에서는 우리가 준비한 데이터셋에 실제로 사용할 RNN 모델을 정의하고 학습해보도록 하겠습니다.


### 모델의 인풋/아웃풋 구조 준비

우리가 사용할 Char-RNN구조는 일정 길이의 시퀀스 단위로 인풋을 처리합니다. 

즉, 일정 길이의 인풋 시퀀스를 받아서 시퀀스 내의 각 스텝이 다음 스텝의 인풋과 같은 엘리먼트를 아웃풋으로 출력하도록 학습되게 하는 것입니다.

Char-RNN은 한 스텝의 인풋 씩 받아서 다음 스텝을 생성하는 모델이지만, 학습할 때는 일정 길이의 시퀀스 단위로 학습을 하도록 합니다. 


* 사실 원래 개념상으로는 데이터 전체에 대해서 RNN이 backpropagation하면서 학습이 이루어져야 합니다.하지만 메모리의 문제 때문에 그렇게 할 수 없으므로 일정 길이의 묶음으로 나누어서 학습을 하기 위해서 이런 방식을 사용합니다. (물론, Sequence-to-sequence 모델에서는 우리의 목적에 따른 길이의 데이터 시퀀스로 학습을 하겠죠.)


따라서, 학습할 때에는 적당한 길이로 잘라서 학습하고, 나중에 생성할 때에는 한번에 한 엘리먼트씩 생성됩니다. (생성된 아웃풋을 다음 스텝의 인풋으로 사용.)




(즉, 제한된 sequence길이 내에서 unrolling을 하면서 학습이 이루어지도록 모델을 설계합니다. 대신, 학습 완료 후 한번에 하나씩 생성시에는 바로 전 스텝으로부터 hidden state만 받아오면 되기 때문에 길이에 제한없이 생성이 가능합니다.)


```python
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time, os
import pickle
```

학습을 진행해봅시다.

먼저, 저장해놓았던 데이터를 불러옵시다.


```python
preprocessed_dir = "./preprocessed_data/"

with open(preprocessed_dir + "vocab_size.p", "rb") as fp:   
    vocab_size = pickle.load(fp)
    
with open(preprocessed_dir + "input_sequences.p", "rb") as fp:   
    input_sequences = pickle.load(fp)
    
with open(preprocessed_dir + "label_sequences.p", "rb") as fp:   
    label_sequences = pickle.load(fp)

with open(preprocessed_dir + "mel_set.p", "rb") as fp:   
    mel_set = pickle.load(fp)

with open(preprocessed_dir + "mel_i_v.p", "rb") as fp:   
    mel_i_v = pickle.load(fp)
```


```python
class model_RNN(object):
    def __init__(self, 
                 sess, 
                 batch_size=16, 
                 learning_rate=0.001,
                 num_layers = 3,
                 num_vocab = 1,
                 hidden_layer_units = 64,
                 sequence_length = 8,
                 data_dir='preprocessed_data/',
                 checkpoint_dir='checkpoint/',
                 sample_dir=None):

        self.sess = sess
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.hidden_layer_units = hidden_layer_units
        self.num_vocab = num_vocab
        self.sequence_length = sequence_length
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir

        # input place holders
        self.X = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_length], name='input')
        self.Y = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_length], name='label')
        
        self.x_one_hot = tf.one_hot(self.X, self.num_vocab)
        self.y_one_hot = tf.one_hot(self.Y, self.num_vocab)

        self.optimizer, self.sequence_loss, self.curr_state = self.build_model()

        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


    def create_rnn_cell(self):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_layer_units,
                                            state_is_tuple = True)
        return cell


    def create_rnn(self):
        
        multi_cells = tf.contrib.rnn.MultiRNNCell([self.create_rnn_cell()
                                                   for _ in range(self.num_layers)],
                                                   state_is_tuple=True)
        self.multi_cells = rnn.DropoutWrapper(multi_cells, input_keep_prob=0.9, output_keep_prob=0.9)

        # prepare initial state value
        self.rnn_initial_state = self.multi_cells.zero_state(self.batch_size, tf.float32)

        rnn_outputs, out_states = tf.nn.dynamic_rnn(multi_cells, self.x_one_hot, dtype=tf.float32, initial_state=self.rnn_initial_state)
        return rnn_outputs, out_states


    def build_model(self): 
        
        rnn_output, self.out_state = self.create_rnn()
        rnn_output_flat = tf.reshape(rnn_output, [-1, self.hidden_layer_units]) # [N x sequence_length, hidden]
        
        self.logits = tf.contrib.layers.fully_connected(rnn_output_flat, self.num_vocab, None)

        # for generation 
        y_softmax = tf.nn.softmax(self.logits)         # [N x seqlen, vocab_size]
        pred = tf.argmax(y_softmax, axis=1)       # [N x seqlen]
        self.pred = tf.reshape(pred, [self.batch_size, -1]) # [N, seqlen]

        y_flat = tf.reshape(self.y_one_hot, [-1, self.num_vocab]) # [N x sequence_length, vocab_size]

        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_flat, logits=self.logits)
        sequence_loss = tf.reduce_mean(losses)

        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(sequence_loss)

        tf.summary.scalar('training loss', sequence_loss)
        self.merged_summary = tf.summary.merge_all()
        
        return opt, sequence_loss, self.out_state


    ## save current model
    def save_model(self, checkpoint_dir, step): 
        model_name = "melodyRNN.model"
        model_dir = "model"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    ## load saved model
    def load(self, checkpoint_dir):   
        print(" [*] Reading checkpoint...")

        model_dir = "model"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(ckpt_name)
            print(tf.train.latest_checkpoint(checkpoint_dir))
            self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
            return True
        else:
            return False

    def train(self, input_sequences, label_sequences, num_epochs): 

        ## initialize                         
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        counter = 0
        start_time = time.time()

        ## loading model 
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        num_all_sequences = input_sequences.shape[0]
        num_batches = int(num_all_sequences / self.batch_size)

        loss_per_epoch = []
        ## training loop
        for epoch in range(num_epochs):
            for batch_idx in range(num_batches):
                start_time = time.time()
                losses_per_epoch = []
            
                _, loss, logits, curr_state, summary_str = self.sess.run([self.optimizer, 
                                                             self.sequence_loss, 
                                                             self.logits, 
                                                             self.curr_state,
                                                             self.merged_summary], 
                              feed_dict={
                                self.X: input_sequences[batch_idx * self.batch_size:(batch_idx+1)*self.batch_size], 
                                self.Y: label_sequences[batch_idx * self.batch_size:(batch_idx+1)*self.batch_size] 
                                })

                self.writer.add_summary(summary_str, epoch)

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
                    % (epoch, batch_idx, num_batches,
                        time.time() - start_time,
                        loss))
                losses_per_epoch.append(loss)
            loss_per_epoch.append(np.mean(np.array(losses_per_epoch)))
            
            counter += 1

            # if np.mod(counter, 10) == 1:
            self.save_model(self.checkpoint_dir, counter)

                # # Get sample 
                # if np.mod(counter, 200) == 1:
                #   self.get_sample(epoch, idx, 'train')
                #   self.get_sample( epoch, idx, 'val')

                # # Saving current model
                # if np.mod(counter, 500) == 2:
                #   self.save(args.checkpoint_dir, counter)
            np.savetxt('avg_loss_txt/averaged_loss_per_epoch_' + str(epoch) + '.txt', loss_per_epoch) 


    ## generate melody from input
    def predict(self, user_input_sequence, mel_i_v):
        self.predict_opt = True
        print("User input : ", user_input_sequence.shape)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        ## prepare input sequence
        print('[1] preparing user input data') # done at prdeict.py

        ## generate corresponding melody
        print('[2] generating sequence from RNN')
        print('firstly, iterating through input')
        
        hidden_state = self.sess.run(self.multi_cells.zero_state(self.batch_size, tf.float32))
        
        for i in range(user_input_sequence.shape[0]):
            print(i)
            print(user_input_sequence[i])
            new_logits, prediction, hidden_state = self.sess.run([self.logits, self.pred, self.out_state], 
                                                feed_dict={self.X: user_input_sequence[i], self.rnn_initial_state: hidden_state})
            print(new_logits)
            print(prediction)
        print(new_logits.shape)

        print('secondly, generating')
        generated_input_seq = []

        for one_hot in new_logits:
            generated_input_seq.append(np.argmax(one_hot))
        generated_input_seq = np.expand_dims(np.array(generated_input_seq), axis=0)

        ## generate melody 
        generated_melody = []
        generated_melody_length = 0

        while(generated_melody_length < 4):

            generated_pred = self.sess.run([self.pred], 
                                            feed_dict={self.X: generated_input_seq})
            for p in generated_pred[0][0]:
                curr_curve = mel_i_v[p]
                generated_melody_length += curr_curve[1]
                if generated_melody_length > 4:
                    break
                else:
                    generated_melody.append(curr_curve)
                    generated_input_seq = generated_pred[-1]


        print(np.array(generated_melody).shape)
        
        return generated_melody
```

모델을 학습해봅시다.


```python
with tf.Session() as sess:
    model = model_RNN(sess, 
                     batch_size=16, 
                     learning_rate=0.001,
                     num_layers = 3,
                     num_vocab = vocab_size,
                     hidden_layer_units = 64,
                     sequence_length = 8,
                     data_dir='preprocessed_data/')

model.train(input_sequences, label_sequences, 2)
```

    INFO:tensorflow:Summary name training loss is illegal; using training_loss instead.
     [*] Reading checkpoint...
    melodyRNN.model-1
    checkpoint/model/melodyRNN.model-1
    INFO:tensorflow:Restoring parameters from checkpoint/model/melodyRNN.model-1
     [*] Load SUCCESS
     [*] Reading checkpoint...
    melodyRNN.model-1
    checkpoint/model/melodyRNN.model-1
    INFO:tensorflow:Restoring parameters from checkpoint/model/melodyRNN.model-1
     [*] Load SUCCESS
    Epoch: [ 0] [   0/1365] time: 0.1073, loss: 0.01227980
    Epoch: [ 0] [   1/1365] time: 0.0356, loss: 0.01183742
    Epoch: [ 0] [   2/1365] time: 0.0332, loss: 0.01166602
    Epoch: [ 0] [   3/1365] time: 0.0339, loss: 0.01166958
    Epoch: [ 0] [   4/1365] time: 0.0407, loss: 0.01322884
    Epoch: [ 0] [   5/1365] time: 0.0389, loss: 0.01297888
    Epoch: [ 0] [   6/1365] time: 0.0375, loss: 0.01371813
    Epoch: [ 0] [   7/1365] time: 0.0334, loss: 0.01417347
    Epoch: [ 0] [   8/1365] time: 0.0341, loss: 0.01202493
    Epoch: [ 0] [   9/1365] time: 0.0402, loss: 0.01165101
    Epoch: [ 0] [  10/1365] time: 0.0414, loss: 0.01255168
    Epoch: [ 0] [  11/1365] time: 0.0387, loss: 0.01443481
    Epoch: [ 0] [  12/1365] time: 0.0357, loss: 0.01440440
    Epoch: [ 0] [  13/1365] time: 0.0351, loss: 0.01347904
    Epoch: [ 0] [  14/1365] time: 0.0348, loss: 0.01134635
    Epoch: [ 0] [  15/1365] time: 0.0343, loss: 0.01106020
    Epoch: [ 0] [  16/1365] time: 0.0392, loss: 0.01455718
    Epoch: [ 0] [  17/1365] time: 0.0377, loss: 0.01784862
    Epoch: [ 0] [  18/1365] time: 0.0407, loss: 0.01138028
    Epoch: [ 0] [  19/1365] time: 0.0326, loss: 0.01164448
    Epoch: [ 0] [  20/1365] time: 0.0351, loss: 0.01186905
    Epoch: [ 0] [  21/1365] time: 0.0350, loss: 0.01205283
    Epoch: [ 0] [  22/1365] time: 0.0414, loss: 0.01629560
    Epoch: [ 0] [  23/1365] time: 0.0389, loss: 0.01943969
    Epoch: [ 0] [  24/1365] time: 0.0336, loss: 0.01219613
    Epoch: [ 0] [  25/1365] time: 0.0331, loss: 0.01119737
    Epoch: [ 0] [  26/1365] time: 0.0329, loss: 0.00991108
    Epoch: [ 0] [  27/1365] time: 0.0392, loss: 0.00996724
    Epoch: [ 0] [  28/1365] time: 0.0332, loss: 0.01124631
    Epoch: [ 0] [  29/1365] time: 0.0384, loss: 0.01084351
    Epoch: [ 0] [  30/1365] time: 0.0364, loss: 0.00964384
    Epoch: [ 0] [  31/1365] time: 0.0365, loss: 0.01213323
    Epoch: [ 0] [  32/1365] time: 0.0413, loss: 0.00993359
    Epoch: [ 0] [  33/1365] time: 0.0369, loss: 0.01077970
    Epoch: [ 0] [  34/1365] time: 0.0366, loss: 0.01393969
    Epoch: [ 0] [  35/1365] time: 0.0428, loss: 0.01632438
    Epoch: [ 0] [  36/1365] time: 0.0428, loss: 0.01191803
    Epoch: [ 0] [  37/1365] time: 0.0317, loss: 0.01215405
    Epoch: [ 0] [  38/1365] time: 0.0359, loss: 0.01149487
    Epoch: [ 0] [  39/1365] time: 0.0499, loss: 0.01101835
    Epoch: [ 0] [  40/1365] time: 0.0416, loss: 0.01465303
    Epoch: [ 0] [  41/1365] time: 0.0454, loss: 0.01670063
    Epoch: [ 0] [  42/1365] time: 0.0411, loss: 0.01704605
    Epoch: [ 0] [  43/1365] time: 0.0368, loss: 0.01633429
    Epoch: [ 0] [  44/1365] time: 0.0555, loss: 0.01681992
    Epoch: [ 0] [  45/1365] time: 0.0430, loss: 0.01457074
    Epoch: [ 0] [  46/1365] time: 0.0319, loss: 0.01739969
    Epoch: [ 0] [  47/1365] time: 0.0392, loss: 0.01721600
    Epoch: [ 0] [  48/1365] time: 0.0307, loss: 0.01676632
    Epoch: [ 0] [  49/1365] time: 0.0303, loss: 0.01734963
    Epoch: [ 0] [  50/1365] time: 0.0302, loss: 0.01366553
    Epoch: [ 0] [  51/1365] time: 0.0357, loss: 0.01257492
    Epoch: [ 0] [  52/1365] time: 0.0331, loss: 0.01344770
    Epoch: [ 0] [  53/1365] time: 0.0374, loss: 0.01547591
    Epoch: [ 0] [  54/1365] time: 0.0359, loss: 0.01405522
    Epoch: [ 0] [  55/1365] time: 0.0330, loss: 0.01067577
    Epoch: [ 0] [  56/1365] time: 0.0323, loss: 0.00991006
    Epoch: [ 0] [  57/1365] time: 0.0313, loss: 0.01227889
    Epoch: [ 0] [  58/1365] time: 0.0369, loss: 0.01271936
    Epoch: [ 0] [  59/1365] time: 0.0321, loss: 0.01053283
    Epoch: [ 0] [  60/1365] time: 0.0320, loss: 0.01130947
    Epoch: [ 0] [  61/1365] time: 0.0404, loss: 0.01234551
    Epoch: [ 0] [  62/1365] time: 0.0416, loss: 0.01204493
    Epoch: [ 0] [  63/1365] time: 0.0406, loss: 0.01234496
    Epoch: [ 0] [  64/1365] time: 0.0393, loss: 0.01242534
    Epoch: [ 0] [  65/1365] time: 0.0412, loss: 0.01192902
    Epoch: [ 0] [  66/1365] time: 0.0356, loss: 0.01181208
    Epoch: [ 0] [  67/1365] time: 0.0328, loss: 0.01301221
    Epoch: [ 0] [  68/1365] time: 0.0392, loss: 0.01381258
    Epoch: [ 0] [  69/1365] time: 0.0376, loss: 0.01217523
    Epoch: [ 0] [  70/1365] time: 0.0462, loss: 0.01144683
    Epoch: [ 0] [  71/1365] time: 0.0313, loss: 0.01225224
    Epoch: [ 0] [  72/1365] time: 0.0318, loss: 0.01253465
    Epoch: [ 0] [  73/1365] time: 0.0343, loss: 0.01223819
    Epoch: [ 0] [  74/1365] time: 0.0370, loss: 0.01267881
    Epoch: [ 0] [  75/1365] time: 0.0400, loss: 0.01245953
    Epoch: [ 0] [  76/1365] time: 0.0418, loss: 0.01238960
    Epoch: [ 0] [  77/1365] time: 0.0390, loss: 0.01315559
    Epoch: [ 0] [  78/1365] time: 0.0397, loss: 0.01123616
    Epoch: [ 0] [  79/1365] time: 0.0390, loss: 0.01200103
    Epoch: [ 0] [  80/1365] time: 0.0373, loss: 0.01046393
    Epoch: [ 0] [  81/1365] time: 0.0312, loss: 0.01028034
    Epoch: [ 0] [  82/1365] time: 0.0311, loss: 0.01098193
    Epoch: [ 0] [  83/1365] time: 0.0353, loss: 0.01154767
    Epoch: [ 0] [  84/1365] time: 0.0297, loss: 0.01271915
    Epoch: [ 0] [  85/1365] time: 0.0326, loss: 0.01183020
    Epoch: [ 0] [  86/1365] time: 0.0399, loss: 0.01167221
    Epoch: [ 0] [  87/1365] time: 0.0464, loss: 0.01225883
    Epoch: [ 0] [  88/1365] time: 0.0521, loss: 0.00981299
    Epoch: [ 0] [  89/1365] time: 0.0496, loss: 0.01248628
    Epoch: [ 0] [  90/1365] time: 0.0388, loss: 0.01200684
    Epoch: [ 0] [  91/1365] time: 0.0403, loss: 0.01645267
    Epoch: [ 0] [  92/1365] time: 0.0433, loss: 0.01304499
    Epoch: [ 0] [  93/1365] time: 0.0386, loss: 0.01799627
    Epoch: [ 0] [  94/1365] time: 0.0440, loss: 0.01015713
    Epoch: [ 0] [  95/1365] time: 0.0456, loss: 0.01022493
    Epoch: [ 0] [  96/1365] time: 0.0630, loss: 0.01303635
    Epoch: [ 0] [  97/1365] time: 0.0459, loss: 0.01231260
    Epoch: [ 0] [  98/1365] time: 0.0459, loss: 0.01360910
    Epoch: [ 0] [  99/1365] time: 0.0413, loss: 0.01379847
    Epoch: [ 0] [ 100/1365] time: 0.0371, loss: 0.01324955
    Epoch: [ 0] [ 101/1365] time: 0.0385, loss: 0.01180333
    Epoch: [ 0] [ 102/1365] time: 0.0576, loss: 0.01107692
    Epoch: [ 0] [ 103/1365] time: 0.0442, loss: 0.01243506
    Epoch: [ 0] [ 104/1365] time: 0.0410, loss: 0.01397087
    Epoch: [ 0] [ 105/1365] time: 0.0408, loss: 0.01903139
    Epoch: [ 0] [ 106/1365] time: 0.0450, loss: 0.01128499
    Epoch: [ 0] [ 107/1365] time: 0.0429, loss: 0.01183139
    Epoch: [ 0] [ 108/1365] time: 0.0409, loss: 0.00953016
    Epoch: [ 0] [ 109/1365] time: 0.0362, loss: 0.01056260
    Epoch: [ 0] [ 110/1365] time: 0.0332, loss: 0.01040010
    Epoch: [ 0] [ 111/1365] time: 0.0345, loss: 0.01007690
    Epoch: [ 0] [ 112/1365] time: 0.0321, loss: 0.00999980
    Epoch: [ 0] [ 113/1365] time: 0.0329, loss: 0.00992298
    Epoch: [ 0] [ 114/1365] time: 0.0372, loss: 0.01260582
    Epoch: [ 0] [ 115/1365] time: 0.0350, loss: 0.01002330
    Epoch: [ 0] [ 116/1365] time: 0.0309, loss: 0.01181019
    Epoch: [ 0] [ 117/1365] time: 0.0327, loss: 0.01203035
    Epoch: [ 0] [ 118/1365] time: 0.0323, loss: 0.01040241
    Epoch: [ 0] [ 119/1365] time: 0.0341, loss: 0.01043008
    Epoch: [ 0] [ 120/1365] time: 0.0335, loss: 0.01072824
    Epoch: [ 0] [ 121/1365] time: 0.0417, loss: 0.01136068
    Epoch: [ 0] [ 122/1365] time: 0.0362, loss: 0.01074808
    Epoch: [ 0] [ 123/1365] time: 0.0343, loss: 0.01036445
    Epoch: [ 0] [ 124/1365] time: 0.0364, loss: 0.01023357
    Epoch: [ 0] [ 125/1365] time: 0.0391, loss: 0.01020062
    Epoch: [ 0] [ 126/1365] time: 0.0361, loss: 0.00948595
    Epoch: [ 0] [ 127/1365] time: 0.0356, loss: 0.00990249
    Epoch: [ 0] [ 128/1365] time: 0.0422, loss: 0.00966220
    Epoch: [ 0] [ 129/1365] time: 0.0460, loss: 0.00985432
    Epoch: [ 0] [ 130/1365] time: 0.0489, loss: 0.00950366
    Epoch: [ 0] [ 131/1365] time: 0.0464, loss: 0.01086980
    Epoch: [ 0] [ 132/1365] time: 0.0314, loss: 0.01018756
    Epoch: [ 0] [ 133/1365] time: 0.0312, loss: 0.01103145
    Epoch: [ 0] [ 134/1365] time: 0.0313, loss: 0.01111246
    Epoch: [ 0] [ 135/1365] time: 0.0327, loss: 0.01057042
    Epoch: [ 0] [ 136/1365] time: 0.0337, loss: 0.01040635
    Epoch: [ 0] [ 137/1365] time: 0.0379, loss: 0.00974708
    Epoch: [ 0] [ 138/1365] time: 0.0338, loss: 0.01062156
    Epoch: [ 0] [ 139/1365] time: 0.0414, loss: 0.01099224
    Epoch: [ 0] [ 140/1365] time: 0.0354, loss: 0.01064457
    Epoch: [ 0] [ 141/1365] time: 0.0384, loss: 0.01081020
    Epoch: [ 0] [ 142/1365] time: 0.0382, loss: 0.01063744
    Epoch: [ 0] [ 143/1365] time: 0.0372, loss: 0.01021691
    Epoch: [ 0] [ 144/1365] time: 0.0350, loss: 0.01032595
    Epoch: [ 0] [ 145/1365] time: 0.0381, loss: 0.00998410
    Epoch: [ 0] [ 146/1365] time: 0.0381, loss: 0.00998766
    Epoch: [ 0] [ 147/1365] time: 0.0375, loss: 0.00978755
    Epoch: [ 0] [ 148/1365] time: 0.0403, loss: 0.00958450
    Epoch: [ 0] [ 149/1365] time: 0.0425, loss: 0.00953718
    Epoch: [ 0] [ 150/1365] time: 0.0434, loss: 0.00939583
    Epoch: [ 0] [ 151/1365] time: 0.0363, loss: 0.00873898
    Epoch: [ 0] [ 152/1365] time: 0.0366, loss: 0.00912441
    Epoch: [ 0] [ 153/1365] time: 0.0346, loss: 0.00923539
    Epoch: [ 0] [ 154/1365] time: 0.0326, loss: 0.00902938
    Epoch: [ 0] [ 155/1365] time: 0.0341, loss: 0.00893830
    Epoch: [ 0] [ 156/1365] time: 0.0355, loss: 0.00818168
    Epoch: [ 0] [ 157/1365] time: 0.0349, loss: 0.00884095
    Epoch: [ 0] [ 158/1365] time: 0.0295, loss: 0.00934683
    Epoch: [ 0] [ 159/1365] time: 0.0293, loss: 0.00956121
    Epoch: [ 0] [ 160/1365] time: 0.0293, loss: 0.00928412
    Epoch: [ 0] [ 161/1365] time: 0.0320, loss: 0.01004663
    Epoch: [ 0] [ 162/1365] time: 0.0342, loss: 0.00908643
    Epoch: [ 0] [ 163/1365] time: 0.0310, loss: 0.01091307
    Epoch: [ 0] [ 164/1365] time: 0.0304, loss: 0.01067484
    Epoch: [ 0] [ 165/1365] time: 0.0329, loss: 0.01059570
    Epoch: [ 0] [ 166/1365] time: 0.0334, loss: 0.01141817
    Epoch: [ 0] [ 167/1365] time: 0.0327, loss: 0.01132046
    Epoch: [ 0] [ 168/1365] time: 0.0337, loss: 0.01076432
    Epoch: [ 0] [ 169/1365] time: 0.0346, loss: 0.01239941
    Epoch: [ 0] [ 170/1365] time: 0.0315, loss: 0.01111845
    Epoch: [ 0] [ 171/1365] time: 0.0316, loss: 0.01081454
    Epoch: [ 0] [ 172/1365] time: 0.0301, loss: 0.01179717
    Epoch: [ 0] [ 173/1365] time: 0.0293, loss: 0.01157175
    Epoch: [ 0] [ 174/1365] time: 0.0310, loss: 0.01019969
    Epoch: [ 0] [ 175/1365] time: 0.0318, loss: 0.00986080
    Epoch: [ 0] [ 176/1365] time: 0.0330, loss: 0.00955504
    Epoch: [ 0] [ 177/1365] time: 0.0295, loss: 0.01026786
    Epoch: [ 0] [ 178/1365] time: 0.0296, loss: 0.01058690
    Epoch: [ 0] [ 179/1365] time: 0.0288, loss: 0.01033658
    Epoch: [ 0] [ 180/1365] time: 0.0294, loss: 0.01016038
    Epoch: [ 0] [ 181/1365] time: 0.0293, loss: 0.01038371
    Epoch: [ 0] [ 182/1365] time: 0.0392, loss: 0.00996564
    Epoch: [ 0] [ 183/1365] time: 0.0348, loss: 0.01045170
    Epoch: [ 0] [ 184/1365] time: 0.0304, loss: 0.01067292
    Epoch: [ 0] [ 185/1365] time: 0.0297, loss: 0.01099296
    Epoch: [ 0] [ 186/1365] time: 0.0310, loss: 0.01061209
    Epoch: [ 0] [ 187/1365] time: 0.0296, loss: 0.01063124
    Epoch: [ 0] [ 188/1365] time: 0.0293, loss: 0.01026091
    Epoch: [ 0] [ 189/1365] time: 0.0312, loss: 0.01022283
    Epoch: [ 0] [ 190/1365] time: 0.0325, loss: 0.01070239
    Epoch: [ 0] [ 191/1365] time: 0.0296, loss: 0.00987651
    Epoch: [ 0] [ 192/1365] time: 0.0297, loss: 0.00942471
    Epoch: [ 0] [ 193/1365] time: 0.0303, loss: 0.01056517
    Epoch: [ 0] [ 194/1365] time: 0.0338, loss: 0.00990420
    Epoch: [ 0] [ 195/1365] time: 0.0374, loss: 0.01013235
    Epoch: [ 0] [ 196/1365] time: 0.0425, loss: 0.01009125
    Epoch: [ 0] [ 197/1365] time: 0.0405, loss: 0.01007984
    Epoch: [ 0] [ 198/1365] time: 0.0417, loss: 0.01063485
    Epoch: [ 0] [ 199/1365] time: 0.0361, loss: 0.01020288
    Epoch: [ 0] [ 200/1365] time: 0.0362, loss: 0.00830832
    Epoch: [ 0] [ 201/1365] time: 0.0354, loss: 0.00864904
    Epoch: [ 0] [ 202/1365] time: 0.0398, loss: 0.00823885
    Epoch: [ 0] [ 203/1365] time: 0.0386, loss: 0.00864236
    Epoch: [ 0] [ 204/1365] time: 0.0374, loss: 0.01034521
    Epoch: [ 0] [ 205/1365] time: 0.0354, loss: 0.01030337
    Epoch: [ 0] [ 206/1365] time: 0.0347, loss: 0.00975111
    Epoch: [ 0] [ 207/1365] time: 0.0327, loss: 0.00986703
    Epoch: [ 0] [ 208/1365] time: 0.0345, loss: 0.01004434
    Epoch: [ 0] [ 209/1365] time: 0.0404, loss: 0.01077599
    Epoch: [ 0] [ 210/1365] time: 0.0359, loss: 0.01029723
    Epoch: [ 0] [ 211/1365] time: 0.0388, loss: 0.00904616
    Epoch: [ 0] [ 212/1365] time: 0.0337, loss: 0.00977736
    Epoch: [ 0] [ 213/1365] time: 0.0316, loss: 0.01057123
    Epoch: [ 0] [ 214/1365] time: 0.0348, loss: 0.00992870
    Epoch: [ 0] [ 215/1365] time: 0.0408, loss: 0.00916338
    Epoch: [ 0] [ 216/1365] time: 0.0373, loss: 0.00970244
    Epoch: [ 0] [ 217/1365] time: 0.0352, loss: 0.00989487
    Epoch: [ 0] [ 218/1365] time: 0.0313, loss: 0.01166918
    Epoch: [ 0] [ 219/1365] time: 0.0321, loss: 0.01034407
    Epoch: [ 0] [ 220/1365] time: 0.0339, loss: 0.01004780
    Epoch: [ 0] [ 221/1365] time: 0.0368, loss: 0.01035021
    Epoch: [ 0] [ 222/1365] time: 0.0324, loss: 0.01024655
    Epoch: [ 0] [ 223/1365] time: 0.0303, loss: 0.00926393
    Epoch: [ 0] [ 224/1365] time: 0.0328, loss: 0.01094127
    Epoch: [ 0] [ 225/1365] time: 0.0411, loss: 0.00963236
    Epoch: [ 0] [ 226/1365] time: 0.0366, loss: 0.00915337
    Epoch: [ 0] [ 227/1365] time: 0.0421, loss: 0.01044550
    Epoch: [ 0] [ 228/1365] time: 0.0375, loss: 0.01088389
    Epoch: [ 0] [ 229/1365] time: 0.0323, loss: 0.01155888
    Epoch: [ 0] [ 230/1365] time: 0.0318, loss: 0.01015209
    Epoch: [ 0] [ 231/1365] time: 0.0307, loss: 0.00916140
    Epoch: [ 0] [ 232/1365] time: 0.0323, loss: 0.01011287
    Epoch: [ 0] [ 233/1365] time: 0.0359, loss: 0.01047875
    Epoch: [ 0] [ 234/1365] time: 0.0325, loss: 0.00960989
    Epoch: [ 0] [ 235/1365] time: 0.0334, loss: 0.01163329
    Epoch: [ 0] [ 236/1365] time: 0.0345, loss: 0.01266979
    Epoch: [ 0] [ 237/1365] time: 0.0314, loss: 0.00863803
    Epoch: [ 0] [ 238/1365] time: 0.0315, loss: 0.00850399
    Epoch: [ 0] [ 239/1365] time: 0.0409, loss: 0.00972935
    Epoch: [ 0] [ 240/1365] time: 0.0370, loss: 0.01017102
    Epoch: [ 0] [ 241/1365] time: 0.0426, loss: 0.00939778
    Epoch: [ 0] [ 242/1365] time: 0.0378, loss: 0.00974188
    Epoch: [ 0] [ 243/1365] time: 0.0355, loss: 0.01148683
    Epoch: [ 0] [ 244/1365] time: 0.0403, loss: 0.00961835
    Epoch: [ 0] [ 245/1365] time: 0.0363, loss: 0.00896017
    Epoch: [ 0] [ 246/1365] time: 0.0316, loss: 0.01034075
    Epoch: [ 0] [ 247/1365] time: 0.0312, loss: 0.01114745
    Epoch: [ 0] [ 248/1365] time: 0.0311, loss: 0.01253228
    Epoch: [ 0] [ 249/1365] time: 0.0308, loss: 0.01199395
    Epoch: [ 0] [ 250/1365] time: 0.0316, loss: 0.00915614
    Epoch: [ 0] [ 251/1365] time: 0.0363, loss: 0.00873350
    Epoch: [ 0] [ 252/1365] time: 0.0381, loss: 0.01012966
    Epoch: [ 0] [ 253/1365] time: 0.0374, loss: 0.00924951
    Epoch: [ 0] [ 254/1365] time: 0.0318, loss: 0.01024568
    Epoch: [ 0] [ 255/1365] time: 0.0306, loss: 0.00999185
    Epoch: [ 0] [ 256/1365] time: 0.0314, loss: 0.01308408
    Epoch: [ 0] [ 257/1365] time: 0.0327, loss: 0.01073188
    Epoch: [ 0] [ 258/1365] time: 0.0354, loss: 0.00950861
    Epoch: [ 0] [ 259/1365] time: 0.0340, loss: 0.00980617
    Epoch: [ 0] [ 260/1365] time: 0.0315, loss: 0.00981219
    Epoch: [ 0] [ 261/1365] time: 0.0305, loss: 0.00916557
    Epoch: [ 0] [ 262/1365] time: 0.0292, loss: 0.00942751
    Epoch: [ 0] [ 263/1365] time: 0.0323, loss: 0.00883127
    Epoch: [ 0] [ 264/1365] time: 0.0312, loss: 0.00966691
    Epoch: [ 0] [ 265/1365] time: 0.0337, loss: 0.01039517
    Epoch: [ 0] [ 266/1365] time: 0.0309, loss: 0.01138915
    Epoch: [ 0] [ 267/1365] time: 0.0298, loss: 0.01061506
    Epoch: [ 0] [ 268/1365] time: 0.0305, loss: 0.00980624
    Epoch: [ 0] [ 269/1365] time: 0.0298, loss: 0.00967449
    Epoch: [ 0] [ 270/1365] time: 0.0353, loss: 0.01052630
    Epoch: [ 0] [ 271/1365] time: 0.0326, loss: 0.01047136
    Epoch: [ 0] [ 272/1365] time: 0.0310, loss: 0.01021014
    Epoch: [ 0] [ 273/1365] time: 0.0320, loss: 0.01037945
    Epoch: [ 0] [ 274/1365] time: 0.0288, loss: 0.00983490
    Epoch: [ 0] [ 275/1365] time: 0.0272, loss: 0.01068631
    Epoch: [ 0] [ 276/1365] time: 0.0295, loss: 0.00996939
    Epoch: [ 0] [ 277/1365] time: 0.0300, loss: 0.00993229
    Epoch: [ 0] [ 278/1365] time: 0.0296, loss: 0.00969514
    Epoch: [ 0] [ 279/1365] time: 0.0306, loss: 0.00959571
    Epoch: [ 0] [ 280/1365] time: 0.0305, loss: 0.01029683
    Epoch: [ 0] [ 281/1365] time: 0.0297, loss: 0.01006464
    Epoch: [ 0] [ 282/1365] time: 0.0279, loss: 0.00929754
    Epoch: [ 0] [ 283/1365] time: 0.0283, loss: 0.00964848
    Epoch: [ 0] [ 284/1365] time: 0.0294, loss: 0.00981589
    Epoch: [ 0] [ 285/1365] time: 0.0298, loss: 0.00971170
    Epoch: [ 0] [ 286/1365] time: 0.0332, loss: 0.00980132
    Epoch: [ 0] [ 287/1365] time: 0.0304, loss: 0.01005958
    Epoch: [ 0] [ 288/1365] time: 0.0318, loss: 0.01130047
    Epoch: [ 0] [ 289/1365] time: 0.0302, loss: 0.00993387
    Epoch: [ 0] [ 290/1365] time: 0.0295, loss: 0.00956535
    Epoch: [ 0] [ 291/1365] time: 0.0290, loss: 0.01042691
    Epoch: [ 0] [ 292/1365] time: 0.0296, loss: 0.00929342
    Epoch: [ 0] [ 293/1365] time: 0.0299, loss: 0.00927101
    Epoch: [ 0] [ 294/1365] time: 0.0301, loss: 0.00847943
    Epoch: [ 0] [ 295/1365] time: 0.0321, loss: 0.00942436
    Epoch: [ 0] [ 296/1365] time: 0.0315, loss: 0.01010689
    Epoch: [ 0] [ 297/1365] time: 0.0293, loss: 0.01083804
    Epoch: [ 0] [ 298/1365] time: 0.0292, loss: 0.00935849
    Epoch: [ 0] [ 299/1365] time: 0.0306, loss: 0.01076567
    Epoch: [ 0] [ 300/1365] time: 0.0311, loss: 0.01043693
    Epoch: [ 0] [ 301/1365] time: 0.0353, loss: 0.01000731
    Epoch: [ 0] [ 302/1365] time: 0.0349, loss: 0.01068650
    Epoch: [ 0] [ 303/1365] time: 0.0299, loss: 0.01147724
    Epoch: [ 0] [ 304/1365] time: 0.0297, loss: 0.00994270
    Epoch: [ 0] [ 305/1365] time: 0.0298, loss: 0.00992636
    Epoch: [ 0] [ 306/1365] time: 0.0291, loss: 0.01048006
    Epoch: [ 0] [ 307/1365] time: 0.0325, loss: 0.00959411
    Epoch: [ 0] [ 308/1365] time: 0.0331, loss: 0.00954598
    Epoch: [ 0] [ 309/1365] time: 0.0294, loss: 0.00892119
    Epoch: [ 0] [ 310/1365] time: 0.0293, loss: 0.00904182
    Epoch: [ 0] [ 311/1365] time: 0.0289, loss: 0.01089106
    Epoch: [ 0] [ 312/1365] time: 0.0516, loss: 0.01100600
    Epoch: [ 0] [ 313/1365] time: 0.0385, loss: 0.01073057
    Epoch: [ 0] [ 314/1365] time: 0.0459, loss: 0.01360644
    Epoch: [ 0] [ 315/1365] time: 0.0456, loss: 0.01245354
    Epoch: [ 0] [ 316/1365] time: 0.0497, loss: 0.01236690
    Epoch: [ 0] [ 317/1365] time: 0.0415, loss: 0.01286544
    Epoch: [ 0] [ 318/1365] time: 0.0405, loss: 0.00890683
    Epoch: [ 0] [ 319/1365] time: 0.0429, loss: 0.00989958
    Epoch: [ 0] [ 320/1365] time: 0.0450, loss: 0.01023311
    Epoch: [ 0] [ 321/1365] time: 0.0457, loss: 0.00994848
    Epoch: [ 0] [ 322/1365] time: 0.0423, loss: 0.01091717
    Epoch: [ 0] [ 323/1365] time: 0.0375, loss: 0.01507465
    Epoch: [ 0] [ 324/1365] time: 0.0452, loss: 0.01533649
    Epoch: [ 0] [ 325/1365] time: 0.0433, loss: 0.01465926
    Epoch: [ 0] [ 326/1365] time: 0.0380, loss: 0.00951732
    Epoch: [ 0] [ 327/1365] time: 0.0318, loss: 0.00928097
    Epoch: [ 0] [ 328/1365] time: 0.0318, loss: 0.00959854
    Epoch: [ 0] [ 329/1365] time: 0.0388, loss: 0.01049309
    Epoch: [ 0] [ 330/1365] time: 0.0354, loss: 0.00918502
    Epoch: [ 0] [ 331/1365] time: 0.0308, loss: 0.00947886
    Epoch: [ 0] [ 332/1365] time: 0.0311, loss: 0.00990053
    Epoch: [ 0] [ 333/1365] time: 0.0306, loss: 0.00945846
    Epoch: [ 0] [ 334/1365] time: 0.0311, loss: 0.01034040
    Epoch: [ 0] [ 335/1365] time: 0.0320, loss: 0.00974940
    Epoch: [ 0] [ 336/1365] time: 0.0434, loss: 0.00910312
    Epoch: [ 0] [ 337/1365] time: 0.0373, loss: 0.00911972
    Epoch: [ 0] [ 338/1365] time: 0.0311, loss: 0.01040651
    Epoch: [ 0] [ 339/1365] time: 0.0331, loss: 0.01009060
    Epoch: [ 0] [ 340/1365] time: 0.0307, loss: 0.00970776
    Epoch: [ 0] [ 341/1365] time: 0.0316, loss: 0.00943709
    Epoch: [ 0] [ 342/1365] time: 0.0346, loss: 0.00952998
    Epoch: [ 0] [ 343/1365] time: 0.0366, loss: 0.00984751
    Epoch: [ 0] [ 344/1365] time: 0.0332, loss: 0.01002996
    Epoch: [ 0] [ 345/1365] time: 0.0341, loss: 0.00979317
    Epoch: [ 0] [ 346/1365] time: 0.0351, loss: 0.01062486
    Epoch: [ 0] [ 347/1365] time: 0.0347, loss: 0.00968764
    Epoch: [ 0] [ 348/1365] time: 0.0364, loss: 0.00915593
    Epoch: [ 0] [ 349/1365] time: 0.0406, loss: 0.01101461
    Epoch: [ 0] [ 350/1365] time: 0.0358, loss: 0.01064210
    Epoch: [ 0] [ 351/1365] time: 0.0424, loss: 0.01153558
    Epoch: [ 0] [ 352/1365] time: 0.0358, loss: 0.01012346
    Epoch: [ 0] [ 353/1365] time: 0.0367, loss: 0.00984233
    Epoch: [ 0] [ 354/1365] time: 0.0410, loss: 0.00950143
    Epoch: [ 0] [ 355/1365] time: 0.0344, loss: 0.00988764
    Epoch: [ 0] [ 356/1365] time: 0.0352, loss: 0.00971399
    Epoch: [ 0] [ 357/1365] time: 0.0342, loss: 0.01049067
    Epoch: [ 0] [ 358/1365] time: 0.0352, loss: 0.01053032
    Epoch: [ 0] [ 359/1365] time: 0.0354, loss: 0.01013568
    Epoch: [ 0] [ 360/1365] time: 0.0364, loss: 0.00977141
    Epoch: [ 0] [ 361/1365] time: 0.0408, loss: 0.01028755
    Epoch: [ 0] [ 362/1365] time: 0.0351, loss: 0.00810358
    Epoch: [ 0] [ 363/1365] time: 0.0350, loss: 0.00856666
    Epoch: [ 0] [ 364/1365] time: 0.0356, loss: 0.00934793
    Epoch: [ 0] [ 365/1365] time: 0.0403, loss: 0.01099529
    Epoch: [ 0] [ 366/1365] time: 0.0434, loss: 0.01069771
    Epoch: [ 0] [ 367/1365] time: 0.0391, loss: 0.00841834
    Epoch: [ 0] [ 368/1365] time: 0.0346, loss: 0.00952225
    Epoch: [ 0] [ 369/1365] time: 0.0321, loss: 0.00972152
    Epoch: [ 0] [ 370/1365] time: 0.0320, loss: 0.00896193
    Epoch: [ 0] [ 371/1365] time: 0.0306, loss: 0.00942709
    Epoch: [ 0] [ 372/1365] time: 0.0309, loss: 0.01183057
    Epoch: [ 0] [ 373/1365] time: 0.0337, loss: 0.01192992
    Epoch: [ 0] [ 374/1365] time: 0.0307, loss: 0.00873922
    Epoch: [ 0] [ 375/1365] time: 0.0286, loss: 0.00851067
    Epoch: [ 0] [ 376/1365] time: 0.0295, loss: 0.00909693
    Epoch: [ 0] [ 377/1365] time: 0.0293, loss: 0.01002956
    Epoch: [ 0] [ 378/1365] time: 0.0294, loss: 0.01027025
    Epoch: [ 0] [ 379/1365] time: 0.0307, loss: 0.00906956
    Epoch: [ 0] [ 380/1365] time: 0.0336, loss: 0.01074834
    Epoch: [ 0] [ 381/1365] time: 0.0352, loss: 0.00898812
    Epoch: [ 0] [ 382/1365] time: 0.0342, loss: 0.01026097
    Epoch: [ 0] [ 383/1365] time: 0.0296, loss: 0.00928625
    Epoch: [ 0] [ 384/1365] time: 0.0294, loss: 0.00950848
    Epoch: [ 0] [ 385/1365] time: 0.0355, loss: 0.00954468
    Epoch: [ 0] [ 386/1365] time: 0.0324, loss: 0.00938522
    Epoch: [ 0] [ 387/1365] time: 0.0309, loss: 0.01020356
    Epoch: [ 0] [ 388/1365] time: 0.0293, loss: 0.00985497
    Epoch: [ 0] [ 389/1365] time: 0.0294, loss: 0.00868859
    Epoch: [ 0] [ 390/1365] time: 0.0291, loss: 0.00903940
    Epoch: [ 0] [ 391/1365] time: 0.0288, loss: 0.00949053
    Epoch: [ 0] [ 392/1365] time: 0.0300, loss: 0.00922843
    Epoch: [ 0] [ 393/1365] time: 0.0310, loss: 0.00976094
    Epoch: [ 0] [ 394/1365] time: 0.0332, loss: 0.01005141
    Epoch: [ 0] [ 395/1365] time: 0.0301, loss: 0.00988741
    Epoch: [ 0] [ 396/1365] time: 0.0297, loss: 0.00909795
    Epoch: [ 0] [ 397/1365] time: 0.0294, loss: 0.00933481
    Epoch: [ 0] [ 398/1365] time: 0.0318, loss: 0.00958583
    Epoch: [ 0] [ 399/1365] time: 0.0344, loss: 0.00940477
    Epoch: [ 0] [ 400/1365] time: 0.0326, loss: 0.00888727
    Epoch: [ 0] [ 401/1365] time: 0.0333, loss: 0.00963655
    Epoch: [ 0] [ 402/1365] time: 0.0296, loss: 0.01017172
    Epoch: [ 0] [ 403/1365] time: 0.0299, loss: 0.01026260
    Epoch: [ 0] [ 404/1365] time: 0.0297, loss: 0.01023230
    Epoch: [ 0] [ 405/1365] time: 0.0291, loss: 0.00914120
    Epoch: [ 0] [ 406/1365] time: 0.0293, loss: 0.00922259
    Epoch: [ 0] [ 407/1365] time: 0.0298, loss: 0.00892451
    Epoch: [ 0] [ 408/1365] time: 0.0332, loss: 0.00940096
    Epoch: [ 0] [ 409/1365] time: 0.0331, loss: 0.00917827
    Epoch: [ 0] [ 410/1365] time: 0.0294, loss: 0.01034889
    Epoch: [ 0] [ 411/1365] time: 0.0292, loss: 0.01004699
    Epoch: [ 0] [ 412/1365] time: 0.0285, loss: 0.01011248
    Epoch: [ 0] [ 413/1365] time: 0.0293, loss: 0.01109730
    Epoch: [ 0] [ 414/1365] time: 0.0323, loss: 0.00972868
    Epoch: [ 0] [ 415/1365] time: 0.0339, loss: 0.01108715
    Epoch: [ 0] [ 416/1365] time: 0.0357, loss: 0.01106827
    Epoch: [ 0] [ 417/1365] time: 0.0300, loss: 0.01032192
    Epoch: [ 0] [ 418/1365] time: 0.0299, loss: 0.00925999
    Epoch: [ 0] [ 419/1365] time: 0.0297, loss: 0.00960037
    Epoch: [ 0] [ 420/1365] time: 0.0295, loss: 0.00958246
    Epoch: [ 0] [ 421/1365] time: 0.0318, loss: 0.00858472
    Epoch: [ 0] [ 422/1365] time: 0.0330, loss: 0.01026750
    Epoch: [ 0] [ 423/1365] time: 0.0299, loss: 0.01103316
    Epoch: [ 0] [ 424/1365] time: 0.0294, loss: 0.01029230
    Epoch: [ 0] [ 425/1365] time: 0.0295, loss: 0.00982755
    Epoch: [ 0] [ 426/1365] time: 0.0292, loss: 0.01012887
    Epoch: [ 0] [ 427/1365] time: 0.0293, loss: 0.00941766
    Epoch: [ 0] [ 428/1365] time: 0.0309, loss: 0.01032906
    Epoch: [ 0] [ 429/1365] time: 0.0402, loss: 0.01141146
    Epoch: [ 0] [ 430/1365] time: 0.0394, loss: 0.01014161
    Epoch: [ 0] [ 431/1365] time: 0.0462, loss: 0.01077286
    Epoch: [ 0] [ 432/1365] time: 0.0477, loss: 0.01024693
    Epoch: [ 0] [ 433/1365] time: 0.0476, loss: 0.00910581
    Epoch: [ 0] [ 434/1365] time: 0.0437, loss: 0.00907120
    Epoch: [ 0] [ 435/1365] time: 0.0374, loss: 0.00873174
    Epoch: [ 0] [ 436/1365] time: 0.0415, loss: 0.00884196
    Epoch: [ 0] [ 437/1365] time: 0.0329, loss: 0.00928992
    Epoch: [ 0] [ 438/1365] time: 0.0347, loss: 0.01010394
    Epoch: [ 0] [ 439/1365] time: 0.0365, loss: 0.01011232
    Epoch: [ 0] [ 440/1365] time: 0.0342, loss: 0.00963041
    Epoch: [ 0] [ 441/1365] time: 0.0352, loss: 0.01018032
    Epoch: [ 0] [ 442/1365] time: 0.0372, loss: 0.01157726
    Epoch: [ 0] [ 443/1365] time: 0.0343, loss: 0.01280434
    Epoch: [ 0] [ 444/1365] time: 0.0329, loss: 0.00956515
    Epoch: [ 0] [ 445/1365] time: 0.0351, loss: 0.00988004
    Epoch: [ 0] [ 446/1365] time: 0.0399, loss: 0.00957860
    Epoch: [ 0] [ 447/1365] time: 0.0417, loss: 0.01142395
    Epoch: [ 0] [ 448/1365] time: 0.0410, loss: 0.01023492
    Epoch: [ 0] [ 449/1365] time: 0.0347, loss: 0.00993866
    Epoch: [ 0] [ 450/1365] time: 0.0340, loss: 0.00932716
    Epoch: [ 0] [ 451/1365] time: 0.0411, loss: 0.00927429
    Epoch: [ 0] [ 452/1365] time: 0.0332, loss: 0.00945676
    Epoch: [ 0] [ 453/1365] time: 0.0357, loss: 0.01049649
    Epoch: [ 0] [ 454/1365] time: 0.0381, loss: 0.01087590
    Epoch: [ 0] [ 455/1365] time: 0.0392, loss: 0.01033416
    Epoch: [ 0] [ 456/1365] time: 0.0352, loss: 0.01040407
    Epoch: [ 0] [ 457/1365] time: 0.0356, loss: 0.00987595
    Epoch: [ 0] [ 458/1365] time: 0.0430, loss: 0.01062663
    Epoch: [ 0] [ 459/1365] time: 0.0446, loss: 0.01143398
    Epoch: [ 0] [ 460/1365] time: 0.0331, loss: 0.00951248
    Epoch: [ 0] [ 461/1365] time: 0.0433, loss: 0.00918373
    Epoch: [ 0] [ 462/1365] time: 0.0352, loss: 0.01184540
    Epoch: [ 0] [ 463/1365] time: 0.0377, loss: 0.01516632
    Epoch: [ 0] [ 464/1365] time: 0.0374, loss: 0.01124211
    Epoch: [ 0] [ 465/1365] time: 0.0324, loss: 0.00941794
    Epoch: [ 0] [ 466/1365] time: 0.0281, loss: 0.01032684
    Epoch: [ 0] [ 467/1365] time: 0.0337, loss: 0.01108448
    Epoch: [ 0] [ 468/1365] time: 0.0390, loss: 0.01065180
    Epoch: [ 0] [ 469/1365] time: 0.0479, loss: 0.01104162
    Epoch: [ 0] [ 470/1365] time: 0.0368, loss: 0.01008287
    Epoch: [ 0] [ 471/1365] time: 0.0393, loss: 0.00940800
    Epoch: [ 0] [ 472/1365] time: 0.0385, loss: 0.01001647
    Epoch: [ 0] [ 473/1365] time: 0.0402, loss: 0.01018233
    Epoch: [ 0] [ 474/1365] time: 0.0365, loss: 0.01121096
    Epoch: [ 0] [ 475/1365] time: 0.0447, loss: 0.01088513
    Epoch: [ 0] [ 476/1365] time: 0.0353, loss: 0.01148430
    Epoch: [ 0] [ 477/1365] time: 0.0385, loss: 0.00974923
    Epoch: [ 0] [ 478/1365] time: 0.0407, loss: 0.01018998
    Epoch: [ 0] [ 479/1365] time: 0.0350, loss: 0.01076076
    Epoch: [ 0] [ 480/1365] time: 0.0424, loss: 0.00982528
    Epoch: [ 0] [ 481/1365] time: 0.0358, loss: 0.01127912
    Epoch: [ 0] [ 482/1365] time: 0.0337, loss: 0.01060789
    Epoch: [ 0] [ 483/1365] time: 0.0383, loss: 0.00997499
    Epoch: [ 0] [ 484/1365] time: 0.0474, loss: 0.01157521
    Epoch: [ 0] [ 485/1365] time: 0.0428, loss: 0.01021204
    Epoch: [ 0] [ 486/1365] time: 0.0383, loss: 0.00982224
    Epoch: [ 0] [ 487/1365] time: 0.0357, loss: 0.01090489
    Epoch: [ 0] [ 488/1365] time: 0.0381, loss: 0.01059588
    Epoch: [ 0] [ 489/1365] time: 0.0338, loss: 0.00918811
    Epoch: [ 0] [ 490/1365] time: 0.0339, loss: 0.00987569
    Epoch: [ 0] [ 491/1365] time: 0.0339, loss: 0.00964529
    Epoch: [ 0] [ 492/1365] time: 0.0379, loss: 0.01011314
    Epoch: [ 0] [ 493/1365] time: 0.0343, loss: 0.00985106
    Epoch: [ 0] [ 494/1365] time: 0.0376, loss: 0.00934548
    Epoch: [ 0] [ 495/1365] time: 0.0363, loss: 0.00969371
    Epoch: [ 0] [ 496/1365] time: 0.0352, loss: 0.01050636
    Epoch: [ 0] [ 497/1365] time: 0.0352, loss: 0.01076497
    Epoch: [ 0] [ 498/1365] time: 0.0405, loss: 0.01194747
    Epoch: [ 0] [ 499/1365] time: 0.0388, loss: 0.00986711
    Epoch: [ 0] [ 500/1365] time: 0.0412, loss: 0.00965087
    Epoch: [ 0] [ 501/1365] time: 0.0418, loss: 0.00984067
    Epoch: [ 0] [ 502/1365] time: 0.0409, loss: 0.01066902
    Epoch: [ 0] [ 503/1365] time: 0.0338, loss: 0.00987010
    Epoch: [ 0] [ 504/1365] time: 0.0406, loss: 0.00952418
    Epoch: [ 0] [ 505/1365] time: 0.0381, loss: 0.00969503
    Epoch: [ 0] [ 506/1365] time: 0.0346, loss: 0.00919373
    Epoch: [ 0] [ 507/1365] time: 0.0303, loss: 0.00905428
    Epoch: [ 0] [ 508/1365] time: 0.0293, loss: 0.00854212
    Epoch: [ 0] [ 509/1365] time: 0.0293, loss: 0.00983182
    Epoch: [ 0] [ 510/1365] time: 0.0289, loss: 0.01041117
    Epoch: [ 0] [ 511/1365] time: 0.0342, loss: 0.00954108
    Epoch: [ 0] [ 512/1365] time: 0.0389, loss: 0.00973317
    Epoch: [ 0] [ 513/1365] time: 0.0402, loss: 0.01033776
    Epoch: [ 0] [ 514/1365] time: 0.0381, loss: 0.01024364
    Epoch: [ 0] [ 515/1365] time: 0.0410, loss: 0.00928189
    Epoch: [ 0] [ 516/1365] time: 0.0354, loss: 0.00956088
    Epoch: [ 0] [ 517/1365] time: 0.0388, loss: 0.00920723
    Epoch: [ 0] [ 518/1365] time: 0.0350, loss: 0.00922489
    Epoch: [ 0] [ 519/1365] time: 0.0372, loss: 0.00993558
    Epoch: [ 0] [ 520/1365] time: 0.0327, loss: 0.00932370
    Epoch: [ 0] [ 521/1365] time: 0.0370, loss: 0.00890581
    Epoch: [ 0] [ 522/1365] time: 0.0354, loss: 0.00991245
    Epoch: [ 0] [ 523/1365] time: 0.0307, loss: 0.00889857
    Epoch: [ 0] [ 524/1365] time: 0.0339, loss: 0.00860180
    Epoch: [ 0] [ 525/1365] time: 0.0353, loss: 0.00874820
    Epoch: [ 0] [ 526/1365] time: 0.0311, loss: 0.00889194
    Epoch: [ 0] [ 527/1365] time: 0.0311, loss: 0.00927550
    Epoch: [ 0] [ 528/1365] time: 0.0355, loss: 0.01040071
    Epoch: [ 0] [ 529/1365] time: 0.0311, loss: 0.01061619
    Epoch: [ 0] [ 530/1365] time: 0.0297, loss: 0.01027791
    Epoch: [ 0] [ 531/1365] time: 0.0316, loss: 0.01044411
    Epoch: [ 0] [ 532/1365] time: 0.0310, loss: 0.01012474
    Epoch: [ 0] [ 533/1365] time: 0.0296, loss: 0.01129296
    Epoch: [ 0] [ 534/1365] time: 0.0290, loss: 0.01122503
    Epoch: [ 0] [ 535/1365] time: 0.0290, loss: 0.00978359
    Epoch: [ 0] [ 536/1365] time: 0.0300, loss: 0.01250545
    Epoch: [ 0] [ 537/1365] time: 0.0363, loss: 0.01055357
    Epoch: [ 0] [ 538/1365] time: 0.0360, loss: 0.00840649
    Epoch: [ 0] [ 539/1365] time: 0.0379, loss: 0.00988527
    Epoch: [ 0] [ 540/1365] time: 0.0400, loss: 0.01033473
    Epoch: [ 0] [ 541/1365] time: 0.0348, loss: 0.01064920
    Epoch: [ 0] [ 542/1365] time: 0.0375, loss: 0.01011846
    Epoch: [ 0] [ 543/1365] time: 0.0418, loss: 0.00915680
    Epoch: [ 0] [ 544/1365] time: 0.0416, loss: 0.00923187
    Epoch: [ 0] [ 545/1365] time: 0.0384, loss: 0.00927498
    Epoch: [ 0] [ 546/1365] time: 0.0350, loss: 0.00996925
    Epoch: [ 0] [ 547/1365] time: 0.0343, loss: 0.00908066
    Epoch: [ 0] [ 548/1365] time: 0.0348, loss: 0.01034830
    Epoch: [ 0] [ 549/1365] time: 0.0343, loss: 0.01061889
    Epoch: [ 0] [ 550/1365] time: 0.0375, loss: 0.00949600
    Epoch: [ 0] [ 551/1365] time: 0.0402, loss: 0.00955390
    Epoch: [ 0] [ 552/1365] time: 0.0355, loss: 0.00866595
    Epoch: [ 0] [ 553/1365] time: 0.0337, loss: 0.00937292
    Epoch: [ 0] [ 554/1365] time: 0.0343, loss: 0.00915356
    Epoch: [ 0] [ 555/1365] time: 0.0363, loss: 0.00916457
    Epoch: [ 0] [ 556/1365] time: 0.0367, loss: 0.00966748
    Epoch: [ 0] [ 557/1365] time: 0.0429, loss: 0.00993751
    Epoch: [ 0] [ 558/1365] time: 0.0382, loss: 0.01154323
    Epoch: [ 0] [ 559/1365] time: 0.0333, loss: 0.01124658
    Epoch: [ 0] [ 560/1365] time: 0.0345, loss: 0.01044720
    Epoch: [ 0] [ 561/1365] time: 0.0351, loss: 0.00976032
    Epoch: [ 0] [ 562/1365] time: 0.0381, loss: 0.01038584
    Epoch: [ 0] [ 563/1365] time: 0.0405, loss: 0.01126762
    Epoch: [ 0] [ 564/1365] time: 0.0346, loss: 0.01014308
    Epoch: [ 0] [ 565/1365] time: 0.0344, loss: 0.00980890
    Epoch: [ 0] [ 566/1365] time: 0.0375, loss: 0.00976728
    Epoch: [ 0] [ 567/1365] time: 0.0364, loss: 0.00967669
    Epoch: [ 0] [ 568/1365] time: 0.0381, loss: 0.00996929
    Epoch: [ 0] [ 569/1365] time: 0.0398, loss: 0.01088458
    Epoch: [ 0] [ 570/1365] time: 0.0369, loss: 0.01017988
    Epoch: [ 0] [ 571/1365] time: 0.0417, loss: 0.01098312
    Epoch: [ 0] [ 572/1365] time: 0.0355, loss: 0.01012088
    Epoch: [ 0] [ 573/1365] time: 0.0348, loss: 0.01068360
    Epoch: [ 0] [ 574/1365] time: 0.0393, loss: 0.01031387
    Epoch: [ 0] [ 575/1365] time: 0.0385, loss: 0.01086705
    Epoch: [ 0] [ 576/1365] time: 0.0345, loss: 0.00966627
    Epoch: [ 0] [ 577/1365] time: 0.0345, loss: 0.01148316
    Epoch: [ 0] [ 578/1365] time: 0.0349, loss: 0.01156687
    Epoch: [ 0] [ 579/1365] time: 0.0338, loss: 0.01030715
    Epoch: [ 0] [ 580/1365] time: 0.0381, loss: 0.01055788
    Epoch: [ 0] [ 581/1365] time: 0.0390, loss: 0.01018458
    Epoch: [ 0] [ 582/1365] time: 0.0368, loss: 0.01007275
    Epoch: [ 0] [ 583/1365] time: 0.0354, loss: 0.01017951
    Epoch: [ 0] [ 584/1365] time: 0.0352, loss: 0.00958739
    Epoch: [ 0] [ 585/1365] time: 0.0418, loss: 0.01036832
    Epoch: [ 0] [ 586/1365] time: 0.0416, loss: 0.01007814
    Epoch: [ 0] [ 587/1365] time: 0.0390, loss: 0.01129639
    Epoch: [ 0] [ 588/1365] time: 0.0312, loss: 0.01095043
    Epoch: [ 0] [ 589/1365] time: 0.0298, loss: 0.01022910
    Epoch: [ 0] [ 590/1365] time: 0.0305, loss: 0.01007441
    Epoch: [ 0] [ 591/1365] time: 0.0317, loss: 0.00998794
    Epoch: [ 0] [ 592/1365] time: 0.0295, loss: 0.00986134
    Epoch: [ 0] [ 593/1365] time: 0.0331, loss: 0.01017415
    Epoch: [ 0] [ 594/1365] time: 0.0329, loss: 0.00997295
    Epoch: [ 0] [ 595/1365] time: 0.0299, loss: 0.01058532
    Epoch: [ 0] [ 596/1365] time: 0.0290, loss: 0.00942961
    Epoch: [ 0] [ 597/1365] time: 0.0307, loss: 0.00993895
    Epoch: [ 0] [ 598/1365] time: 0.0293, loss: 0.01098401
    Epoch: [ 0] [ 599/1365] time: 0.0290, loss: 0.00969774
    Epoch: [ 0] [ 600/1365] time: 0.0289, loss: 0.00879381
    Epoch: [ 0] [ 601/1365] time: 0.0332, loss: 0.00873485
    Epoch: [ 0] [ 602/1365] time: 0.0370, loss: 0.01069042
    Epoch: [ 0] [ 603/1365] time: 0.0360, loss: 0.01063958
    Epoch: [ 0] [ 604/1365] time: 0.0402, loss: 0.01133719
    Epoch: [ 0] [ 605/1365] time: 0.0354, loss: 0.01277012
    Epoch: [ 0] [ 606/1365] time: 0.0381, loss: 0.01106979
    Epoch: [ 0] [ 607/1365] time: 0.0341, loss: 0.01108931
    Epoch: [ 0] [ 608/1365] time: 0.0295, loss: 0.01144780
    Epoch: [ 0] [ 609/1365] time: 0.0293, loss: 0.01153423
    Epoch: [ 0] [ 610/1365] time: 0.0367, loss: 0.01193100
    Epoch: [ 0] [ 611/1365] time: 0.0337, loss: 0.01139718
    Epoch: [ 0] [ 612/1365] time: 0.0308, loss: 0.01198336
    Epoch: [ 0] [ 613/1365] time: 0.0382, loss: 0.01132763
    Epoch: [ 0] [ 614/1365] time: 0.0400, loss: 0.01188480
    Epoch: [ 0] [ 615/1365] time: 0.0366, loss: 0.01067531
    Epoch: [ 0] [ 616/1365] time: 0.0306, loss: 0.00959054
    Epoch: [ 0] [ 617/1365] time: 0.0298, loss: 0.01097199
    Epoch: [ 0] [ 618/1365] time: 0.0291, loss: 0.01122814
    Epoch: [ 0] [ 619/1365] time: 0.0292, loss: 0.01028652
    Epoch: [ 0] [ 620/1365] time: 0.0331, loss: 0.01104434
    Epoch: [ 0] [ 621/1365] time: 0.0313, loss: 0.01069309
    Epoch: [ 0] [ 622/1365] time: 0.0370, loss: 0.01041724
    Epoch: [ 0] [ 623/1365] time: 0.0350, loss: 0.00959022
    Epoch: [ 0] [ 624/1365] time: 0.0326, loss: 0.01015475
    Epoch: [ 0] [ 625/1365] time: 0.0412, loss: 0.01119549
    Epoch: [ 0] [ 626/1365] time: 0.0340, loss: 0.01167436
    Epoch: [ 0] [ 627/1365] time: 0.0346, loss: 0.01110580
    Epoch: [ 0] [ 628/1365] time: 0.0399, loss: 0.01235114
    Epoch: [ 0] [ 629/1365] time: 0.0366, loss: 0.01222885
    Epoch: [ 0] [ 630/1365] time: 0.0385, loss: 0.01022626
    Epoch: [ 0] [ 631/1365] time: 0.0356, loss: 0.01187145
    Epoch: [ 0] [ 632/1365] time: 0.0413, loss: 0.01215895
    Epoch: [ 0] [ 633/1365] time: 0.0332, loss: 0.01253644
    Epoch: [ 0] [ 634/1365] time: 0.0374, loss: 0.01068714
    Epoch: [ 0] [ 635/1365] time: 0.0294, loss: 0.00977750
    Epoch: [ 0] [ 636/1365] time: 0.0301, loss: 0.01000890
    Epoch: [ 0] [ 637/1365] time: 0.0323, loss: 0.01130225
    Epoch: [ 0] [ 638/1365] time: 0.0355, loss: 0.00970957
    Epoch: [ 0] [ 639/1365] time: 0.0402, loss: 0.00969917
    Epoch: [ 0] [ 640/1365] time: 0.0373, loss: 0.01101154
    Epoch: [ 0] [ 641/1365] time: 0.0400, loss: 0.01170378
    Epoch: [ 0] [ 642/1365] time: 0.0359, loss: 0.01221360
    Epoch: [ 0] [ 643/1365] time: 0.0304, loss: 0.01321377
    Epoch: [ 0] [ 644/1365] time: 0.0297, loss: 0.01327293
    Epoch: [ 0] [ 645/1365] time: 0.0297, loss: 0.01322686
    Epoch: [ 0] [ 646/1365] time: 0.0380, loss: 0.01298804
    Epoch: [ 0] [ 647/1365] time: 0.0346, loss: 0.01309910
    Epoch: [ 0] [ 648/1365] time: 0.0347, loss: 0.01067425
    Epoch: [ 0] [ 649/1365] time: 0.0413, loss: 0.01151666
    Epoch: [ 0] [ 650/1365] time: 0.0416, loss: 0.01111076
    Epoch: [ 0] [ 651/1365] time: 0.0476, loss: 0.01113212
    Epoch: [ 0] [ 652/1365] time: 0.0468, loss: 0.00981062
    Epoch: [ 0] [ 653/1365] time: 0.0360, loss: 0.00943629
    Epoch: [ 0] [ 654/1365] time: 0.0345, loss: 0.00962683
    Epoch: [ 0] [ 655/1365] time: 0.0352, loss: 0.00948156
    Epoch: [ 0] [ 656/1365] time: 0.0434, loss: 0.00939647
    Epoch: [ 0] [ 657/1365] time: 0.0475, loss: 0.01107983
    Epoch: [ 0] [ 658/1365] time: 0.0427, loss: 0.01257993
    Epoch: [ 0] [ 659/1365] time: 0.0459, loss: 0.01032856
    Epoch: [ 0] [ 660/1365] time: 0.0475, loss: 0.01177211
    Epoch: [ 0] [ 661/1365] time: 0.0448, loss: 0.01318937
    Epoch: [ 0] [ 662/1365] time: 0.0438, loss: 0.01184199
    Epoch: [ 0] [ 663/1365] time: 0.0444, loss: 0.01015281
    Epoch: [ 0] [ 664/1365] time: 0.0384, loss: 0.00975054
    Epoch: [ 0] [ 665/1365] time: 0.0471, loss: 0.00977972
    Epoch: [ 0] [ 666/1365] time: 0.0422, loss: 0.01019474
    Epoch: [ 0] [ 667/1365] time: 0.0394, loss: 0.01033096
    Epoch: [ 0] [ 668/1365] time: 0.0403, loss: 0.01035894
    Epoch: [ 0] [ 669/1365] time: 0.0430, loss: 0.01029840
    Epoch: [ 0] [ 670/1365] time: 0.0421, loss: 0.00993079
    Epoch: [ 0] [ 671/1365] time: 0.0445, loss: 0.00953420
    Epoch: [ 0] [ 672/1365] time: 0.0422, loss: 0.00953248
    Epoch: [ 0] [ 673/1365] time: 0.0457, loss: 0.01062972
    Epoch: [ 0] [ 674/1365] time: 0.0435, loss: 0.01063512
    Epoch: [ 0] [ 675/1365] time: 0.0451, loss: 0.01225550
    Epoch: [ 0] [ 676/1365] time: 0.0362, loss: 0.01186710
    Epoch: [ 0] [ 677/1365] time: 0.0346, loss: 0.01115647
    Epoch: [ 0] [ 678/1365] time: 0.0350, loss: 0.01363173
    Epoch: [ 0] [ 679/1365] time: 0.0328, loss: 0.01144295
    Epoch: [ 0] [ 680/1365] time: 0.0311, loss: 0.01001209
    Epoch: [ 0] [ 681/1365] time: 0.0311, loss: 0.01108990
    Epoch: [ 0] [ 682/1365] time: 0.0305, loss: 0.01248755
    Epoch: [ 0] [ 683/1365] time: 0.0311, loss: 0.01201272
    Epoch: [ 0] [ 684/1365] time: 0.0347, loss: 0.01098939
    Epoch: [ 0] [ 685/1365] time: 0.0328, loss: 0.01081829
    Epoch: [ 0] [ 686/1365] time: 0.0369, loss: 0.01231486
    Epoch: [ 0] [ 687/1365] time: 0.0394, loss: 0.01216889
    Epoch: [ 0] [ 688/1365] time: 0.0312, loss: 0.01205965
    Epoch: [ 0] [ 689/1365] time: 0.0383, loss: 0.01192723
    Epoch: [ 0] [ 690/1365] time: 0.0341, loss: 0.01140248
    Epoch: [ 0] [ 691/1365] time: 0.0350, loss: 0.01082364
    Epoch: [ 0] [ 692/1365] time: 0.0350, loss: 0.01047109
    Epoch: [ 0] [ 693/1365] time: 0.0399, loss: 0.01067126
    Epoch: [ 0] [ 694/1365] time: 0.0323, loss: 0.01076985
    Epoch: [ 0] [ 695/1365] time: 0.0292, loss: 0.01025855
    Epoch: [ 0] [ 696/1365] time: 0.0314, loss: 0.00928040
    Epoch: [ 0] [ 697/1365] time: 0.0389, loss: 0.00986592
    Epoch: [ 0] [ 698/1365] time: 0.0330, loss: 0.01027485
    Epoch: [ 0] [ 699/1365] time: 0.0354, loss: 0.01109968
    Epoch: [ 0] [ 700/1365] time: 0.0400, loss: 0.00940601
    Epoch: [ 0] [ 701/1365] time: 0.0337, loss: 0.01004942
    Epoch: [ 0] [ 702/1365] time: 0.0422, loss: 0.01041323
    Epoch: [ 0] [ 703/1365] time: 0.0394, loss: 0.01191629
    Epoch: [ 0] [ 704/1365] time: 0.0452, loss: 0.01229582
    Epoch: [ 0] [ 705/1365] time: 0.0338, loss: 0.01067513
    Epoch: [ 0] [ 706/1365] time: 0.0368, loss: 0.00904051
    Epoch: [ 0] [ 707/1365] time: 0.0391, loss: 0.01114013
    Epoch: [ 0] [ 708/1365] time: 0.0423, loss: 0.01077452
    Epoch: [ 0] [ 709/1365] time: 0.0420, loss: 0.00899822
    Epoch: [ 0] [ 710/1365] time: 0.0362, loss: 0.00979133
    Epoch: [ 0] [ 711/1365] time: 0.0312, loss: 0.01152075
    Epoch: [ 0] [ 712/1365] time: 0.0295, loss: 0.01293853
    Epoch: [ 0] [ 713/1365] time: 0.0295, loss: 0.01141525
    Epoch: [ 0] [ 714/1365] time: 0.0295, loss: 0.01048370
    Epoch: [ 0] [ 715/1365] time: 0.0303, loss: 0.01049879
    Epoch: [ 0] [ 716/1365] time: 0.0316, loss: 0.00890448
    Epoch: [ 0] [ 717/1365] time: 0.0303, loss: 0.01013307
    Epoch: [ 0] [ 718/1365] time: 0.0290, loss: 0.01342440
    Epoch: [ 0] [ 719/1365] time: 0.0279, loss: 0.01439644
    Epoch: [ 0] [ 720/1365] time: 0.0292, loss: 0.00964423
    Epoch: [ 0] [ 721/1365] time: 0.0291, loss: 0.00883965
    Epoch: [ 0] [ 722/1365] time: 0.0303, loss: 0.01112404
    Epoch: [ 0] [ 723/1365] time: 0.0343, loss: 0.01081299
    Epoch: [ 0] [ 724/1365] time: 0.0309, loss: 0.01080788
    Epoch: [ 0] [ 725/1365] time: 0.0325, loss: 0.01022215
    Epoch: [ 0] [ 726/1365] time: 0.0331, loss: 0.01238558
    Epoch: [ 0] [ 727/1365] time: 0.0339, loss: 0.00948257
    Epoch: [ 0] [ 728/1365] time: 0.0295, loss: 0.00979254
    Epoch: [ 0] [ 729/1365] time: 0.0323, loss: 0.01061000
    Epoch: [ 0] [ 730/1365] time: 0.0345, loss: 0.01215069
    Epoch: [ 0] [ 731/1365] time: 0.0284, loss: 0.01288194
    Epoch: [ 0] [ 732/1365] time: 0.0297, loss: 0.01282509
    Epoch: [ 0] [ 733/1365] time: 0.0288, loss: 0.01060635
    Epoch: [ 0] [ 734/1365] time: 0.0343, loss: 0.01035489
    Epoch: [ 0] [ 735/1365] time: 0.0293, loss: 0.01043632
    Epoch: [ 0] [ 736/1365] time: 0.0312, loss: 0.01128827
    Epoch: [ 0] [ 737/1365] time: 0.0315, loss: 0.01027348
    Epoch: [ 0] [ 738/1365] time: 0.0304, loss: 0.01007855
    Epoch: [ 0] [ 739/1365] time: 0.0292, loss: 0.00972704
    Epoch: [ 0] [ 740/1365] time: 0.0296, loss: 0.01086244
    Epoch: [ 0] [ 741/1365] time: 0.0288, loss: 0.01162071
    Epoch: [ 0] [ 742/1365] time: 0.0292, loss: 0.01157359
    Epoch: [ 0] [ 743/1365] time: 0.0303, loss: 0.01238136
    Epoch: [ 0] [ 744/1365] time: 0.0302, loss: 0.01027317
    Epoch: [ 0] [ 745/1365] time: 0.0314, loss: 0.01208466
    Epoch: [ 0] [ 746/1365] time: 0.0316, loss: 0.00979873
    Epoch: [ 0] [ 747/1365] time: 0.0311, loss: 0.00964724
    Epoch: [ 0] [ 748/1365] time: 0.0306, loss: 0.01000615
    Epoch: [ 0] [ 749/1365] time: 0.0293, loss: 0.00874894
    Epoch: [ 0] [ 750/1365] time: 0.0303, loss: 0.00850447
    Epoch: [ 0] [ 751/1365] time: 0.0297, loss: 0.00956917
    Epoch: [ 0] [ 752/1365] time: 0.0317, loss: 0.01113079
    Epoch: [ 0] [ 753/1365] time: 0.0293, loss: 0.01157014
    Epoch: [ 0] [ 754/1365] time: 0.0304, loss: 0.01431988
    Epoch: [ 0] [ 755/1365] time: 0.0296, loss: 0.00997755
    Epoch: [ 0] [ 756/1365] time: 0.0292, loss: 0.01154228
    Epoch: [ 0] [ 757/1365] time: 0.0315, loss: 0.00971518
    Epoch: [ 0] [ 758/1365] time: 0.0345, loss: 0.00974232
    Epoch: [ 0] [ 759/1365] time: 0.0295, loss: 0.00954968
    Epoch: [ 0] [ 760/1365] time: 0.0287, loss: 0.00965393
    Epoch: [ 0] [ 761/1365] time: 0.0288, loss: 0.00953378
    Epoch: [ 0] [ 762/1365] time: 0.0349, loss: 0.01073273
    Epoch: [ 0] [ 763/1365] time: 0.0383, loss: 0.01109383
    Epoch: [ 0] [ 764/1365] time: 0.0442, loss: 0.01078697
    Epoch: [ 0] [ 765/1365] time: 0.0408, loss: 0.01303062
    Epoch: [ 0] [ 766/1365] time: 0.0357, loss: 0.01082619
    Epoch: [ 0] [ 767/1365] time: 0.0359, loss: 0.01006520
    Epoch: [ 0] [ 768/1365] time: 0.0352, loss: 0.00977985
    Epoch: [ 0] [ 769/1365] time: 0.0346, loss: 0.00969885
    Epoch: [ 0] [ 770/1365] time: 0.0347, loss: 0.00934540
    Epoch: [ 0] [ 771/1365] time: 0.0389, loss: 0.01007592
    Epoch: [ 0] [ 772/1365] time: 0.0327, loss: 0.01046056
    Epoch: [ 0] [ 773/1365] time: 0.0315, loss: 0.01111509
    Epoch: [ 0] [ 774/1365] time: 0.0324, loss: 0.01018135
    Epoch: [ 0] [ 775/1365] time: 0.0315, loss: 0.01059377
    Epoch: [ 0] [ 776/1365] time: 0.0320, loss: 0.01046371
    Epoch: [ 0] [ 777/1365] time: 0.0377, loss: 0.00921717
    Epoch: [ 0] [ 778/1365] time: 0.0375, loss: 0.00931624
    Epoch: [ 0] [ 779/1365] time: 0.0333, loss: 0.00984933
    Epoch: [ 0] [ 780/1365] time: 0.0309, loss: 0.00928781
    Epoch: [ 0] [ 781/1365] time: 0.0310, loss: 0.01089006
    Epoch: [ 0] [ 782/1365] time: 0.0313, loss: 0.01077406
    Epoch: [ 0] [ 783/1365] time: 0.0349, loss: 0.01092257
    Epoch: [ 0] [ 784/1365] time: 0.0320, loss: 0.01091607
    Epoch: [ 0] [ 785/1365] time: 0.0318, loss: 0.01191856
    Epoch: [ 0] [ 786/1365] time: 0.0312, loss: 0.01661264
    Epoch: [ 0] [ 787/1365] time: 0.0326, loss: 0.01425591
    Epoch: [ 0] [ 788/1365] time: 0.0317, loss: 0.01195746
    Epoch: [ 0] [ 789/1365] time: 0.0342, loss: 0.01008389
    Epoch: [ 0] [ 790/1365] time: 0.0353, loss: 0.01030045
    Epoch: [ 0] [ 791/1365] time: 0.0351, loss: 0.00928610
    Epoch: [ 0] [ 792/1365] time: 0.0343, loss: 0.00906744
    Epoch: [ 0] [ 793/1365] time: 0.0355, loss: 0.01208370
    Epoch: [ 0] [ 794/1365] time: 0.0430, loss: 0.01017956
    Epoch: [ 0] [ 795/1365] time: 0.0402, loss: 0.00907306
    Epoch: [ 0] [ 796/1365] time: 0.0396, loss: 0.00899995
    Epoch: [ 0] [ 797/1365] time: 0.0354, loss: 0.01025990
    Epoch: [ 0] [ 798/1365] time: 0.0346, loss: 0.01024077
    Epoch: [ 0] [ 799/1365] time: 0.0319, loss: 0.01115691
    Epoch: [ 0] [ 800/1365] time: 0.0309, loss: 0.01220675
    Epoch: [ 0] [ 801/1365] time: 0.0331, loss: 0.01020098
    Epoch: [ 0] [ 802/1365] time: 0.0363, loss: 0.01000559
    Epoch: [ 0] [ 803/1365] time: 0.0329, loss: 0.01093220
    Epoch: [ 0] [ 804/1365] time: 0.0321, loss: 0.01072681
    Epoch: [ 0] [ 805/1365] time: 0.0336, loss: 0.01067363
    Epoch: [ 0] [ 806/1365] time: 0.0321, loss: 0.01069659
    Epoch: [ 0] [ 807/1365] time: 0.0319, loss: 0.01147459
    Epoch: [ 0] [ 808/1365] time: 0.0358, loss: 0.01107560
    Epoch: [ 0] [ 809/1365] time: 0.0385, loss: 0.00928683
    Epoch: [ 0] [ 810/1365] time: 0.0336, loss: 0.01084379
    Epoch: [ 0] [ 811/1365] time: 0.0395, loss: 0.00980415
    Epoch: [ 0] [ 812/1365] time: 0.0382, loss: 0.01013940
    Epoch: [ 0] [ 813/1365] time: 0.0375, loss: 0.01123291
    Epoch: [ 0] [ 814/1365] time: 0.0399, loss: 0.00972689
    Epoch: [ 0] [ 815/1365] time: 0.0437, loss: 0.00895968
    Epoch: [ 0] [ 816/1365] time: 0.0365, loss: 0.00938585
    Epoch: [ 0] [ 817/1365] time: 0.0338, loss: 0.00935008
    Epoch: [ 0] [ 818/1365] time: 0.0410, loss: 0.00930448
    Epoch: [ 0] [ 819/1365] time: 0.0402, loss: 0.00905155
    Epoch: [ 0] [ 820/1365] time: 0.0335, loss: 0.00918459
    Epoch: [ 0] [ 821/1365] time: 0.0366, loss: 0.01112454
    Epoch: [ 0] [ 822/1365] time: 0.0423, loss: 0.00927751
    Epoch: [ 0] [ 823/1365] time: 0.0451, loss: 0.00963047
    Epoch: [ 0] [ 824/1365] time: 0.0361, loss: 0.01137153
    Epoch: [ 0] [ 825/1365] time: 0.0414, loss: 0.01103306
    Epoch: [ 0] [ 826/1365] time: 0.0365, loss: 0.01088526
    Epoch: [ 0] [ 827/1365] time: 0.0423, loss: 0.01053435
    Epoch: [ 0] [ 828/1365] time: 0.0388, loss: 0.01072026
    Epoch: [ 0] [ 829/1365] time: 0.0339, loss: 0.01346388
    Epoch: [ 0] [ 830/1365] time: 0.0320, loss: 0.01299201
    Epoch: [ 0] [ 831/1365] time: 0.0306, loss: 0.01027208
    Epoch: [ 0] [ 832/1365] time: 0.0378, loss: 0.00964053
    Epoch: [ 0] [ 833/1365] time: 0.0307, loss: 0.00915983
    Epoch: [ 0] [ 834/1365] time: 0.0356, loss: 0.00853182
    Epoch: [ 0] [ 835/1365] time: 0.0324, loss: 0.00894045
    Epoch: [ 0] [ 836/1365] time: 0.0293, loss: 0.01071688
    Epoch: [ 0] [ 837/1365] time: 0.0333, loss: 0.01093440
    Epoch: [ 0] [ 838/1365] time: 0.0326, loss: 0.00980900
    Epoch: [ 0] [ 839/1365] time: 0.0334, loss: 0.01315758
    Epoch: [ 0] [ 840/1365] time: 0.0307, loss: 0.01053548
    Epoch: [ 0] [ 841/1365] time: 0.0327, loss: 0.00874523
    Epoch: [ 0] [ 842/1365] time: 0.0294, loss: 0.01187080
    Epoch: [ 0] [ 843/1365] time: 0.0296, loss: 0.01161345
    Epoch: [ 0] [ 844/1365] time: 0.0294, loss: 0.00972866
    Epoch: [ 0] [ 845/1365] time: 0.0301, loss: 0.00980159
    Epoch: [ 0] [ 846/1365] time: 0.0313, loss: 0.00917788
    Epoch: [ 0] [ 847/1365] time: 0.0295, loss: 0.00913144
    Epoch: [ 0] [ 848/1365] time: 0.0290, loss: 0.00907678
    Epoch: [ 0] [ 849/1365] time: 0.0282, loss: 0.00982859
    Epoch: [ 0] [ 850/1365] time: 0.0294, loss: 0.01024789
    Epoch: [ 0] [ 851/1365] time: 0.0362, loss: 0.01001292
    Epoch: [ 0] [ 852/1365] time: 0.0313, loss: 0.00979173
    Epoch: [ 0] [ 853/1365] time: 0.0292, loss: 0.00948669
    Epoch: [ 0] [ 854/1365] time: 0.0307, loss: 0.00909244
    Epoch: [ 0] [ 855/1365] time: 0.0299, loss: 0.01092569
    Epoch: [ 0] [ 856/1365] time: 0.0304, loss: 0.01169529
    Epoch: [ 0] [ 857/1365] time: 0.0297, loss: 0.00891769
    Epoch: [ 0] [ 858/1365] time: 0.0296, loss: 0.00947198
    Epoch: [ 0] [ 859/1365] time: 0.0308, loss: 0.01147632
    Epoch: [ 0] [ 860/1365] time: 0.0295, loss: 0.01042796
    Epoch: [ 0] [ 861/1365] time: 0.0322, loss: 0.01098680
    Epoch: [ 0] [ 862/1365] time: 0.0292, loss: 0.01287566
    Epoch: [ 0] [ 863/1365] time: 0.0292, loss: 0.00933627
    Epoch: [ 0] [ 864/1365] time: 0.0289, loss: 0.00911309
    Epoch: [ 0] [ 865/1365] time: 0.0287, loss: 0.00966031
    Epoch: [ 0] [ 866/1365] time: 0.0302, loss: 0.01157637
    Epoch: [ 0] [ 867/1365] time: 0.0341, loss: 0.01143211
    Epoch: [ 0] [ 868/1365] time: 0.0360, loss: 0.01056449
    Epoch: [ 0] [ 869/1365] time: 0.0378, loss: 0.01385503
    Epoch: [ 0] [ 870/1365] time: 0.0359, loss: 0.00943636
    Epoch: [ 0] [ 871/1365] time: 0.0302, loss: 0.00859145
    Epoch: [ 0] [ 872/1365] time: 0.0305, loss: 0.01007996
    Epoch: [ 0] [ 873/1365] time: 0.0336, loss: 0.01098502
    Epoch: [ 0] [ 874/1365] time: 0.0303, loss: 0.00849252
    Epoch: [ 0] [ 875/1365] time: 0.0297, loss: 0.00894458
    Epoch: [ 0] [ 876/1365] time: 0.0287, loss: 0.00993854
    Epoch: [ 0] [ 877/1365] time: 0.0288, loss: 0.00950565
    Epoch: [ 0] [ 878/1365] time: 0.0297, loss: 0.00954307
    Epoch: [ 0] [ 879/1365] time: 0.0399, loss: 0.00941609
    Epoch: [ 0] [ 880/1365] time: 0.0384, loss: 0.00868261
    Epoch: [ 0] [ 881/1365] time: 0.0405, loss: 0.00896087
    Epoch: [ 0] [ 882/1365] time: 0.0388, loss: 0.00854399
    Epoch: [ 0] [ 883/1365] time: 0.0349, loss: 0.01180047
    Epoch: [ 0] [ 884/1365] time: 0.0369, loss: 0.01231694
    Epoch: [ 0] [ 885/1365] time: 0.0412, loss: 0.01032113
    Epoch: [ 0] [ 886/1365] time: 0.0334, loss: 0.00997277
    Epoch: [ 0] [ 887/1365] time: 0.0350, loss: 0.00951777
    Epoch: [ 0] [ 888/1365] time: 0.0325, loss: 0.00939326
    Epoch: [ 0] [ 889/1365] time: 0.0386, loss: 0.00917970
    Epoch: [ 0] [ 890/1365] time: 0.0377, loss: 0.00919301
    Epoch: [ 0] [ 891/1365] time: 0.0353, loss: 0.01111166
    Epoch: [ 0] [ 892/1365] time: 0.0337, loss: 0.01027290
    Epoch: [ 0] [ 893/1365] time: 0.0343, loss: 0.00973517
    Epoch: [ 0] [ 894/1365] time: 0.0336, loss: 0.00940098
    Epoch: [ 0] [ 895/1365] time: 0.0382, loss: 0.00899003
    Epoch: [ 0] [ 896/1365] time: 0.0353, loss: 0.00981291
    Epoch: [ 0] [ 897/1365] time: 0.0361, loss: 0.01025057
    Epoch: [ 0] [ 898/1365] time: 0.0356, loss: 0.01143780
    Epoch: [ 0] [ 899/1365] time: 0.0343, loss: 0.01076395
    Epoch: [ 0] [ 900/1365] time: 0.0319, loss: 0.01169094
    Epoch: [ 0] [ 901/1365] time: 0.0296, loss: 0.01118758
    Epoch: [ 0] [ 902/1365] time: 0.0289, loss: 0.01055576
    Epoch: [ 0] [ 903/1365] time: 0.0323, loss: 0.00950104
    Epoch: [ 0] [ 904/1365] time: 0.0358, loss: 0.01131256
    Epoch: [ 0] [ 905/1365] time: 0.0366, loss: 0.01125425
    Epoch: [ 0] [ 906/1365] time: 0.0378, loss: 0.01321252
    Epoch: [ 0] [ 907/1365] time: 0.0332, loss: 0.01284474
    Epoch: [ 0] [ 908/1365] time: 0.0309, loss: 0.01055028
    Epoch: [ 0] [ 909/1365] time: 0.0310, loss: 0.01051545
    Epoch: [ 0] [ 910/1365] time: 0.0381, loss: 0.00983562
    Epoch: [ 0] [ 911/1365] time: 0.0389, loss: 0.00916402
    Epoch: [ 0] [ 912/1365] time: 0.0354, loss: 0.00943234
    Epoch: [ 0] [ 913/1365] time: 0.0357, loss: 0.01027963
    Epoch: [ 0] [ 914/1365] time: 0.0357, loss: 0.01329190
    Epoch: [ 0] [ 915/1365] time: 0.0322, loss: 0.01646791
    Epoch: [ 0] [ 916/1365] time: 0.0329, loss: 0.01544106
    Epoch: [ 0] [ 917/1365] time: 0.0406, loss: 0.01231045
    Epoch: [ 0] [ 918/1365] time: 0.0354, loss: 0.00995842
    Epoch: [ 0] [ 919/1365] time: 0.0347, loss: 0.01081169
    Epoch: [ 0] [ 920/1365] time: 0.0387, loss: 0.01164522
    Epoch: [ 0] [ 921/1365] time: 0.0391, loss: 0.01055534
    Epoch: [ 0] [ 922/1365] time: 0.0420, loss: 0.01036782
    Epoch: [ 0] [ 923/1365] time: 0.0383, loss: 0.01168901
    Epoch: [ 0] [ 924/1365] time: 0.0349, loss: 0.01161558
    Epoch: [ 0] [ 925/1365] time: 0.0427, loss: 0.01132900
    Epoch: [ 0] [ 926/1365] time: 0.0423, loss: 0.01078235
    Epoch: [ 0] [ 927/1365] time: 0.0385, loss: 0.01092818
    Epoch: [ 0] [ 928/1365] time: 0.0427, loss: 0.01240754
    Epoch: [ 0] [ 929/1365] time: 0.0412, loss: 0.01105825
    Epoch: [ 0] [ 930/1365] time: 0.0390, loss: 0.00931135
    Epoch: [ 0] [ 931/1365] time: 0.0353, loss: 0.01319490
    Epoch: [ 0] [ 932/1365] time: 0.0346, loss: 0.01372322
    Epoch: [ 0] [ 933/1365] time: 0.0358, loss: 0.00990826
    Epoch: [ 0] [ 934/1365] time: 0.0350, loss: 0.01082546
    Epoch: [ 0] [ 935/1365] time: 0.0399, loss: 0.01038115
    Epoch: [ 0] [ 936/1365] time: 0.0395, loss: 0.01011050
    Epoch: [ 0] [ 937/1365] time: 0.0457, loss: 0.01018841
    Epoch: [ 0] [ 938/1365] time: 0.0367, loss: 0.01013245
    Epoch: [ 0] [ 939/1365] time: 0.0361, loss: 0.01034216
    Epoch: [ 0] [ 940/1365] time: 0.0394, loss: 0.01012606
    Epoch: [ 0] [ 941/1365] time: 0.0317, loss: 0.01114919
    Epoch: [ 0] [ 942/1365] time: 0.0308, loss: 0.01228000
    Epoch: [ 0] [ 943/1365] time: 0.0313, loss: 0.00913660
    Epoch: [ 0] [ 944/1365] time: 0.0307, loss: 0.00930000
    Epoch: [ 0] [ 945/1365] time: 0.0319, loss: 0.01008864
    Epoch: [ 0] [ 946/1365] time: 0.0317, loss: 0.01100295
    Epoch: [ 0] [ 947/1365] time: 0.0401, loss: 0.01227744
    Epoch: [ 0] [ 948/1365] time: 0.0370, loss: 0.01094795
    Epoch: [ 0] [ 949/1365] time: 0.0365, loss: 0.00969169
    Epoch: [ 0] [ 950/1365] time: 0.0311, loss: 0.01330172
    Epoch: [ 0] [ 951/1365] time: 0.0416, loss: 0.01264552
    Epoch: [ 0] [ 952/1365] time: 0.0319, loss: 0.01565657
    Epoch: [ 0] [ 953/1365] time: 0.0361, loss: 0.01117103
    Epoch: [ 0] [ 954/1365] time: 0.0353, loss: 0.01017584
    Epoch: [ 0] [ 955/1365] time: 0.0355, loss: 0.01259065
    Epoch: [ 0] [ 956/1365] time: 0.0361, loss: 0.01234682
    Epoch: [ 0] [ 957/1365] time: 0.0314, loss: 0.01032372
    Epoch: [ 0] [ 958/1365] time: 0.0360, loss: 0.01364696
    Epoch: [ 0] [ 959/1365] time: 0.0383, loss: 0.01127734
    Epoch: [ 0] [ 960/1365] time: 0.0376, loss: 0.01205361
    Epoch: [ 0] [ 961/1365] time: 0.0346, loss: 0.01177100
    Epoch: [ 0] [ 962/1365] time: 0.0300, loss: 0.00936903
    Epoch: [ 0] [ 963/1365] time: 0.0295, loss: 0.00906023
    Epoch: [ 0] [ 964/1365] time: 0.0293, loss: 0.00976961
    Epoch: [ 0] [ 965/1365] time: 0.0291, loss: 0.00893673
    Epoch: [ 0] [ 966/1365] time: 0.0362, loss: 0.01025617
    Epoch: [ 0] [ 967/1365] time: 0.0315, loss: 0.00973328
    Epoch: [ 0] [ 968/1365] time: 0.0272, loss: 0.00999924
    Epoch: [ 0] [ 969/1365] time: 0.0288, loss: 0.00969749
    Epoch: [ 0] [ 970/1365] time: 0.0304, loss: 0.01010087
    Epoch: [ 0] [ 971/1365] time: 0.0325, loss: 0.01153352
    Epoch: [ 0] [ 972/1365] time: 0.0289, loss: 0.00978770
    Epoch: [ 0] [ 973/1365] time: 0.0401, loss: 0.01013325
    Epoch: [ 0] [ 974/1365] time: 0.0297, loss: 0.01049279
    Epoch: [ 0] [ 975/1365] time: 0.0281, loss: 0.01356531
    Epoch: [ 0] [ 976/1365] time: 0.0368, loss: 0.01138067
    Epoch: [ 0] [ 977/1365] time: 0.0322, loss: 0.00936560
    Epoch: [ 0] [ 978/1365] time: 0.0338, loss: 0.01190993
    Epoch: [ 0] [ 979/1365] time: 0.0338, loss: 0.01233721
    Epoch: [ 0] [ 980/1365] time: 0.0313, loss: 0.00930526
    Epoch: [ 0] [ 981/1365] time: 0.0297, loss: 0.01074050
    Epoch: [ 0] [ 982/1365] time: 0.0347, loss: 0.01080827
    Epoch: [ 0] [ 983/1365] time: 0.0367, loss: 0.01024368
    Epoch: [ 0] [ 984/1365] time: 0.0350, loss: 0.01026657
    Epoch: [ 0] [ 985/1365] time: 0.0310, loss: 0.01016229
    Epoch: [ 0] [ 986/1365] time: 0.0304, loss: 0.01001959
    Epoch: [ 0] [ 987/1365] time: 0.0309, loss: 0.01032674
    Epoch: [ 0] [ 988/1365] time: 0.0373, loss: 0.01052732
    Epoch: [ 0] [ 989/1365] time: 0.0328, loss: 0.01155542
    Epoch: [ 0] [ 990/1365] time: 0.0326, loss: 0.00938337
    Epoch: [ 0] [ 991/1365] time: 0.0343, loss: 0.00976152
    Epoch: [ 0] [ 992/1365] time: 0.0403, loss: 0.01034452
    Epoch: [ 0] [ 993/1365] time: 0.0388, loss: 0.01137494
    Epoch: [ 0] [ 994/1365] time: 0.0399, loss: 0.01107337
    Epoch: [ 0] [ 995/1365] time: 0.0451, loss: 0.00976753
    Epoch: [ 0] [ 996/1365] time: 0.0428, loss: 0.01031896
    Epoch: [ 0] [ 997/1365] time: 0.0418, loss: 0.01044517
    Epoch: [ 0] [ 998/1365] time: 0.0369, loss: 0.01171765
    Epoch: [ 0] [ 999/1365] time: 0.0415, loss: 0.01160805
    Epoch: [ 0] [1000/1365] time: 0.0357, loss: 0.01134673
    Epoch: [ 0] [1001/1365] time: 0.0350, loss: 0.01107399
    Epoch: [ 0] [1002/1365] time: 0.0349, loss: 0.01031098
    Epoch: [ 0] [1003/1365] time: 0.0345, loss: 0.01006135
    Epoch: [ 0] [1004/1365] time: 0.0405, loss: 0.01096927
    Epoch: [ 0] [1005/1365] time: 0.0477, loss: 0.01054722
    Epoch: [ 0] [1006/1365] time: 0.0455, loss: 0.00979665
    Epoch: [ 0] [1007/1365] time: 0.0397, loss: 0.01262339
    Epoch: [ 0] [1008/1365] time: 0.0375, loss: 0.01547083
    Epoch: [ 0] [1009/1365] time: 0.0382, loss: 0.01533757
    Epoch: [ 0] [1010/1365] time: 0.0348, loss: 0.01168708
    Epoch: [ 0] [1011/1365] time: 0.0405, loss: 0.01111939
    Epoch: [ 0] [1012/1365] time: 0.0365, loss: 0.01035488
    Epoch: [ 0] [1013/1365] time: 0.0365, loss: 0.01429167
    Epoch: [ 0] [1014/1365] time: 0.0347, loss: 0.01447864
    Epoch: [ 0] [1015/1365] time: 0.0344, loss: 0.01283040
    Epoch: [ 0] [1016/1365] time: 0.0382, loss: 0.01238232
    Epoch: [ 0] [1017/1365] time: 0.0397, loss: 0.01379132
    Epoch: [ 0] [1018/1365] time: 0.0351, loss: 0.00964692
    Epoch: [ 0] [1019/1365] time: 0.0340, loss: 0.01251254
    Epoch: [ 0] [1020/1365] time: 0.0384, loss: 0.01056094
    Epoch: [ 0] [1021/1365] time: 0.0419, loss: 0.01136762
    Epoch: [ 0] [1022/1365] time: 0.0388, loss: 0.00973126
    Epoch: [ 0] [1023/1365] time: 0.0341, loss: 0.01195912
    Epoch: [ 0] [1024/1365] time: 0.0364, loss: 0.01064182
    Epoch: [ 0] [1025/1365] time: 0.0386, loss: 0.01269916
    Epoch: [ 0] [1026/1365] time: 0.0374, loss: 0.01201514
    Epoch: [ 0] [1027/1365] time: 0.0405, loss: 0.01181687
    Epoch: [ 0] [1028/1365] time: 0.0359, loss: 0.01259734
    Epoch: [ 0] [1029/1365] time: 0.0387, loss: 0.01126699
    Epoch: [ 0] [1030/1365] time: 0.0442, loss: 0.01182588
    Epoch: [ 0] [1031/1365] time: 0.0440, loss: 0.00959936
    Epoch: [ 0] [1032/1365] time: 0.0411, loss: 0.01065766
    Epoch: [ 0] [1033/1365] time: 0.0432, loss: 0.01112179
    Epoch: [ 0] [1034/1365] time: 0.0417, loss: 0.01333120
    Epoch: [ 0] [1035/1365] time: 0.0354, loss: 0.00939979
    Epoch: [ 0] [1036/1365] time: 0.0311, loss: 0.00987315
    Epoch: [ 0] [1037/1365] time: 0.0361, loss: 0.01060547
    Epoch: [ 0] [1038/1365] time: 0.0389, loss: 0.00940967
    Epoch: [ 0] [1039/1365] time: 0.0392, loss: 0.01047415
    Epoch: [ 0] [1040/1365] time: 0.0351, loss: 0.01084948
    Epoch: [ 0] [1041/1365] time: 0.0349, loss: 0.00896497
    Epoch: [ 0] [1042/1365] time: 0.0346, loss: 0.00944889
    Epoch: [ 0] [1043/1365] time: 0.0353, loss: 0.00949914
    Epoch: [ 0] [1044/1365] time: 0.0393, loss: 0.00984299
    Epoch: [ 0] [1045/1365] time: 0.0414, loss: 0.01001985
    Epoch: [ 0] [1046/1365] time: 0.0394, loss: 0.01139945
    Epoch: [ 0] [1047/1365] time: 0.0448, loss: 0.00899794
    Epoch: [ 0] [1048/1365] time: 0.0417, loss: 0.00953660
    Epoch: [ 0] [1049/1365] time: 0.0420, loss: 0.01090270
    Epoch: [ 0] [1050/1365] time: 0.0333, loss: 0.00999267
    Epoch: [ 0] [1051/1365] time: 0.0392, loss: 0.01011256
    Epoch: [ 0] [1052/1365] time: 0.0411, loss: 0.01047455
    Epoch: [ 0] [1053/1365] time: 0.0315, loss: 0.01011792
    Epoch: [ 0] [1054/1365] time: 0.0302, loss: 0.01189115
    Epoch: [ 0] [1055/1365] time: 0.0304, loss: 0.01086484
    Epoch: [ 0] [1056/1365] time: 0.0461, loss: 0.01319982
    Epoch: [ 0] [1057/1365] time: 0.0359, loss: 0.01088592
    Epoch: [ 0] [1058/1365] time: 0.0311, loss: 0.01210121
    Epoch: [ 0] [1059/1365] time: 0.0295, loss: 0.01052733
    Epoch: [ 0] [1060/1365] time: 0.0290, loss: 0.01143574
    Epoch: [ 0] [1061/1365] time: 0.0291, loss: 0.00955327
    Epoch: [ 0] [1062/1365] time: 0.0331, loss: 0.01000870
    Epoch: [ 0] [1063/1365] time: 0.0307, loss: 0.00973957
    Epoch: [ 0] [1064/1365] time: 0.0289, loss: 0.00994548
    Epoch: [ 0] [1065/1365] time: 0.0297, loss: 0.01137079
    Epoch: [ 0] [1066/1365] time: 0.0302, loss: 0.01243959
    Epoch: [ 0] [1067/1365] time: 0.0299, loss: 0.01132874
    Epoch: [ 0] [1068/1365] time: 0.0303, loss: 0.00978397
    Epoch: [ 0] [1069/1365] time: 0.0312, loss: 0.01287605
    Epoch: [ 0] [1070/1365] time: 0.0331, loss: 0.01268794
    Epoch: [ 0] [1071/1365] time: 0.0303, loss: 0.01024152
    Epoch: [ 0] [1072/1365] time: 0.0300, loss: 0.01145180
    Epoch: [ 0] [1073/1365] time: 0.0293, loss: 0.01284324
    Epoch: [ 0] [1074/1365] time: 0.0291, loss: 0.01127443
    Epoch: [ 0] [1075/1365] time: 0.0312, loss: 0.01054708
    Epoch: [ 0] [1076/1365] time: 0.0329, loss: 0.00984630
    Epoch: [ 0] [1077/1365] time: 0.0378, loss: 0.01095347
    Epoch: [ 0] [1078/1365] time: 0.0317, loss: 0.01076199
    Epoch: [ 0] [1079/1365] time: 0.0311, loss: 0.01005801
    Epoch: [ 0] [1080/1365] time: 0.0320, loss: 0.01030866
    Epoch: [ 0] [1081/1365] time: 0.0315, loss: 0.01017944
    Epoch: [ 0] [1082/1365] time: 0.0320, loss: 0.01007095
    Epoch: [ 0] [1083/1365] time: 0.0347, loss: 0.00998789
    Epoch: [ 0] [1084/1365] time: 0.0354, loss: 0.01103613
    Epoch: [ 0] [1085/1365] time: 0.0367, loss: 0.01083609
    Epoch: [ 0] [1086/1365] time: 0.0381, loss: 0.01339462
    Epoch: [ 0] [1087/1365] time: 0.0396, loss: 0.01292581
    Epoch: [ 0] [1088/1365] time: 0.0371, loss: 0.01434042
    Epoch: [ 0] [1089/1365] time: 0.0330, loss: 0.01167187
    Epoch: [ 0] [1090/1365] time: 0.0340, loss: 0.01055604
    Epoch: [ 0] [1091/1365] time: 0.0311, loss: 0.00959087
    Epoch: [ 0] [1092/1365] time: 0.0307, loss: 0.01060534
    Epoch: [ 0] [1093/1365] time: 0.0311, loss: 0.01199930
    Epoch: [ 0] [1094/1365] time: 0.0299, loss: 0.01064827
    Epoch: [ 0] [1095/1365] time: 0.0296, loss: 0.01209836
    Epoch: [ 0] [1096/1365] time: 0.0319, loss: 0.01134704
    Epoch: [ 0] [1097/1365] time: 0.0358, loss: 0.00999729
    Epoch: [ 0] [1098/1365] time: 0.0307, loss: 0.01001967
    Epoch: [ 0] [1099/1365] time: 0.0306, loss: 0.00995962
    Epoch: [ 0] [1100/1365] time: 0.0315, loss: 0.01322762
    Epoch: [ 0] [1101/1365] time: 0.0314, loss: 0.01064025
    Epoch: [ 0] [1102/1365] time: 0.0319, loss: 0.01172335
    Epoch: [ 0] [1103/1365] time: 0.0406, loss: 0.01338152
    Epoch: [ 0] [1104/1365] time: 0.0423, loss: 0.01105345
    Epoch: [ 0] [1105/1365] time: 0.0371, loss: 0.01138057
    Epoch: [ 0] [1106/1365] time: 0.0349, loss: 0.01639030
    Epoch: [ 0] [1107/1365] time: 0.0381, loss: 0.01699610
    Epoch: [ 0] [1108/1365] time: 0.0430, loss: 0.01794033
    Epoch: [ 0] [1109/1365] time: 0.0450, loss: 0.01060381
    Epoch: [ 0] [1110/1365] time: 0.0415, loss: 0.01003995
    Epoch: [ 0] [1111/1365] time: 0.0420, loss: 0.00973488
    Epoch: [ 0] [1112/1365] time: 0.0355, loss: 0.01046461
    Epoch: [ 0] [1113/1365] time: 0.0348, loss: 0.00982343
    Epoch: [ 0] [1114/1365] time: 0.0350, loss: 0.01053286
    Epoch: [ 0] [1115/1365] time: 0.0393, loss: 0.01057070
    Epoch: [ 0] [1116/1365] time: 0.0385, loss: 0.01036234
    Epoch: [ 0] [1117/1365] time: 0.0347, loss: 0.01062550
    Epoch: [ 0] [1118/1365] time: 0.0346, loss: 0.01089544
    Epoch: [ 0] [1119/1365] time: 0.0380, loss: 0.01073771
    Epoch: [ 0] [1120/1365] time: 0.0348, loss: 0.01127565
    Epoch: [ 0] [1121/1365] time: 0.0383, loss: 0.01417000
    Epoch: [ 0] [1122/1365] time: 0.0401, loss: 0.01064160
    Epoch: [ 0] [1123/1365] time: 0.0344, loss: 0.01079230
    Epoch: [ 0] [1124/1365] time: 0.0346, loss: 0.01077761
    Epoch: [ 0] [1125/1365] time: 0.0309, loss: 0.00983772
    Epoch: [ 0] [1126/1365] time: 0.0314, loss: 0.01242181
    Epoch: [ 0] [1127/1365] time: 0.0330, loss: 0.00993404
    Epoch: [ 0] [1128/1365] time: 0.0367, loss: 0.01009686
    Epoch: [ 0] [1129/1365] time: 0.0320, loss: 0.01005145
    Epoch: [ 0] [1130/1365] time: 0.0320, loss: 0.01170140
    Epoch: [ 0] [1131/1365] time: 0.0308, loss: 0.01394810
    Epoch: [ 0] [1132/1365] time: 0.0319, loss: 0.01439018
    Epoch: [ 0] [1133/1365] time: 0.0319, loss: 0.01300651
    Epoch: [ 0] [1134/1365] time: 0.0376, loss: 0.01112513
    Epoch: [ 0] [1135/1365] time: 0.0354, loss: 0.01053786
    Epoch: [ 0] [1136/1365] time: 0.0306, loss: 0.01186419
    Epoch: [ 0] [1137/1365] time: 0.0334, loss: 0.01133125
    Epoch: [ 0] [1138/1365] time: 0.0320, loss: 0.00976722
    Epoch: [ 0] [1139/1365] time: 0.0305, loss: 0.01141077
    Epoch: [ 0] [1140/1365] time: 0.0310, loss: 0.01580909
    Epoch: [ 0] [1141/1365] time: 0.0378, loss: 0.01484155
    Epoch: [ 0] [1142/1365] time: 0.0346, loss: 0.01253555
    Epoch: [ 0] [1143/1365] time: 0.0390, loss: 0.01158003
    Epoch: [ 0] [1144/1365] time: 0.0328, loss: 0.00954256
    Epoch: [ 0] [1145/1365] time: 0.0362, loss: 0.00956835
    Epoch: [ 0] [1146/1365] time: 0.0333, loss: 0.01099906
    Epoch: [ 0] [1147/1365] time: 0.0404, loss: 0.01071045
    Epoch: [ 0] [1148/1365] time: 0.0347, loss: 0.01472708
    Epoch: [ 0] [1149/1365] time: 0.0340, loss: 0.01190078
    Epoch: [ 0] [1150/1365] time: 0.0335, loss: 0.00934088
    Epoch: [ 0] [1151/1365] time: 0.0379, loss: 0.00978979
    Epoch: [ 0] [1152/1365] time: 0.0364, loss: 0.00973606
    Epoch: [ 0] [1153/1365] time: 0.0368, loss: 0.00995266
    Epoch: [ 0] [1154/1365] time: 0.0426, loss: 0.01041897
    Epoch: [ 0] [1155/1365] time: 0.0391, loss: 0.01139652
    Epoch: [ 0] [1156/1365] time: 0.0352, loss: 0.01036791
    Epoch: [ 0] [1157/1365] time: 0.0342, loss: 0.01001250
    Epoch: [ 0] [1158/1365] time: 0.0406, loss: 0.01103972
    Epoch: [ 0] [1159/1365] time: 0.0380, loss: 0.01120989
    Epoch: [ 0] [1160/1365] time: 0.0390, loss: 0.01015604
    Epoch: [ 0] [1161/1365] time: 0.0362, loss: 0.00951324
    Epoch: [ 0] [1162/1365] time: 0.0350, loss: 0.01054863
    Epoch: [ 0] [1163/1365] time: 0.0351, loss: 0.01361241
    Epoch: [ 0] [1164/1365] time: 0.0326, loss: 0.01058382
    Epoch: [ 0] [1165/1365] time: 0.0304, loss: 0.01013382
    Epoch: [ 0] [1166/1365] time: 0.0320, loss: 0.01130073
    Epoch: [ 0] [1167/1365] time: 0.0297, loss: 0.01062403
    Epoch: [ 0] [1168/1365] time: 0.0295, loss: 0.00976135
    Epoch: [ 0] [1169/1365] time: 0.0296, loss: 0.00960943
    Epoch: [ 0] [1170/1365] time: 0.0297, loss: 0.01006659
    Epoch: [ 0] [1171/1365] time: 0.0325, loss: 0.01078457
    Epoch: [ 0] [1172/1365] time: 0.0345, loss: 0.01125073
    Epoch: [ 0] [1173/1365] time: 0.0340, loss: 0.00974700
    Epoch: [ 0] [1174/1365] time: 0.0340, loss: 0.01029833
    Epoch: [ 0] [1175/1365] time: 0.0334, loss: 0.01274642
    Epoch: [ 0] [1176/1365] time: 0.0343, loss: 0.01130011
    Epoch: [ 0] [1177/1365] time: 0.0335, loss: 0.01245226
    Epoch: [ 0] [1178/1365] time: 0.0302, loss: 0.01200222
    Epoch: [ 0] [1179/1365] time: 0.0301, loss: 0.01104973
    Epoch: [ 0] [1180/1365] time: 0.0329, loss: 0.01209354
    Epoch: [ 0] [1181/1365] time: 0.0312, loss: 0.01103829
    Epoch: [ 0] [1182/1365] time: 0.0303, loss: 0.00946269
    Epoch: [ 0] [1183/1365] time: 0.0288, loss: 0.01065219
    Epoch: [ 0] [1184/1365] time: 0.0302, loss: 0.01246157
    Epoch: [ 0] [1185/1365] time: 0.0310, loss: 0.01119849
    Epoch: [ 0] [1186/1365] time: 0.0312, loss: 0.01089657
    Epoch: [ 0] [1187/1365] time: 0.0311, loss: 0.01037614
    Epoch: [ 0] [1188/1365] time: 0.0346, loss: 0.01118172
    Epoch: [ 0] [1189/1365] time: 0.0324, loss: 0.00992209
    Epoch: [ 0] [1190/1365] time: 0.0351, loss: 0.01000156
    Epoch: [ 0] [1191/1365] time: 0.0293, loss: 0.00987096
    Epoch: [ 0] [1192/1365] time: 0.0314, loss: 0.01099829
    Epoch: [ 0] [1193/1365] time: 0.0349, loss: 0.00972676
    Epoch: [ 0] [1194/1365] time: 0.0302, loss: 0.01021231
    Epoch: [ 0] [1195/1365] time: 0.0365, loss: 0.00928163
    Epoch: [ 0] [1196/1365] time: 0.0367, loss: 0.01038870
    Epoch: [ 0] [1197/1365] time: 0.0308, loss: 0.01329598
    Epoch: [ 0] [1198/1365] time: 0.0300, loss: 0.01121736
    Epoch: [ 0] [1199/1365] time: 0.0355, loss: 0.00995117
    Epoch: [ 0] [1200/1365] time: 0.0340, loss: 0.00946526
    Epoch: [ 0] [1201/1365] time: 0.0404, loss: 0.00989367
    Epoch: [ 0] [1202/1365] time: 0.0327, loss: 0.01078323
    Epoch: [ 0] [1203/1365] time: 0.0300, loss: 0.00942109
    Epoch: [ 0] [1204/1365] time: 0.0298, loss: 0.01303831
    Epoch: [ 0] [1205/1365] time: 0.0307, loss: 0.01181312
    Epoch: [ 0] [1206/1365] time: 0.0328, loss: 0.00913274
    Epoch: [ 0] [1207/1365] time: 0.0334, loss: 0.01157686
    Epoch: [ 0] [1208/1365] time: 0.0291, loss: 0.01136579
    Epoch: [ 0] [1209/1365] time: 0.0291, loss: 0.00979780
    Epoch: [ 0] [1210/1365] time: 0.0291, loss: 0.00951910
    Epoch: [ 0] [1211/1365] time: 0.0297, loss: 0.00932433
    Epoch: [ 0] [1212/1365] time: 0.0293, loss: 0.00947464
    Epoch: [ 0] [1213/1365] time: 0.0306, loss: 0.01086060
    Epoch: [ 0] [1214/1365] time: 0.0341, loss: 0.00938051
    Epoch: [ 0] [1215/1365] time: 0.0303, loss: 0.00936851
    Epoch: [ 0] [1216/1365] time: 0.0287, loss: 0.01094334
    Epoch: [ 0] [1217/1365] time: 0.0291, loss: 0.00945960
    Epoch: [ 0] [1218/1365] time: 0.0292, loss: 0.01113532
    Epoch: [ 0] [1219/1365] time: 0.0338, loss: 0.01032241
    Epoch: [ 0] [1220/1365] time: 0.0399, loss: 0.01008635
    Epoch: [ 0] [1221/1365] time: 0.0398, loss: 0.01002563
    Epoch: [ 0] [1222/1365] time: 0.0364, loss: 0.00930737
    Epoch: [ 0] [1223/1365] time: 0.0409, loss: 0.00903735
    Epoch: [ 0] [1224/1365] time: 0.0403, loss: 0.00923193
    Epoch: [ 0] [1225/1365] time: 0.0373, loss: 0.01062948
    Epoch: [ 0] [1226/1365] time: 0.0423, loss: 0.01500271
    Epoch: [ 0] [1227/1365] time: 0.0370, loss: 0.01805849
    Epoch: [ 0] [1228/1365] time: 0.0352, loss: 0.01408037
    Epoch: [ 0] [1229/1365] time: 0.0347, loss: 0.01614440
    Epoch: [ 0] [1230/1365] time: 0.0344, loss: 0.01340690
    Epoch: [ 0] [1231/1365] time: 0.0354, loss: 0.01413032
    Epoch: [ 0] [1232/1365] time: 0.0398, loss: 0.01401609
    Epoch: [ 0] [1233/1365] time: 0.0413, loss: 0.01401030
    Epoch: [ 0] [1234/1365] time: 0.0333, loss: 0.01395208
    Epoch: [ 0] [1235/1365] time: 0.0337, loss: 0.01374440
    Epoch: [ 0] [1236/1365] time: 0.0341, loss: 0.01373700
    Epoch: [ 0] [1237/1365] time: 0.0350, loss: 0.01192154
    Epoch: [ 0] [1238/1365] time: 0.0372, loss: 0.01105073
    Epoch: [ 0] [1239/1365] time: 0.0401, loss: 0.00897545
    Epoch: [ 0] [1240/1365] time: 0.0362, loss: 0.00923139
    Epoch: [ 0] [1241/1365] time: 0.0346, loss: 0.01104807
    Epoch: [ 0] [1242/1365] time: 0.0346, loss: 0.01136997
    Epoch: [ 0] [1243/1365] time: 0.0351, loss: 0.00894908
    Epoch: [ 0] [1244/1365] time: 0.0374, loss: 0.00988421
    Epoch: [ 0] [1245/1365] time: 0.0396, loss: 0.01140448
    Epoch: [ 0] [1246/1365] time: 0.0355, loss: 0.01057541
    Epoch: [ 0] [1247/1365] time: 0.0344, loss: 0.00932366
    Epoch: [ 0] [1248/1365] time: 0.0348, loss: 0.01196860
    Epoch: [ 0] [1249/1365] time: 0.0357, loss: 0.01388004
    Epoch: [ 0] [1250/1365] time: 0.0384, loss: 0.01059955
    Epoch: [ 0] [1251/1365] time: 0.0409, loss: 0.01076805
    Epoch: [ 0] [1252/1365] time: 0.0350, loss: 0.01040514
    Epoch: [ 0] [1253/1365] time: 0.0353, loss: 0.01137643
    Epoch: [ 0] [1254/1365] time: 0.0343, loss: 0.01123349
    Epoch: [ 0] [1255/1365] time: 0.0340, loss: 0.01128095
    Epoch: [ 0] [1256/1365] time: 0.0376, loss: 0.00954920
    Epoch: [ 0] [1257/1365] time: 0.0374, loss: 0.00930521
    Epoch: [ 0] [1258/1365] time: 0.0363, loss: 0.00971557
    Epoch: [ 0] [1259/1365] time: 0.0336, loss: 0.00975067
    Epoch: [ 0] [1260/1365] time: 0.0314, loss: 0.01052624
    Epoch: [ 0] [1261/1365] time: 0.0315, loss: 0.00989753
    Epoch: [ 0] [1262/1365] time: 0.0307, loss: 0.01007478
    Epoch: [ 0] [1263/1365] time: 0.0379, loss: 0.00998327
    Epoch: [ 0] [1264/1365] time: 0.0324, loss: 0.01234128
    Epoch: [ 0] [1265/1365] time: 0.0327, loss: 0.01175732
    Epoch: [ 0] [1266/1365] time: 0.0294, loss: 0.00986166
    Epoch: [ 0] [1267/1365] time: 0.0318, loss: 0.01017963
    Epoch: [ 0] [1268/1365] time: 0.0307, loss: 0.01061595
    Epoch: [ 0] [1269/1365] time: 0.0358, loss: 0.01032115
    Epoch: [ 0] [1270/1365] time: 0.0352, loss: 0.01005036
    Epoch: [ 0] [1271/1365] time: 0.0316, loss: 0.00951496
    Epoch: [ 0] [1272/1365] time: 0.0325, loss: 0.00956894
    Epoch: [ 0] [1273/1365] time: 0.0324, loss: 0.01119597
    Epoch: [ 0] [1274/1365] time: 0.0310, loss: 0.01135905
    Epoch: [ 0] [1275/1365] time: 0.0301, loss: 0.01021619
    Epoch: [ 0] [1276/1365] time: 0.0326, loss: 0.01164305
    Epoch: [ 0] [1277/1365] time: 0.0322, loss: 0.01054975
    Epoch: [ 0] [1278/1365] time: 0.0293, loss: 0.01195483
    Epoch: [ 0] [1279/1365] time: 0.0301, loss: 0.01167582
    Epoch: [ 0] [1280/1365] time: 0.0285, loss: 0.01315068
    Epoch: [ 0] [1281/1365] time: 0.0285, loss: 0.01145999
    Epoch: [ 0] [1282/1365] time: 0.0294, loss: 0.01107323
    Epoch: [ 0] [1283/1365] time: 0.0301, loss: 0.01103211
    Epoch: [ 0] [1284/1365] time: 0.0350, loss: 0.01014876
    Epoch: [ 0] [1285/1365] time: 0.0302, loss: 0.01003134
    Epoch: [ 0] [1286/1365] time: 0.0284, loss: 0.01046908
    Epoch: [ 0] [1287/1365] time: 0.0301, loss: 0.01076706
    Epoch: [ 0] [1288/1365] time: 0.0295, loss: 0.01048952
    Epoch: [ 0] [1289/1365] time: 0.0297, loss: 0.01080734
    Epoch: [ 0] [1290/1365] time: 0.0334, loss: 0.01196236
    Epoch: [ 0] [1291/1365] time: 0.0336, loss: 0.01328491
    Epoch: [ 0] [1292/1365] time: 0.0290, loss: 0.01241917
    Epoch: [ 0] [1293/1365] time: 0.0283, loss: 0.00961472
    Epoch: [ 0] [1294/1365] time: 0.0289, loss: 0.00912474
    Epoch: [ 0] [1295/1365] time: 0.0292, loss: 0.00916234
    Epoch: [ 0] [1296/1365] time: 0.0291, loss: 0.01001318
    Epoch: [ 0] [1297/1365] time: 0.0304, loss: 0.01234318
    Epoch: [ 0] [1298/1365] time: 0.0336, loss: 0.01445868
    Epoch: [ 0] [1299/1365] time: 0.0298, loss: 0.01036344
    Epoch: [ 0] [1300/1365] time: 0.0294, loss: 0.01034290
    Epoch: [ 0] [1301/1365] time: 0.0292, loss: 0.01053669
    Epoch: [ 0] [1302/1365] time: 0.0295, loss: 0.00989921
    Epoch: [ 0] [1303/1365] time: 0.0292, loss: 0.00967307
    Epoch: [ 0] [1304/1365] time: 0.0301, loss: 0.01000510
    Epoch: [ 0] [1305/1365] time: 0.0322, loss: 0.01027246
    Epoch: [ 0] [1306/1365] time: 0.0300, loss: 0.01026816
    Epoch: [ 0] [1307/1365] time: 0.0308, loss: 0.01037396
    Epoch: [ 0] [1308/1365] time: 0.0299, loss: 0.01143780
    Epoch: [ 0] [1309/1365] time: 0.0288, loss: 0.01011193
    Epoch: [ 0] [1310/1365] time: 0.0289, loss: 0.00981015
    Epoch: [ 0] [1311/1365] time: 0.0319, loss: 0.01052458
    Epoch: [ 0] [1312/1365] time: 0.0343, loss: 0.01064508
    Epoch: [ 0] [1313/1365] time: 0.0321, loss: 0.01002312
    Epoch: [ 0] [1314/1365] time: 0.0298, loss: 0.01059581
    Epoch: [ 0] [1315/1365] time: 0.0293, loss: 0.00932998
    Epoch: [ 0] [1316/1365] time: 0.0291, loss: 0.00923032
    Epoch: [ 0] [1317/1365] time: 0.0293, loss: 0.00936282
    Epoch: [ 0] [1318/1365] time: 0.0308, loss: 0.01057055
    Epoch: [ 0] [1319/1365] time: 0.0325, loss: 0.01312167
    Epoch: [ 0] [1320/1365] time: 0.0307, loss: 0.01373660
    Epoch: [ 0] [1321/1365] time: 0.0292, loss: 0.01239934
    Epoch: [ 0] [1322/1365] time: 0.0305, loss: 0.01272420
    Epoch: [ 0] [1323/1365] time: 0.0291, loss: 0.01147959
    Epoch: [ 0] [1324/1365] time: 0.0291, loss: 0.01098424
    Epoch: [ 0] [1325/1365] time: 0.0322, loss: 0.01282860
    Epoch: [ 0] [1326/1365] time: 0.0332, loss: 0.01186381
    Epoch: [ 0] [1327/1365] time: 0.0307, loss: 0.01105738
    Epoch: [ 0] [1328/1365] time: 0.0301, loss: 0.01219784
    Epoch: [ 0] [1329/1365] time: 0.0296, loss: 0.01256119
    Epoch: [ 0] [1330/1365] time: 0.0294, loss: 0.01162121
    Epoch: [ 0] [1331/1365] time: 0.0297, loss: 0.01133303
    Epoch: [ 0] [1332/1365] time: 0.0316, loss: 0.01151707
    Epoch: [ 0] [1333/1365] time: 0.0340, loss: 0.01153178
    Epoch: [ 0] [1334/1365] time: 0.0325, loss: 0.01015005
    Epoch: [ 0] [1335/1365] time: 0.0294, loss: 0.01189392
    Epoch: [ 0] [1336/1365] time: 0.0291, loss: 0.01130904
    Epoch: [ 0] [1337/1365] time: 0.0291, loss: 0.01111847
    Epoch: [ 0] [1338/1365] time: 0.0293, loss: 0.01264139
    Epoch: [ 0] [1339/1365] time: 0.0316, loss: 0.01057983
    Epoch: [ 0] [1340/1365] time: 0.0382, loss: 0.01059128
    Epoch: [ 0] [1341/1365] time: 0.0358, loss: 0.01256333
    Epoch: [ 0] [1342/1365] time: 0.0346, loss: 0.01268695
    Epoch: [ 0] [1343/1365] time: 0.0356, loss: 0.01325932
    Epoch: [ 0] [1344/1365] time: 0.0351, loss: 0.01183985
    Epoch: [ 0] [1345/1365] time: 0.0373, loss: 0.01264037
    Epoch: [ 0] [1346/1365] time: 0.0391, loss: 0.01251099
    Epoch: [ 0] [1347/1365] time: 0.0367, loss: 0.01191451
    Epoch: [ 0] [1348/1365] time: 0.0327, loss: 0.01214654
    Epoch: [ 0] [1349/1365] time: 0.0349, loss: 0.01140731
    Epoch: [ 0] [1350/1365] time: 0.0346, loss: 0.01276339
    Epoch: [ 0] [1351/1365] time: 0.0377, loss: 0.01153913
    Epoch: [ 0] [1352/1365] time: 0.0409, loss: 0.01132276
    Epoch: [ 0] [1353/1365] time: 0.0366, loss: 0.01116588
    Epoch: [ 0] [1354/1365] time: 0.0358, loss: 0.01159018
    Epoch: [ 0] [1355/1365] time: 0.0348, loss: 0.01085093
    Epoch: [ 0] [1356/1365] time: 0.0347, loss: 0.01080583
    Epoch: [ 0] [1357/1365] time: 0.0382, loss: 0.01134036
    Epoch: [ 0] [1358/1365] time: 0.0402, loss: 0.01102290
    Epoch: [ 0] [1359/1365] time: 0.0355, loss: 0.01099798
    Epoch: [ 0] [1360/1365] time: 0.0349, loss: 0.01132508
    Epoch: [ 0] [1361/1365] time: 0.0347, loss: 0.01090703
    Epoch: [ 0] [1362/1365] time: 0.0344, loss: 0.01071203
    Epoch: [ 0] [1363/1365] time: 0.0373, loss: 0.01055275
    Epoch: [ 0] [1364/1365] time: 0.0411, loss: 0.01058536



    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-4-95940d444adf> in <module>()
          9                      data_dir='preprocessed_data/')
         10 
    ---> 11 model.train(input_sequences, label_sequences, 2)
    

    <ipython-input-3-1f15d0b1c15d> in train(self, input_sequences, label_sequences, num_epochs)
        172                 # if np.mod(counter, 500) == 2:
        173                 #   self.save(args.checkpoint_dir, counter)
    --> 174             np.savetxt('avg_loss_txt/averaged_loss_per_epoch_' + str(epoch) + '.txt', loss_per_epoch)
        175 
        176 


    ~/audio3-env/lib/python3.6/site-packages/numpy/lib/npyio.py in savetxt(fname, X, fmt, delimiter, newline, header, footer, comments)
       1190         else:
       1191             if sys.version_info[0] >= 3:
    -> 1192                 fh = open(fname, 'wb')
       1193             else:
       1194                 fh = open(fname, 'w')


    FileNotFoundError: [Errno 2] No such file or directory: 'avg_loss_txt/averaged_loss_per_epoch_0.txt'


학습된 모델을 가지고 불특정 인풋에 대해 이어지는 멜로디를 생성하는 작업을 해봅시다.

현재 노트북의 cpu 컴퓨팅 파워로는 학습을 제대로 진행하기가 어렵습니다. 

미리 동일한 코드로 2000 epoch 학습을 시켜서 저장해놓은 weight값을 불러와서 실제로 어떤식으로 결과물을 출력하는지 확인해보겠습니다.

