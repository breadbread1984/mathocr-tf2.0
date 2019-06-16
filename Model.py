#!/usr/bin/python3

import pickle;
import numpy as np;
import tensorflow as tf;

def DenseNet(input_shape, blocks = 3, level = 16, growth_rate = 24, output_filters = 48, dropout_rate = 0.2):
    
    # output channel is ((48 + 384)/2 + 384)/2 + 384 = 684
    # with default parameters, output dimension of input whose dimension is (h,w,c) is (h/8,w/8,684).
    inputs = tf.keras.Input(shape = input_shape);
    results = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = (7,7), strides = (2,2), padding = 'same', use_bias = False)(inputs);
    results = tf.keras.layers.BatchNormalization(momentum = 0.9, gamma_initializer = 'glorot_uniform', epsilon = 0.0001)(results);
    results = tf.keras.layers.ReLU()(results);
    # NOTE: height and width are halved
    results = tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2))(results);
    for i in range(blocks):
        # denseblock
        # channel growth 16 * 24 = 384
        for j in range(level):
            shortcut = results;
            # conv 1x1
            results = tf.keras.layers.Conv2D(filters = 4 * growth_rate, kernel_size = (1,1), padding = 'same', use_bias = False)(results);
            results = tf.keras.layers.BatchNormalization(momentum = 0.9, gamma_initializer = 'glorot_uniform', epsilon = 0.0001)(results);
            results = tf.keras.layers.ReLU()(results);
            results = tf.keras.layers.Dropout(rate = dropout_rate)(results);
            # conv 3x3
            results = tf.keras.layers.Conv2D(filters = growth_rate, kernel_size = (3,3), padding = 'same', use_bias = False)(results);
            results = tf.keras.layers.BatchNormalization(momentum = 0.9, gamma_initializer = 'glorot_uniform', epsilon = 0.0001)(results);
            results = tf.keras.layers.ReLU()(results);
            results = tf.keras.layers.Dropout(rate = dropout_rate)(results);
            # bottleneck out
            results = tf.keras.layers.Concatenate()([shortcut, results]);
        # transition
        # channel halved
        if i < blocks - 1:
            results = tf.keras.layers.Conv2D(filters = results.shape[-1] // 2, kernel_size = (1,1), padding = 'same', use_bias = False)(results);
            results = tf.keras.layers.BatchNormalization(momentum = 0.9, gamma_initializer = 'glorot_uniform', epsilon = 0.0001)(results);
            results = tf.keras.layers.ReLU()(results);
            results = tf.keras.layers.Dropout(rate = dropout_rate)(results);
            # NOTE: height and width are halved
            results = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2))(results);

    return tf.keras.Model(inputs = inputs, outputs = results);

def Attender(code_shape, hat_s_t_shape, alpha_sum_shape, kernel_size):

    code = tf.keras.Input(shape = code_shape);
    hat_s_t = tf.keras.Input(shape = hat_s_t_shape);
    alpha_sum = tf.keras.Input(shape = alpha_sum_shape);
    # F.shape = (batch, input height, input width, output_filters)
    F = tf.keras.layers.Conv2D(filters = 512, kernel_size = kernel_size, padding = 'same', use_bias = False)(alpha_sum);
    # Uf.shape = (batch, input_height, input_width, 512)
    Uf = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same')(F);
    # Ua.shape = (batch, input_height, input_width, 512)
    Ua = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same')(code);
    # Us.shape = (batch, 1, 1, 512)
    Us = tf.keras.layers.Dense(units = 512)(hat_s_t);
    Us = tf.keras.layers.Reshape((1,1,Us.shape[-1],))(Us);
    # response.shape = (batch, input_height, input_width, 512)
    Us = tf.keras.layers.Lambda(lambda x: tf.tile(x, (1,code.shape[1],code.shape[2],1)))(Us);
    s = tf.keras.layers.Add()([Us,Ua,Uf]);
    response = tf.keras.layers.Lambda(lambda x: tf.math.tanh(x))(s);
    # e_t.shape = (batch, input_height, input_width, 1)
    e_t = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1,1), padding = 'same')(response);
    # alpha_t.shape = (batch, input_height * input_width)
    alpha_t = tf.keras.layers.Softmax()(tf.keras.layers.Flatten()(e_t));
    # alpha_t.shape = (batch, input_height, input_width, 1)
    alpha_t = tf.keras.layers.Reshape((code.shape[1], code.shape[2], 1,))(alpha_t);
    new_alpha_sum = tf.keras.layers.Add()([alpha_sum, alpha_t]);
    # weighted_inputs.shape = (batch, input_height, input_width, input_filters)
    alpha_t = tf.keras.layers.Lambda(lambda x: tf.tile(x, (1,1,1,code.shape[-1])))(alpha_t);
    weighted_inputs = tf.keras.layers.Multiply()([alpha_t, code]);
    # context.shape = (batch, input_filters)
    context = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis = [1,2]))(weighted_inputs);
    return tf.keras.Model(inputs = (code, hat_s_t, alpha_sum), outputs = (context, new_alpha_sum));

def Decoder(code_shape, hidden_shape, alpha_shape, num_classes, embedding_dim = 256, hidden_size = 256, dropout_rate = 0.2):

    prev_token = tf.keras.Input(shape = (1,)); # previous token
    code = tf.keras.Input(shape = code_shape); # image low resolution encode
    # context
    s_tm1 = tf.keras.Input(shape = hidden_shape);
    alpha_sum = tf.keras.Input(shape = alpha_shape);
    # prev_token.shape = (batch, seq_length = 1)
    # s_tm1.shape = (batch, s_tm1 size = 256)
    # y_tm1.shape = (batch, seq_length = 1,embedding size = 256)
    y_tm1 = tf.keras.layers.Embedding(input_dim = num_classes, output_dim = embedding_dim)(prev_token);
    # s_t.shape = (batch, s_tm1 size = 256)
    s_t = tf.keras.layers.GRU(units = hidden_size)(y_tm1, initial_state = s_tm1);
    # hat_s_t.shape = (batch, 512)
    hat_s_t = tf.keras.layers.Dense(units = 512)(s_t);
    # context_low.shape = (batch, 684)
    context, new_alpha_sum = Attender(code.shape[1:], hat_s_t.shape[1:], alpha_sum.shape[1:], kernel_size = (11,11))([code, hat_s_t, alpha_sum]);
    # c_t.shape = (batch,seq_length = 1, 684)
    c_t = tf.keras.layers.Reshape((1,context.shape[-1],))(context);
    # s_t.shape = (batch, s_tm1 size = 256)
    s_t = tf.keras.layers.GRU(units = hidden_size)(c_t, initial_state = s_t);
    # NOTE: original version returns here
    # w_s.shape = (batch, embedding size = 256)
    w_s = tf.keras.layers.Dense(units = embedding_dim)(s_t);
    # w_c.shape = (batch, embedding size = 256)
    w_c = tf.keras.layers.Dense(units = embedding_dim)(context);
    # out.shape = (batch, embedding size = 256)
    y_tm1 = tf.keras.layers.Reshape((y_tm1.shape[-1],))(y_tm1);
    out = tf.keras.layers.Add()([y_tm1, w_s, w_c]);
    # out.shape = (batch, 128)
    out = tf.keras.layers.Reshape((out.shape[-1] // 2, 2,))(out);
    out = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis = -1))(out);
    # out.shape = (batch, num classes)
    out = tf.keras.layers.Dropout(rate = dropout_rate)(out);
    out = tf.keras.layers.Dense(units = num_classes)(out);
    return tf.keras.Model(inputs = (prev_token, code, s_tm1, alpha_sum), outputs = (out, s_t, new_alpha_sum));

class MathOCR(tf.keras.Model):

    START = "<SOS>";
    END = "<EOS>";
    PAD = "<PAD>";
    SPECIAL_TOKENS = [START, END, PAD];

    def __init__(self, input_shape = (256,256,1,), output_filters = 48, dropout_rate = 0.2, embedding_dim = 256, hidden_size = 256, tokens_length_max = 90):

        super(MathOCR, self).__init__();
        with open('token_id_map.dat','rb') as f:
            self.token_to_id = pickle.load(f);
            self.id_to_token = pickle.load(f);
        assert len(self.token_to_id) == len(self.id_to_token);
        
        self.hidden_size = hidden_size;
        self.tokens_length_max = tokens_length_max;
        self.encoder = DenseNet(input_shape[-3:], 3, 16, 24, output_filters, dropout_rate);
        self.dense = tf.keras.layers.Dense(units = self.hidden_size, activation = tf.math.tanh);
        self.decoder = Decoder(
            self.encoder.outputs[0].shape[1:],
            (hidden_size,),
            (tf.constant(input_shape[-3:]) // (16, 16, input_shape[-1],)).numpy(),
            len(self.token_to_id),
            embedding_dim,
            hidden_size,
            dropout_rate
        );

    def call(self, image):

        img_shape = tf.shape(image);
        batch_num = img_shape[0];
        # whole sequence of token id
        token_id_sequence = [tf.ones((batch_num,1,1), dtype = tf.int64) * self.token_to_id[self.START]];
        # decoded sequence without without head
        logits_sequence = [];
        # encode the input image
        code = self.encoder(image);
        # loop variables
        i = tf.constant(0);
        token_id = tf.ones((batch_num,1), dtype = tf.int64) * self.token_to_id[self.START];
        s_t = self.dense(tf.math.reduce_mean(code, axis = [1,2]));
        alpha_sum = tf.zeros(img_shape // (1, 16, 16, img_shape[-1]), dtype = tf.float32);

        def step(i, prev_token_id, s_tm1, prev_alpha_sum):
            # predict Ua token
            cur_out, s_t, cur_alpha_sum = self.decoder([prev_token_id, code, s_tm1, prev_alpha_sum]);
            # token id = (batch, 1)
            _, cur_token_id = tf.math.top_k(cur_out,1);
            # append token id
            cur_token_id = tf.cast(cur_token_id, dtype = tf.int64);
            token_id_sequence.append(tf.expand_dims(cur_token_id, axis = 1));
            # append logits
            logits_sequence.append(tf.expand_dims(cur_out, axis = 1));
            # increase counter
            i = i + 1;
            return i, cur_token_id, s_t, cur_alpha_sum;

        tf.while_loop(lambda i, token_id, s_t, alpha_sum: tf.less(i,self.tokens_length_max - 1), 
                      step, [i, token_id, s_t, alpha_sum]);
        # decoded.shape = (batch, seq_length = 89, num_classes)
        logits_sequence = tf.concat(logits_sequence, axis = 1);
        token_id_sequence = tf.concat(token_id_sequence, axis = 1);

        # convert to readable string
        inputs = token_id_sequence.numpy();
        inputs_shape = inputs.shape;
        flattened = np.reshape(inputs,(-1));
        outputs = list(map(lambda x: self.id_to_token[x], flattened));
        outputs = np.reshape(outputs,(inputs_shape[0],-1));
        outputs = [''.join(sample) for sample in outputs]; 
        return outputs, logits_sequence;

    def loss(self, image, tokens):

        img_shape = tf.shape(image);
        batch_num = img_shape[0];
        # encode the input image
        code = self.encoder(image);
        common_code = tf.reshape(
            tf.tile(tf.expand_dims(code, axis = 1), (1, self.tokens_length_max - 2, 1, 1, 1)),
            (-1, code.shape[-3], code.shape[-2], code.shape[-1])
        ); # code.shape = (batch * (tokens_length_max - 2), code h, code w, code channel)
        # prev_token.shape(batch * (token_length_max - 2),1)
        prev_token0 = tf.reshape(tokens[:,0:self.tokens_length_max - 2],(-1,1));
        prev_token1 = tf.reshape(tokens[:,1:self.tokens_length_max - 1],(-1,1));
        # initial values of loop variables
        s_t0 = self.dense(tf.math.reduce_mean(code, axis = [1,2]));
        s_t0 = tf.reshape(
            tf.tile(tf.expand_dims(s_t0, axis = 1), (1, self.tokens_length_max - 2, 1)),
            (-1, self.hidden_size)
        ); # s_t.shape = (batch * (tokens_length_max - 2), hidden_size)
        alpha_sum0 = tf.zeros(
            (batch_num * (self.tokens_length_max - 2), code.shape[-3], code.shape[-2], 1), 
            dtype = tf.float32); # alpha_sum.shape = (batch * (tokens_length_max - 2), code h, code w, 1)
        # propagation for 2 steps
        out1, s_t1, alpha_sum1 = self.decoder([prev_token0, common_code, s_t0, alpha_sum0]);
        out2, s_t2, alpha_sum2 = self.decoder([prev_token1, common_code, s_t1, alpha_sum1]);
        # decoded.shape = (batch, tokens_length_max - 2, num_classes)
        logits = tf.reshape(out2, (-1, self.tokens_length_max - 2, len(self.token_to_id)));
        expected = tokens[:, 2:self.tokens_length_max];
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)(expected, logits);
        return loss;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    import cv2;
    from train_mathocr import parse_function_generator;
    mathocr = MathOCR();
    optimizer = tf.keras.optimizers.Adam(1e-3);
    checkpoint = tf.train.Checkpoint(model = mathocr, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoint'));
    testset = tf.data.TFRecordDataset('testset.tfrecord').map(parse_function_generator(mathocr.token_to_id[mathocr.PAD], True, True));
    for data, tokens in testset:
        img = (data.numpy() * 255.).astype('uint8');
        cv2.imshow('image',img);
        data = tf.expand_dims(data, 0);
        s, _ = mathocr(data);
        print('predicted:',s);
        tokens = tf.expand_dims(tokens,0);
        inputs = tokens.numpy();
        inputs_shape = inputs.shape;
        flattened = np.reshape(inputs,(-1));
        outputs = list(map(lambda x: mathocr.id_to_token[x], flattened));
        outputs = np.reshape(outputs,(inputs_shape[0],-1));
        outputs = [''.join(sample) for sample in outputs];
        print('expected:',outputs[0]);
        cv2.waitKey();
