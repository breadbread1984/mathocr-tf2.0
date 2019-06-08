#!/usr/bin/python3

import pickle;
import numpy as np;
import tensorflow as tf;

def Bottleneck(code_shape, growth_rate, dropout_rate = 0.2):
    
    # dimension of output is the same as input's
    code = tf.keras.Input(shape = code_shape);
    results = tf.keras.layers.BatchNormalization()(code);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(filters = 3 * growth_rate, kernel_size = (1,1), use_bias = False)(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(filters = growth_rate, kernel_size = (3,3), padding = 'same', use_bias = False)(results);
    results = tf.keras.layers.Dropout(rate = dropout_rate)(results);
    results = tf.keras.layers.Concatenate()([code, results]);
    return tf.keras.Model(code = code, outputs = results);

def Transition(code_shape, output_filters):
    
    # h,w of output = 1/2 h, 1/2 w of input's
    code = tf.keras.Input(shape = code_shape);
    results = tf.keras.layers.BatchNormalization()(code);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = (1,1), use_bias = False)(results);
    results = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2))(results);
    return tf.keras.Model(code = code, outputs = results);

def BottleneckArray(code_shape, growth_rate, depth, dropout_rate = 0.2):
    
    # dimension of output is the same as input's
    code = tf.keras.Input(shape = code_shape);
    results = code;
    for i in range(depth):
        results = Bottleneck(results.shape[1:], growth_rate, dropout_rate)(results);
    return tf.keras.Model(code = code, outputs = results);

def Encoder(code_shape, output_filters = 48, dropout_rate = 0.2):
    
    # h,w of outA(low_res) = 1/16 h, 1/16 w of input's
    # h,w of outB(high_res) = 1/8 h, 1/8 w of input's
    code = tf.keras.Input(shape = code_shape);
    results = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = (7,7), strides = (2,2), padding = 'same', use_bias = False)(code);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
    # height and width are halved
    results = tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2))(results);
    results = BottleneckArray(results.shape[1:], growth_rate = 24, depth = 16, dropout_rate = dropout_rate)(results);
    # height and width are halved
    results = Transition(results.shape[1:], results.shape[-1] // 2)(results);
    results = BottleneckArray(results.shape[1:], growth_rate = 24, depth = 16, dropout_rate = dropout_rate)(results);
    results = tf.keras.layers.BatchNormalization()(results);
    before_trans2 = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(filters = before_trans2.shape[-1] // 2, kernel_size = (1,1), use_bias = False)(before_trans2);
    # height and width are halved
    results = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2))(results);
    outA = BottleneckArray(results.shape[1:], growth_rate = 24, depth = 16, dropout_rate = dropout_rate)(results);
    outB = BottleneckArray(before_trans2.shape[1:], growth_rate = 24, depth = 8, dropout_rate = dropout_rate)(before_trans2);
    return tf.keras.Model(code = code, outputs = (outA, outB));

def CoverageAttention(code_shape, s_t_shape, alpha_sum_shape, output_filters, kernel_size):

    code = tf.keras.Input(shape = code_shape);
    s_t = tf.keras.Input(shape = s_t_shape);
    alpha_sum = tf.keras.Input(shape = alpha_sum_shape);
    # F.shape = (batch, input height, input width, output_filters)
    F = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = kernel_size, padding = 'same')(alpha_sum);
    # Uf.shape = (batch, input_height, input_width, 512)
    Uf = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same', use_bias = False)(F);
    # Ua.shape = (batch, input_height, input_width, 512)
    Ua = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same', use_bias = False)(code);
    # Us.shape = (batch, 1, 1, 512)
    Us = tf.keras.layers.Reshape((1,1,s_t.shape[-1],))(s_t);
    # response.shape = (batch, input_height, input_width, 512)
    Us = tf.keras.layers.Lambda(lambda x: tf.tile(x, (1,code.shape[1],code.shape[2],1)))(Us);
    s = tf.keras.layers.Add()([Us,Ua,Uf]);
    response = tf.keras.layers.Lambda(lambda x: tf.math.tanh(x))(s);
    # e_t.shape = (batch, input_height, input_width, 1)
    e_t = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1,1), padding = 'same', use_bias = False)(response);
    # alpha_t.shape = (batch, input_height * input_width)
    alpha_t = tf.keras.layers.Softmax()(tf.keras.layers.Flatten()(e_t));
    # alpha_t.shape = (batch, input_height, input_width, 1)
    alpha_t = tf.keras.layers.Reshape((code.shape[1], code.shape[2], 1,))(alpha_t);
    new_alpha_sum = tf.keras.layers.Add()([alpha_sum, tf.stop_gradient(alpha_t)]);
    # weighted_inputs.shape = (batch, input_height, input_width, input_filters)
    alpha_t = tf.keras.layers.Lambda(lambda x: tf.tile(x, (1,1,1,code.shape[-1])))(alpha_t);
    weighted_inputs = tf.keras.layers.Multiply()([alpha_t, code]);
    # context.shape = (batch, input_filters)
    context = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis = [1,2]))(weighted_inputs);
    return tf.keras.Model(code = (code, s_t, alpha_sum), outputs = (context, new_alpha_sum));

def Decoder(low_res_shape, high_res_shape, hidden_shape, attn_sum_low_shape, attn_sum_high_shape, num_classes, embedding_dim = 256, hidden_size = 256):
        
    code = tf.keras.Input(shape = (1,)); # previous token
    low_res = tf.keras.Input(shape = low_res_shape); # image low resolution encode
    high_res = tf.keras.Input(shape = high_res_shape); # image high resolution encode
    # context
    hidden = tf.keras.Input(shape = hidden_shape);
    alpha_sum_low = tf.keras.Input(shape = attn_sum_low_shape);
    alpha_sum_high = tf.keras.Input(shape = attn_sum_high_shape);
        
    # code.shape = (batch, seq_length = 1)
    # hidden.shape = (batch, hidden size = 256)
    # embedded.shape = (batch, seq_length = 1,embedding size = 256)
    embedded = tf.keras.layers.Embedding(input_dim = num_classes, output_dim = embedding_dim)(code);
    # s_t.shape = (batch, hidden size = 256)
    s_t = tf.keras.layers.GRU(units = hidden_size)(embedded, initial_state = hidden);
    # s_t.shape = (batch, 512)
    s_t = tf.keras.layers.Dense(units = 512, use_bias = False)(s_t);
    # context_low.shape = (batch, 512)
    context_low, new_attn_sum_low = CoverageAttention(low_res.shape[1:], s_t.shape[1:], alpha_sum_low.shape[1:], output_filters = 256, kernel_size = (11,11))([low_res, s_t, alpha_sum_low]);
    # context_high.shape = (batch, 512)
    context_high, new_attn_sum_high = CoverageAttention(high_res.shape[1:], s_t.shape[1:], alpha_sum_high.shape[1:], output_filters = 256, kernel_size = (7,7))([high_res, s_t, alpha_sum_high]);
    # context.shape = (batch, 1024)
    context = tf.keras.layers.Concatenate(axis = -1)([context_low,context_high]);
    # c_t.shape = (batch,seq_length = 1, 1024)
    c_t = tf.keras.layers.Reshape((1,context.shape[-1],))(context);
    # new_hidden.shape = (batch, hidden size = 256)
    new_hidden = tf.keras.layers.GRU(units = hidden_size)(c_t, initial_state = s_t);
    # w_s.shape = (batch, embedding size = 256)
    w_s = tf.keras.layers.Dense(units = embedding_dim, use_bias = False)(new_hidden);
    # w_c.shape = (batch, embedding size = 256)
    w_c = tf.keras.layers.Dense(units = embedding_dim, use_bias = False)(context);
    # out.shape = (batch, embedding size = 256)
    embedded = tf.keras.layers.Reshape((embedded.shape[-1],))(embedded);
    out = tf.keras.layers.Add()([embedded, w_s, w_c]);
    # out.shape = (batch, 128)
    out = tf.keras.layers.Reshape((out.shape[-1] // 2, 2,))(out);
    out = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis = -1))(out);
    # out.shape = (batch, num classes)
    out = tf.keras.layers.Dense(units = num_classes, use_bias = False)(out);
    return tf.keras.Model(code = (code,low_res,high_res,hidden,alpha_sum_low,alpha_sum_high), outputs = (out, tf.stop_gradient(new_hidden), new_attn_sum_low, new_attn_sum_high));

class MathOCR(tf.keras.Model):
    
    START = "<SOS>";
    END = "<EOS>";
    PAD = "<PAD>";
    SPECIAL_TOKENS = [START, END, PAD];
    
    def __init__(self, code_shape = (128,128,1,), output_filters = 48, dropout_rate = 0.2, embedding_dim = 256, hidden_size = 256, tokens_length_max = 90):
        
        super(MathOCR, self).__init__();
        with open('token_id_map.dat','rb') as f:
            self.token_to_id = pickle.load(f);
            self.id_to_token = pickle.load(f);
        assert len(self.token_to_id) == len(self.id_to_token);
        
        self.hidden_size = hidden_size;
        self.tokens_length_max = tokens_length_max;
        self.encoder = Encoder(code_shape[-3:], output_filters, dropout_rate);
        self.decoder = Decoder(
            self.encoder.layers[-2].output_shape[1:],
            self.encoder.layers[-1].output_shape[1:],
            (hidden_size,),
            (tf.constant(code_shape[-3:]) // (16, 16, code_shape[-1],)).numpy(),
            (tf.constant(code_shape[-3:]) // (8, 8, code_shape[-1],)).numpy(),
            len(self.token_to_id),
            embedding_dim,
            hidden_size
        );
        
    def call(self, image):
        
        img_shape = tf.shape(image);
        batch_num = img_shape[0];
        # whole sequence of token id
        token_id_sequence = [tf.ones((batch_num,1,1), dtype = tf.int64) * self.token_to_id[self.START]];
        # decoded sequence without without head
        logits_sequence = [];
        # encode the input image
        low_res, high_res = self.encoder(image);
        # loop variables
        i = tf.constant(0);
        token_id = tf.ones((batch_num,1), dtype = tf.int64) * self.token_to_id[self.START];
        out = tf.zeros((batch_num, len(self.token_to_id)), dtype = tf.float32);
        hidden = tf.zeros((batch_num, self.hidden_size));
        alpha_sum_low = tf.zeros(img_shape // (1, 16, 16, img_shape[-1]));
        alpha_sum_high = tf.zeros(img_shape // (1, 8, 8, img_shape[-1]));

        def step(i, prev_token_id, prev_hidden, prev_attn_sum_low, prev_attn_sum_high):
            # predict Ua token
            cur_out, cur_hidden, cur_attn_sum_low, cur_attn_sum_high = self.decoder([prev_token_id, low_res, high_res, prev_hidden, prev_attn_sum_low, prev_attn_sum_high]);
            # token id = (batch, 1)
            _, cur_token_id = tf.math.top_k(cur_out,1);
            # append token id
            cur_token_id = tf.cast(cur_token_id, dtype = tf.int64);
            token_id_sequence.append(tf.expand_dims(cur_token_id, axis = 1));
            # append logits
            logits_sequence.append(tf.expand_dims(cur_out, axis = 1));
            # increase counter
            i = i + 1;
            return i, cur_token_id, cur_hidden, cur_attn_sum_low, cur_attn_sum_high;
            
        tf.while_loop(lambda i, token_id, hidden, alpha_sum_low, alpha_sum_high: tf.less(i,self.tokens_length_max - 1), 
                      step, [i, token_id, hidden, alpha_sum_low, alpha_sum_high]);
        # decoded.shape = (batch, seq_length = 89, num_classes)
        logits_sequence = tf.concat(logits_sequence, axis = 1);
        token_id_sequence = tf.concat(token_id_sequence, axis = 1);

        # convert to readable string
        code = token_id_sequence.numpy();
        code_shape = code.shape;
        flattened = np.reshape(code,(-1));
        outputs = list(map(lambda x: self.id_to_token[x], flattened));
        outputs = np.reshape(outputs,(code_shape[0],-1));
        outputs = [''.join(sample) for sample in outputs]; 
        return outputs, logits_sequence;

    def train(self, image, tokens):
        
        img_shape = tf.shape(image);
        batch_num = img_shape[0];
        # decoded sequence without without head
        logits_sequence = [];
        # encode the input image
        low_res, high_res = self.encoder(image);
        # loop variables
        i = tf.constant(0);
        token_id = tf.ones((batch_num,1), dtype = tf.int64) * self.token_to_id[self.START];
        out = tf.zeros((batch_num, len(self.token_to_id)), dtype = tf.float32);
        hidden = tf.zeros((batch_num, self.hidden_size), dtype = tf.float32);
        alpha_sum_low = tf.zeros(img_shape // (1, 16, 16, img_shape[-1]), dtype = tf.float32);
        alpha_sum_high = tf.zeros(img_shape // (1, 8, 8, img_shape[-1]), dtype = tf.float32);

        def step(i, prev_token_id, prev_hidden, prev_attn_sum_low, prev_attn_sum_high):
            # previous.shape = (batch, 1)
            prev_token_id = tf.cond(
                tf.less(tf.random.uniform(shape=(), minval = 0, maxval = 1, dtype = tf.float32),0.5),
                lambda: tokens[:,i:i+1], lambda: prev_token_id
            );
            # predict Ua token
            cur_out, cur_hidden, cur_attn_sum_low, cur_attn_sum_high = self.decoder([prev_token_id, low_res, high_res, prev_hidden, prev_attn_sum_low, prev_attn_sum_high]);
            # token id = (batch, 1)
            _, cur_token_id = tf.math.top_k(cur_out,1);
            # cast value type
            cur_token_id = tf.cast(cur_token_id, dtype = tf.int64);
            # append logits
            logits_sequence.append(tf.expand_dims(cur_out, axis = 1));
            # increase counter
            i = i + 1;
            return i, cur_token_id, cur_hidden, cur_attn_sum_low, cur_attn_sum_high;
            
        tf.while_loop(lambda i, token_id, hidden, alpha_sum_low, alpha_sum_high: tf.less(i,self.tokens_length_max - 1), 
                      step, [i, token_id, hidden, alpha_sum_low, alpha_sum_high]);
        # decoded.shape = (batch, seq_length = 89, num_classes)
        logits_sequence = tf.concat(logits_sequence, axis = 1);
        return logits_sequence;

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
        print(s);
        cv2.waitKey();

