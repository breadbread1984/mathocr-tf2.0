#!/usr/bin/python3

import pickle;
import numpy as np;
import tensorflow as tf;

def Bottleneck(input_shape, growth_rate, dropout_rate = 0.2):
    
    # dimension of output is the same as input's
    inputs = tf.keras.Input(shape = input_shape);
    results = tf.keras.layers.BatchNormalization()(inputs);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(filters = 3 * growth_rate, kernel_size = (1,1), use_bias = False)(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(filters = growth_rate, kernel_size = (3,3), padding = 'same', use_bias = False)(results);
    results = tf.keras.layers.Dropout(rate = dropout_rate)(results);
    results = tf.keras.layers.Concatenate()([inputs, results]);
    return tf.keras.Model(inputs = inputs, outputs = results);

def Transition(input_shape, output_filters):
    
    # h,w of output = 1/2 h, 1/2 w of input's
    inputs = tf.keras.Input(shape = input_shape);
    results = tf.keras.layers.BatchNormalization()(inputs);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = (1,1), use_bias = False)(results);
    results = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2))(results);
    return tf.keras.Model(inputs = inputs, outputs = results);

def BottleneckArray(input_shape, growth_rate, depth, dropout_rate = 0.2):
    
    # dimension of output is the same as input's
    inputs = tf.keras.Input(shape = input_shape);
    results = inputs;
    for i in range(depth):
        results = Bottleneck(results.shape[1:], growth_rate, dropout_rate)(results);
    return tf.keras.Model(inputs = inputs, outputs = results);

def Encoder(input_shape, output_filters = 48, dropout_rate = 0.2):
    
    # h,w of outA(low_res) = 1/16 h, 1/16 w of input's
    # h,w of outB(high_res) = 1/8 h, 1/8 w of input's
    inputs = tf.keras.Input(shape = input_shape);
    results = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = (7,7), strides = (2,2), padding = 'same', use_bias = False)(inputs);
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
    return tf.keras.Model(inputs = inputs, outputs = (outA, outB));

def CoverageAttention(input_shape, pred_shape, attn_sum_shape, output_filters, kernel_size):

    inputs = tf.keras.Input(shape = input_shape);
    pred = tf.keras.Input(shape = pred_shape);
    attn_sum = tf.keras.Input(shape = attn_sum_shape);
    # attn_prev.shape = (batch, input height, input width, output_filters)
    attn_prev = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = kernel_size, padding = 'same')(attn_sum);
    # prev.shape = (batch, input_height, input_width, 512)
    prev = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same', use_bias = False)(attn_prev);
    # current.shape = (batch, input_height, input_width, 512)
    current = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same', use_bias = False)(inputs);
    # nxt.shape = (batch, 1, 1, 512)
    nxt = tf.keras.layers.Reshape((1,1,pred.shape[-1],))(pred);
    # response.shape = (batch, input_height, input_width, 512)
    nxt = tf.keras.layers.Lambda(lambda x: tf.tile(x, (1,inputs.shape[1],inputs.shape[2],1)))(nxt);
    s = tf.keras.layers.Add()([nxt,current,prev]);
    response = tf.keras.layers.Lambda(lambda x: tf.math.tanh(x))(s);
    # e_t.shape = (batch, input_height, input_width, 1)
    e_t = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1,1), padding = 'same', use_bias = False)(response);
    # attn.shape = (batch, input_height * input_width)
    attn = tf.keras.layers.Softmax()(tf.keras.layers.Flatten()(e_t));
    # attn.shape = (batch, input_height, input_width, 1)
    attn = tf.keras.layers.Reshape((inputs.shape[1], inputs.shape[2], 1,))(attn);
    new_attn_sum = tf.keras.layers.Add()([attn_sum, attn]);
    # weighted_inputs.shape = (batch, input_height, input_width, input_filters)
    attn = tf.keras.layers.Lambda(lambda x: tf.tile(x, (1,1,1,inputs.shape[-1])))(attn);
    weighted_inputs = tf.keras.layers.Multiply()([attn, inputs]);
    # context.shape = (batch, input_filters)
    context = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis = [1,2]))(weighted_inputs);
    return tf.keras.Model(inputs = (inputs, pred, attn_sum), outputs = (context, new_attn_sum));

def Decoder(low_res_shape, high_res_shape, hidden_shape, attn_sum_low_shape, attn_sum_high_shape, num_classes, embedding_dim = 256, hidden_size = 256):
        
    inputs = tf.keras.Input(shape = (1,)); # previous token
    low_res = tf.keras.Input(shape = low_res_shape); # image low resolution encode
    high_res = tf.keras.Input(shape = high_res_shape); # image high resolution encode
    # context
    hidden = tf.keras.Input(shape = hidden_shape);
    attn_sum_low = tf.keras.Input(shape = attn_sum_low_shape);
    attn_sum_high = tf.keras.Input(shape = attn_sum_high_shape);
        
    # inputs.shape = (batch, seq_length = 1)
    # hidden.shape = (batch, hidden size = 256)
    # embedded.shape = (batch, seq_length = 1,embedding size = 256)
    embedded = tf.keras.layers.Embedding(input_dim = num_classes, output_dim = embedding_dim)(inputs);
    # pred.shape = (batch, hidden size = 256)
    pred = tf.keras.layers.GRU(units = hidden_size)(embedded, initial_state = hidden);
    # u_pred.shape = (batch, 512)
    u_pred = tf.keras.layers.Dense(units = 512, use_bias = False)(pred);
    # context_low.shape = (batch, 512)
    context_low, new_attn_sum_low = CoverageAttention(low_res.shape[1:], u_pred.shape[1:], attn_sum_low.shape[1:], output_filters = 256, kernel_size = (11,11))([low_res, u_pred, attn_sum_low]);
    # context_high.shape = (batch, 512)
    context_high, new_attn_sum_high = CoverageAttention(high_res.shape[1:], u_pred.shape[1:], attn_sum_high.shape[1:], output_filters = 256, kernel_size = (7,7))([high_res, u_pred, attn_sum_high]);
    # context.shape = (batch, 1024)
    context = tf.keras.layers.Concatenate(axis = -1)([context_low,context_high]);
    # gru2_input.shape = (batch,seq_length = 1, 1024)
    gru2_input = tf.keras.layers.Reshape((1,context.shape[-1],))(context);
    # new_hidden.shape = (batch, hidden size = 256)
    new_hidden = tf.keras.layers.GRU(units = hidden_size)(gru2_input, initial_state = pred);
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
    return tf.keras.Model(inputs = (inputs,low_res,high_res,hidden,attn_sum_low,attn_sum_high), outputs = (out, new_hidden, new_attn_sum_low, new_attn_sum_high));

class MathOCR(tf.keras.Model):
    
    START = "<SOS>";
    END = "<EOS>";
    PAD = "<PAD>";
    SPECIAL_TOKENS = [START, END, PAD];
    
    def __init__(self, input_shape = (128,128,1,), output_filters = 48, dropout_rate = 0.2, embedding_dim = 256, hidden_size = 256, tokens_length_max = 90):
        
        super(MathOCR, self).__init__();
        with open('token_id_map.dat','rb') as f:
            self.token_to_id = pickle.load(f);
            self.id_to_token = pickle.load(f);
        assert len(self.token_to_id) == len(self.id_to_token);
        
        self.hidden_size = hidden_size;
        self.tokens_length_max = tokens_length_max;
        self.encoder = Encoder(input_shape[-3:], output_filters, dropout_rate);
        self.decoder = Decoder(
            self.encoder.layers[-2].output_shape[1:],
            self.encoder.layers[-1].output_shape[1:],
            (hidden_size,),
            (tf.constant(input_shape[-3:]) // (16, 16, input_shape[-1],)).numpy(),
            (tf.constant(input_shape[-3:]) // (8, 8, input_shape[-1],)).numpy(),
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
        attn_sum_low = tf.zeros(img_shape // (1, 16, 16, img_shape[-1]));
        attn_sum_high = tf.zeros(img_shape // (1, 8, 8, img_shape[-1]));

        def step(i, prev_token_id, prev_hidden, prev_attn_sum_low, prev_attn_sum_high):
            # predict current token
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
            
        tf.while_loop(lambda i, token_id, hidden, attn_sum_low, attn_sum_high: tf.less(i,self.tokens_length_max - 1), 
                      step, [i, token_id, hidden, attn_sum_low, attn_sum_high]);
        # decoded.shape = (batch, seq_length = 89, num_classes)
        logits_sequence = tf.concat(logits_sequence, axis = 1);
        token_id_sequence = tf.concat(token_id_sequence, axis = 1);

        # convert to readable string
        inputs = token_id_sequence.numpy();
        input_shape = inputs.shape;
        flattened = np.reshape(inputs,(-1));
        outputs = list(map(lambda x: self.id_to_token[x], flattened));
        outputs = np.reshape(outputs,(input_shape[0],-1));
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
        attn_sum_low = tf.zeros(img_shape // (1, 16, 16, img_shape[-1]), dtype = tf.float32);
        attn_sum_high = tf.zeros(img_shape // (1, 8, 8, img_shape[-1]), dtype = tf.float32);

        def step(i, prev_token_id, prev_hidden, prev_attn_sum_low, prev_attn_sum_high):
            # previous.shape = (batch, 1)
            prev_token_id = tf.cond(
                tf.less(tf.random.uniform(shape=(), minval = 0, maxval = 1, dtype = tf.float32),0.5),
                lambda: tokens[:,i:i+1], lambda: prev_token_id
            );
            # predict current token
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
            
        tf.while_loop(lambda i, token_id, hidden, attn_sum_low, attn_sum_high: tf.less(i,self.tokens_length_max - 1), 
                      step, [i, token_id, hidden, attn_sum_low, attn_sum_high]);
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
        img = data.numpy().astype('uint8');
        cv2.imshow('image',img);
        data = tf.expand_dims(data, 0);
        s, _ = mathocr(data);
        print(s);
        cv2.waitKey();

