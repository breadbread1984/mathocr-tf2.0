#!/usr/bin/python3

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

class CoverageAttention(tf.keras.Model):

    def __init__(self, attn_shape, output_filters, kernel_size):
        
        super(CoverageAttention, self).__init__();
        self.attn_sum = tf.zeros(attn_shape);
        self.conv1 = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = kernel_size, padding = 'same');
        self.conv2 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same', use_bias = False);
        self.conv3 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same', use_bias = False);
        self.conv4 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1,1), padding = 'same', use_bias = False);
        self.flatten = tf.keras.layers.Flatten();
        self.softmax = tf.keras.layers.Softmax();

    @tf.function
    def call(self, inputs, pred, reset = False):

        # attn_prev.shape = (batch, input height, input width, output_filters)
        attn_prev = tf.cond(
            tf.equal(reset,True),
            lambda: self.conv1(tf.zeros_like(inputs[...,0:1])),
            lambda: self.conv1(self.attn_sum)
        );
        # prev.shape = (batch, input_height, input_width, 512)
        prev = self.conv2(attn_prev);
        # current.shape = (batch, input_height, input_width, 512)
        current = self.conv3(inputs);
        # nxt.shape = (batch, 1, 1, 512)
        nxt = tf.expand_dims(tf.expand_dims(pred, 1),1);
        # response.shape = (batch, input_height, input_width, 512)
        response = tf.math.tanh(nxt + current + prev);
        # e_t.shape = (batch, input_height, input_width, 1)
        e_t = self.conv4(response);
        # attn.shape = (batch, input_height * input_width)
        attn = self.softmax(self.flatten(e_t));
        # attn.shape = (batch, input_height, input_width, 1)
        attn = tf.reshape(attn, (-1, tf.shape(inputs)[1], tf.shape(inputs)[2], 1));
        self.attn_sum = tf.cond(
            tf.equal(reset,True),
            lambda: attn,
            lambda: tf.math.add(self.attn_sum, attn)
        );
        # weighted_inputs.shape = (batch, input_height, input_width, input_filters)
        weighted_inputs = attn * inputs;
        # context.shape = (batch, input_filters)
        context = tf.math.reduce_sum(weighted_inputs, axis = [1,2]);
        return context;

class Maxout(tf.keras.Model):
    
    def __init__(self, pool_size):
        
        super(Maxout, self).__init__();
        self.pool_size = pool_size;
        
    @tf.function
    def call(self, inputs):
    
        results = tf.keras.layers.Reshape(tuple(tf.shape(inputs)[:-1]) + (tf.shape(inputs)[-1] // self.pool_size, self.pool_size))(inputs);
        results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis = -1))(results);
        return results;

class Decoder(tf.keras.Model):
    
    def __init__(self, input_shape, num_classes, embedding_dim = 256, hidden_size = 256):
        
        super(Decoder,self).__init__();
        self.embedding = tf.keras.layers.Embedding(input_dim = num_classes, output_dim = embedding_dim);
        self.gru1 = tf.keras.layers.GRU(units = hidden_size);
        self.gru2 = tf.keras.layers.GRU(units = hidden_size);
        self.dense1 = tf.keras.layers.Dense(units = 512, use_bias = False);
        self.dense2 = tf.keras.layers.Dense(units = embedding_dim, use_bias = False);
        self.dense3 = tf.keras.layers.Dense(units = embedding_dim, use_bias = False);
        self.dense4 = tf.keras.layers.Dense(units = num_classes, use_bias = False);
        input_shape = tf.convert_to_tensor(input_shape);
        self.coverage_attn_low = CoverageAttention(input_shape // (1, 16, 16, input_shape[-1]), output_filters = 256, kernel_size = (11,11));
        self.coverage_attn_high = CoverageAttention(input_shape // (1, 8, 8, input_shape[-1]), output_filters = 256, kernel_size = (7,7));
        self.maxout = Maxout(2);
        
    @tf.function
    def call(self, inputs, low_res, high_res, hidden = None, reset = False):
        
        tf.Assert(tf.equal(tf.shape(inputs)[1],1),[tf.shape(inputs)]);
        # inputs.shape = (batch, seq_length = 1)
        # embedded.shape = (batch, seq_length = 1,embedding size = 256)
        # hidden.shape = (batch, hidden size = 256)
        embedded = self.embedding(inputs);
        # pred.shape = (batch, hidden size = 256)
        pred = tf.cond(
            tf.equal(reset,True),
            lambda:self.gru1(embedded, initial_state = tf.zeros((tf.shape(inputs)[0], self.gru1.units))),
            lambda:self.gru1(embedded, initial_state = hidden)
        );
        # u_pred.shape = (batch, 512)
        u_pred = self.dense1(pred);
        # context_low.shape = (batch, 512)
        context_low = self.coverage_attn_low(low_res, u_pred, reset);
        # context_high.shape = (batch, 512)
        context_high = self.coverage_attn_high(high_res, u_pred, reset);
        # context.shape = (batch,seq_length = 1, 1024)
        context = tf.expand_dims(tf.concat([context_low,context_high], axis = -1), axis = 1);
        # new_hidden.shape = (batch, hidden size = 256)
        new_hidden = self.gru2(context, initial_state = pred);
        # w_s.shape = (batch, embedding size = 256)
        w_s = self.dense2(new_hidden);
        # w_c.shape = (batch, embedding size = 256)
        w_c = self.dense3(context);
        # out.shape = (batch, embedding size = 256)
        out = tf.squeeze(embedded, axis = 1) + w_s + w_c;
        # out.shape = (batch, 128)
        out = self.maxout(out);
        # out.shape = (batch, num classes)
        out = self.dense4(out);
        return out, new_hidden;

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    encoder = Encoder((256,256,1));
    decoder = Decoder((10,128,128,3), 100);
