#!/usr/bin/python3

import tensorflow as tf;

def Bottleneck(input_shape, growth_rate, dropout_rate = 0.2):
    
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
    
    inputs = tf.keras.Input(shape = input_shape);
    results = tf.keras.layers.BatchNormalization()(inputs);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = (1,1), use_bias = False)(results);
    results = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2))(results);
    return tf.keras.Model(inputs = inputs, outputs = results);

def BottleneckArray(input_shape, growth_rate, depth, dropout_rate = 0.2):
    
    inputs = tf.keras.Input(shape = input_shape);
    results = inputs;
    for i in range(depth):
        results = Bottleneck(results.shape[1:], growth_rate, dropout_rate)(results);
    return tf.keras.Model(inputs = inputs, outputs = results);

def Encoder(input_shape, output_filters = 48, dropout_rate = 0.2):
    
    inputs = tf.keras.Input(shape = input_shape);
    results = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = (7,7), strides = (2,2), padding = 'same', use_bias = False)(inputs);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2))(results);
    results = BottleneckArray(results.shape[1:], growth_rate = 24, depth = 16, dropout_rate = dropout_rate)(results);
    results = Transition(results.shape[1:], results.shape[-1] // 2)(results);
    results = BottleneckArray(results.shape[1:], growth_rate = 24, depth = 16, dropout_rate = dropout_rate)(results);
    results = tf.keras.layers.BatchNormalization()(results);
    before_trans2 = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(filters = before_trans2.shape[-1] // 2, kernel_size = (1,1), use_bias = False)(before_trans2);
    results = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2))(results);
    outA = BottleneckArray(results.shape[1:], growth_rate = 24, depth = 16, dropout_rate = dropout_rate)(results);
    outB = BottleneckArray(before_trans2.shape[1:], growth_rate = 24, depth = 8, dropout_rate = dropout_rate)(before_trans2);
    return tf.keras.Model(inputs = inputs, outputs = (outA, outB));

class CoverageAttention(tf.keras.Model):

    def __init__(self, output_filters, kernel_size):
        
        super(CoverageAttention, self).__init__();
        self.attn_history = None;
        self.initial_alpha_shape = None;
        self.conv = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = kernel_size, padding = 'same');
        self.dense1 = tf.keras.layers.Dense(units = 512, use_bias = False);
        self.dense2 = tf.keras.layers.Dense(units = 512, use_bias = False);
        self.dense3 = tf.keras.layers.Dense(units = 1, use_bias = False);

    @tf.function
    def reset_alpha(self):
        
        assert self.initial_alpha_shape is not None;
        self.attn_history = tf.zeros(self.initial_alpha_shape); # a reference to attn

    @tf.function
    def call(self, inputs, pred):

        if self.attn_history = None:
            self.reset(tf.shape(inputs));
            self.initial_alpha_shape = tuple(tf.shape(inputs)[:3]) + (1,);
        # alpha_tm1.shape = (batch, input height, input width, output_filters)
        alpha_tm1 = self.conv(tf.math.reduce_sum(self.attn_history, axis = -1, keepdims = True));
        # alpha_tm1_dim_merged.shape = (batch * input_height * input_width, output_filters)
        alpha_tm1_dim_merged = tf.reshape(alpha_tm1, (-1, tf.shape(alpha_tm1)[3]));
        # prev.shape = (batch, input_height * input_width, 512)
        prev = tf.reshape(self.dense2(alpha_tm1_dim_merged), (-1, tf.shape(inputs)[1] * tf.shape(inputs)[2], 512));
        # inputs_dim_merged.shape = (batch * input_height * input_width, input_filters)
        inputs_dim_merged = tf.reshape(inputs, (-1, tf.shape(inputs)[3]));
        # current.shape = (batch, input_height * input_width, 512)
        current = tf.reshape(self.dense1(inputs_dim_merged), (-1, tf.shape(inputs)[1] * tf.shape(inputs)[2], 512));
        # nxt.shape = (batch, 1, 512)
        nxt = tf.expand_dims(pred, 1);
        # response.shape = (batch, input_height * input_width, 512)
        response = tf.math.tanh(nxt + current + prev);
        # response_dim_merged.shape = (batch * input_height * input_width, 512)
        response_dim_merged = tf.reshape(response, (-1, 512));
        # e_t.shape = (batch, input_height * input_width, 1)
        e_t = tf.reshape(self.dense3(response_dim_merged), (-1, tf.shape(inputs)[1] * tf.shape(inputs)[2], 1));
        # attn.shape = (batch, input_height * input_width, 1)
        attn = tf.nn.softmax(e_t, axis = 1);
        self.attn_history = tf.concat([
            self.attn_history,
            tf.reshape(attn, (-1, tf.shape(inputs)[1], tf.shape(inputs)[2], 1))
        ], axis = -1);
        # cA_t_L.shape = (batch, input_height * input_width, input_filters)
        weighted_inputs = attn * tf.reshape(inputs, (-1, tf.shape(inputs)[1] * tf.shape(inputs)[2], tf.shape(inputs)[3]));
        # context.shape = (batch, input_filters)
        context = tf.math.reduce_sum(weighted_inputs, axis = 1);
        return context;

class Maxout(tf.keras.Model):
    
    def __init__(self, pool_size):
        
        self.pool_size = pool_size;
        
    @tf.function
    def call(self, inputs):
    
        results = tf.keras.layers.Reshape(tuple(tf.shape(inputs)[:-1]) + (tf.shape(inputs)[-1] // self.pool_size, self.pool_size))(inputs);
        results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis = -1));
        return results;

class Decoder(tf.keras.Model):
    
    def __init__(self, num_classes, embedding_dim = 256, hidden_size = 256,  **kwargs)
        
        super(Decoder,self).__init__();
        self.embedding = tf.keras.layers.Embedding(input_dim = num_classes, output_dim = embedding_dim);
        self.gru1 = tf.keras.layers.GRU(units = hidden_size);
        self.gru2 = tf.keras.layers.GRU(units = hidden_size);
        self.dense1 = tf.keras.layers.Dense(units = 512, use_bias = False);
        self.dense2 = tf.keras.layers.Dense(units = embedding_dim, use_bias = False);
        self.dense3 = tf.keras.layers.Dense(units = embedding_dim, use_bias = False);
        self.dense4 = tf.keras.layers.Dense(units = num_classes);
        self.coverage_attn_low = CoverageAttention(output_filters = 256, kernel_size = (11,11));
        self.coverage_attn_high = CoverageAttention(output_filters = 256, kernel_size = (7,7));
        self.maxout = Maxout(2);

    @tf.function
    def reset_attention(self, low_input_shape, high_input_shape):
        
        self.coverage_attn_low.reset_alpha(low_input_shape);
        self.coverage_attn_high.reset_alpha(high_input_shape);
        
    @tf.function
    def call(self, inputs, hidden, low_res, high_res):
        
        assert tf.shape(inputs)[1] == 1;
        # inputs.shape = (batch, seq_length = 1, num classes)
        # hidden.shape = (batch, hidden size = 256)
        # embedded.shape = (batch, seq_length = 1,embedding size = 256)
        embedded = self.embedding(inputs);
        # pred.shape = (batch, hidden size = 256)
        pred = self.gru1(embedded, initial_state = hidden);
        # u_pred.shape = (batch, 512)
        u_pred = self.dense1(pred);
        # context_low.shape = (batch, 512)
        context_low = self.coverage_attn_low(low_res, u_pred);
        # context_high.shape = (batch, 512)
        context_high = self.coverage_attn_high(high_res, u_pred);
        # context.shape = (batch,seq_length = 1024)
        context = tf.expand_dims(tf.concat([contex_low,contex_high], axis = -1), axis = 1);
        # new_hidden.shape = (batch, hidden size = 256)
        new_hidden = self.gru2(context, pred);
        # w_s.shape = (batch, embedding size = 256)
        w_s = self.dense2(new_hidden);
        # w_c.shape = (batch, embedding size = 256)
        w_c = self.dense3(context);
        # out.shape = (batch, embedding size = 256)
        out = tf.squeeze(embedded, axis = 1) +_w_s + w_c;
        # out.shape = (batch, 128)
        out = self.maxout(out);
        # out.shape = (batch, num classes)
        out = self.dense4(out);
        return out, new_hidden; 
