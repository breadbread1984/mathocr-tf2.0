#!/usr/bin/python3

import pickle;
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

    def __init__(self, output_filters, kernel_size):
        
        super(CoverageAttention, self).__init__();
        self.conv1 = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = kernel_size, padding = 'same');
        self.conv2 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same', use_bias = False);
        self.conv3 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same', use_bias = False);
        self.conv4 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1,1), padding = 'same', use_bias = False);
        self.flatten = tf.keras.layers.Flatten();
        self.softmax = tf.keras.layers.Softmax();

    @tf.function
    def call(self, inputs, pred, attn_sum):

        # attn_prev.shape = (batch, input height, input width, output_filters)
        attn_prev = self.conv1(attn_sum);
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
        new_attn_sum = attn_sum + attn;
        # weighted_inputs.shape = (batch, input_height, input_width, input_filters)
        weighted_inputs = attn * inputs;
        # context.shape = (batch, input_filters)
        context = tf.math.reduce_sum(weighted_inputs, axis = [1,2]);
        return context, new_attn_sum;

class Decoder(tf.keras.Model):
    
    def __init__(self, num_classes, embedding_dim = 256, hidden_size = 256):
        
        super(Decoder,self).__init__();
        self.embedding = tf.keras.layers.Embedding(input_dim = num_classes, output_dim = embedding_dim);
        self.gru1 = tf.keras.layers.GRU(units = hidden_size);
        self.gru2 = tf.keras.layers.GRU(units = hidden_size);
        self.dense1 = tf.keras.layers.Dense(units = 512, use_bias = False);
        self.dense2 = tf.keras.layers.Dense(units = embedding_dim, use_bias = False);
        self.dense3 = tf.keras.layers.Dense(units = embedding_dim, use_bias = False);
        self.dense4 = tf.keras.layers.Dense(units = num_classes, use_bias = False);
        self.coverage_attn_low = CoverageAttention(output_filters = 256, kernel_size = (11,11));
        self.coverage_attn_high = CoverageAttention(output_filters = 256, kernel_size = (7,7));
        
    @tf.function
    def call(self, inputs, low_res, high_res, context):
        
        tf.Assert(tf.equal(tf.shape(inputs)[1],1),[tf.shape(inputs)]);
        # inputs.shape = (batch, seq_length = 1)
        # embedded.shape = (batch, seq_length = 1,embedding size = 256)
        # hidden.shape = (batch, hidden size = 256)
        embedded = self.embedding(inputs);
        # pred.shape = (batch, hidden size = 256)
        pred = self.gru1(embedded, initial_state = context[0]);
        # u_pred.shape = (batch, 512)
        u_pred = self.dense1(pred);
        # context_low.shape = (batch, 512)
        context_low, new_attn_sum_low = self.coverage_attn_low(low_res, u_pred, context[1]);
        # context_high.shape = (batch, 512)
        context_high, new_attn_sum_high = self.coverage_attn_high(high_res, u_pred, context[2]);
        # context.shape = (batch,seq_length = 1, 1024)
        context = tf.concat([context_low,context_high], axis = -1);
        # new_hidden.shape = (batch, hidden size = 256)
        new_hidden = self.gru2(tf.expand_dims(context, axis = 1), initial_state = pred);
        # w_s.shape = (batch, embedding size = 256)
        w_s = self.dense2(new_hidden);
        # w_c.shape = (batch, embedding size = 256)
        w_c = self.dense3(context);
        # out.shape = (batch, embedding size = 256)
        out = tf.squeeze(embedded, axis = 1) + w_s + w_c;
        # out.shape = (batch, 128)
        out = tf.reshape(out, (-1, tf.shape(out)[-1] // 2, 2));
        out = tf.math.reduce_max(out, axis = -1);
        # out.shape = (batch, num classes)
        out = self.dense4(out);
        new_context = (new_hidden, new_attn_sum_low, new_attn_sum_high)
        return out, new_context;

class MathOCR(tf.keras.Model):
    
    START = "<SOS>";
    END = "<EOS>";
    PAD = "<PAD>";
    SPECIAL_TOKENS = [START, END, PAD];
    
    def __init__(self, input_shape, output_filters = 48, dropout_rate = 0.2, embedding_dim = 256, hidden_size = 256, tokens_length_max = 90):
        
        super(MathOCR, self).__init__();
        with open('token_id_map.dat','rb') as f:
            self.token_to_id = pickle.load(f);
            self.id_to_token = pickle.load(f);
        assert len(self.token_to_id) == len(self.id_to_token);
        self.hidden_size = hidden_size;
        self.tokens_length_max = tokens_length_max;
        self.encoder = Encoder(input_shape, output_filters = output_filters, dropout_rate = dropout_rate);
        self.decoder = Decoder(len(self.token_to_id), embedding_dim = embedding_dim, hidden_size = hidden_size);
        
    @tf.function
    def call(self, image):
        
        img_shape = tf.shape(image);
        batch_num = img_shape[0];
        # context tensors
        context = (
            tf.zeros((batch_num, self.hidden_size)), 
            tf.zeros(img_shape // (1, 16, 16, img_shape[-1])),
            tf.zeros(img_shape // (1, 8, 8, img_shape[-1]))
        );
        sequence = tf.ones((batch_num,1), dtype = tf.int64) * self.token_to_id[self.START];
        decoded_values = tf.TensorArray(dtype = tf.float32, size = self.tokens_length_max - 1);
        # encode the input image
        low_res, high_res = self.encoder(image);
        # decode into a sequence of tokens
        for i in tf.range(self.tokens_length_max - 1):
            # random choose whether the previous token is from prediction or from ground truth
            # previous.shape = (batch, 1)
            previous = sequence[:,-1:];
            # predict current token
            out, context = self.decoder(previous, low_res, high_res, context = context);
            # token id = (batch, 1)
            _, top1_id = tf.math.top_k(out,1);
            # append token id
            sequence = tf.concat([sequence, tf.cast(top1_id, dtype = tf.int64)], axis = -1);
            # append logits
            decoded_values.write(i, out);
        # decoded.shape = (batch, seq_length = 89, num_classes)
        decoded = tf.transpose(decoded_values.stack(), perm = (1,0,2));
        return decoded, sequence;

    def train(self, image, tokens):
        
        img_shape = tf.shape(image);
        batch_num = img_shape[0];
        # context tensors
        context = (
            tf.zeros((batch_num, self.hidden_size)), 
            tf.zeros(img_shape // (1, 16, 16, img_shape[-1])),
            tf.zeros(img_shape // (1, 8, 8, img_shape[-1]))
        );
        sequence = tf.ones((batch_num,1), dtype = tf.int64) * self.token_to_id[self.START];
        decoded_values = tf.TensorArray(dtype = tf.float32, size = self.tokens_length_max - 1);
        # encode the input image
        low_res, high_res = self.encoder(image);
        # decode into a sequence of tokens
        for i in tf.range(self.tokens_length_max - 1):
            # random choose whether the previous token is from prediction or from ground truth
            # previous.shape = (batch, 1)
            previous = tf.cond(
                tf.less(tf.random.uniform(shape=(), minval = 0, maxval = 1, dtype = tf.float32),0.5),
                lambda: tokens[:,i:i+1], lambda: sequence[:,-1:]
            );
            # predict current token
            out, context = self.decoder(previous, low_res, high_res, context = context);
            # token id = (batch, 1)
            _, top1_id = tf.math.top_k(out,1);
            # append token id
            sequence = tf.concat([sequence, tf.cast(top1_id, dtype = tf.int64)], axis = -1);
            # append logits
            decoded_values.write(i, out);
        # decoded.shape = (batch, seq_length = 89, num_classes)
        decoded = tf.transpose(decoded_values.stack(), perm = (1,0,2));
        return decoded, sequence;

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    mathocr = MathOCR((256,256,1));
