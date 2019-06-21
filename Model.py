#!/usr/bin/python3

import pickle;
import numpy as np;
import tensorflow as tf;

def Encoder(input_shape):

    inputs = tf.keras.Input(input_shape[-3:]);
    model = tf.keras.applications.MobileNetV2(input_tensor = inputs, weights='imagenet', include_top = False);
    model.trainable = False;
    return tf.keras.Model(inputs = inputs, outputs = (model.layers[-36].output,model.layers[-98].output));

def CoverageAttention(code_shape, hat_s_t_shape, alpha_sum_shape, output_filters, kernel_size):

    code = tf.keras.Input(shape = code_shape);
    hat_s_t = tf.keras.Input(shape = hat_s_t_shape);
    alpha_sum = tf.keras.Input(shape = alpha_sum_shape);
    # F.shape = (batch, input height, input width, output_filters)
    F = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = kernel_size, padding = 'same')(alpha_sum);
    # Uf.shape = (batch, input_height, input_width, 512)
    Uf = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same')(F);
    # Ua.shape = (batch, input_height, input_width, 512)
    Ua = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), padding = 'same')(code);
    # Us.shape = (batch, 1, 1, 512)
    Us = tf.keras.layers.Reshape((1,1,hat_s_t.shape[-1],))(hat_s_t);
    # response.shape = (batch, input_height, input_width, 512)
    Us = tf.keras.layers.Lambda(lambda x,h,w: tf.tile(x, (1,h,w,1)), arguments={'h':code.shape[1],'w':code.shape[2]})(Us);
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
    alpha_t = tf.keras.layers.Lambda(lambda x,c: tf.tile(x, (1,1,1,c)), arguments={'c':code.shape[-1]})(alpha_t);
    weighted_inputs = tf.keras.layers.Multiply()([alpha_t, code]);
    # context.shape = (batch, input_filters)
    context = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis = [1,2]))(weighted_inputs);
    return tf.keras.Model(inputs = (code, hat_s_t, alpha_sum), outputs = (context, new_alpha_sum));

def Decoder(low_res_shape, high_res_shape, hidden_shape, attn_sum_low_shape, attn_sum_high_shape, num_classes, embedding_dim = 256, hidden_size = 256):
        
    prev_token = tf.keras.Input(shape = (1,)); # previous token
    low_res = tf.keras.Input(shape = low_res_shape); # image low resolution encode
    high_res = tf.keras.Input(shape = high_res_shape); # image high resolution encode
    # context
    s_tm1 = tf.keras.Input(shape = hidden_shape);
    alpha_sum_low = tf.keras.Input(shape = attn_sum_low_shape);
    alpha_sum_high = tf.keras.Input(shape = attn_sum_high_shape);
        
    # prev_token.shape = (batch, seq_length = 1)
    # s_tm1.shape = (batch, s_tm1 size = 256)
    # y_tm1.shape = (batch, seq_length = 1,embedding size = 256)
    y_tm1 = tf.keras.layers.Embedding(input_dim = num_classes, output_dim = embedding_dim)(prev_token);
    # s_t.shape = (batch, s_tm1 size = 256)
    s_t = tf.keras.layers.GRU(units = hidden_size)(y_tm1, initial_state = s_tm1);
    # hat_s_t.shape = (batch, 512)
    hat_s_t = tf.keras.layers.Dense(units = 512)(s_t);
    # context_low.shape = (batch, 512)
    context_low, new_attn_sum_low = CoverageAttention(low_res.shape[1:], hat_s_t.shape[1:], alpha_sum_low.shape[1:], output_filters = 256, kernel_size = (11,11))([low_res, hat_s_t, alpha_sum_low]);
    # context_high.shape = (batch, 512)
    context_high, new_attn_sum_high = CoverageAttention(high_res.shape[1:], hat_s_t.shape[1:], alpha_sum_high.shape[1:], output_filters = 256, kernel_size = (7,7))([high_res, hat_s_t, alpha_sum_high]);
    # context.shape = (batch, 1024)
    context = tf.keras.layers.Concatenate(axis = -1)([context_low,context_high]);
    # c_t.shape = (batch,seq_length = 1, 1024)
    c_t = tf.keras.layers.Reshape((1,context.shape[-1],))(context);
    # s_t.shape = (batch, s_tm1 size = 256)
    s_t = tf.keras.layers.GRU(units = hidden_size)(c_t, initial_state = s_t);
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
    out = tf.keras.layers.Dense(units = num_classes)(out);
    return tf.keras.Model(inputs = (prev_token, low_res, high_res, s_tm1, alpha_sum_low, alpha_sum_high), outputs = (out, s_t, new_attn_sum_low, new_attn_sum_high));

class MathOCR(tf.keras.Model):
    
    START = "<SOS>";
    END = "<EOS>";
    PAD = "<PAD>";
    SPECIAL_TOKENS = [START, END, PAD];
    
    def __init__(self, input_shape = (256,256,3,), output_filters = 48, dropout_rate = 0.2, embedding_dim = 256, hidden_size = 256, tokens_length_max = 90):
        
        super(MathOCR, self).__init__();
        with open('token_id_map.dat','rb') as f:
            self.token_to_id = pickle.load(f);
            self.id_to_token = pickle.load(f);
        assert len(self.token_to_id) == len(self.id_to_token);
        
        self.hidden_size = hidden_size;
        self.tokens_length_max = tokens_length_max;
        self.encoder = Encoder(input_shape[-3:]);
        self.dense = tf.keras.layers.Dense(units = hidden_size, activation = tf.math.tanh);
        self.decoder = Decoder(
            self.encoder.outputs[0].shape[1:],
            self.encoder.outputs[1].shape[1:],
            (hidden_size,),
            (tf.constant(input_shape[-3:]) // (16, 16, input_shape[-1],)).numpy(),
            (tf.constant(input_shape[-3:]) // (8, 8, input_shape[-1],)).numpy(),
            len(self.token_to_id),
            embedding_dim,
            hidden_size
        );

    @tf.function
    def call(self, image):
        
        img_shape = tf.shape(image);
        batch_num = img_shape[0];
        # whole sequence of token id
        token_id_sequence = tf.TensorArray(dtype = tf.int64, size = self.tokens_length_max, clear_after_read = False);
        token_id_sequence.write(0,tf.ones((batch_num,1), dtype = tf.int64) * self.token_to_id[self.START]);
        # decoded sequence without without head
        logits_sequence = tf.TensorArray(dtype = tf.float32, size = self.tokens_length_max - 1, clear_after_read = False);
        # encode the input image
        low_res, high_res = self.encoder(image);
        # loop variables
        i = tf.constant(0);
        token_id = tf.ones((batch_num,1), dtype = tf.int64) * self.token_to_id[self.START];
        s_t = self.dense(tf.math.reduce_mean(low_res, axis = [1,2]));
        alpha_sum_low = tf.zeros(img_shape // (1, 16, 16, img_shape[-1]), dtype = tf.float32);
        alpha_sum_high = tf.zeros(img_shape // (1, 8, 8, img_shape[-1]), dtype = tf.float32);

        @tf.function
        def step(i, prev_token_id, s_tm1, prev_alpha_sum_low, prev_alpha_sum_high):
            # predict Ua token
            cur_out, s_t, cur_attn_sum_low, cur_attn_sum_high = self.decoder([prev_token_id, low_res, high_res, s_tm1, prev_alpha_sum_low, prev_alpha_sum_high]);
            # token id = (batch, 1)
            _, cur_token_id = tf.math.top_k(cur_out,1);
            # append token id
            cur_token_id = tf.cast(cur_token_id, dtype = tf.int64);
            token_id_sequence.write(i + 1, tf.cast(cur_token_id, dtype = tf.int64));
            # append logits
            logits_sequence.write(i, cur_out);
            # increase counter
            i = i + 1;
            return i, cur_token_id, s_t, cur_attn_sum_low, cur_attn_sum_high;
            
        tf.while_loop(lambda i, token_id, s_t, alpha_sum_low, alpha_sum_high: tf.less(i,self.tokens_length_max - 1), 
                      step, [i, token_id, s_t, alpha_sum_low, alpha_sum_high]);
        # decoded.shape = (batch, seq_length = 89, num_classes)
        logits_sequence = tf.transpose(logits_sequence.stack(), perm = (1,0,2));
        token_id_sequence = tf.transpose(token_id_sequence.stack(), perm = (1,0,2));

        return token_id_sequence, logits_sequence;

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
        s_t = self.dense(tf.math.reduce_mean(low_res, axis = [1,2]));
        alpha_sum_low = tf.zeros(img_shape // (1, 16, 16, img_shape[-1]), dtype = tf.float32);
        alpha_sum_high = tf.zeros(img_shape // (1, 8, 8, img_shape[-1]), dtype = tf.float32);

        def step(i, prev_token_id, s_tm1, prev_alpha_sum_low, prev_alpha_sum_high):
            # previous.shape = (batch, 1)
            prev_token_id = tokens[:,i:i+1];
            # predict Ua token
            cur_out, s_t, cur_attn_sum_low, cur_attn_sum_high = self.decoder([prev_token_id, low_res, high_res, s_tm1, prev_alpha_sum_low, prev_alpha_sum_high]);
            # token id = (batch, 1)
            _, cur_token_id = tf.math.top_k(cur_out,1);
            # cast value type
            cur_token_id = tf.cast(cur_token_id, dtype = tf.int64);
            # append logits
            logits_sequence.append(tf.expand_dims(cur_out, axis = 1));
            # increase counter
            i = i + 1;
            return i, cur_token_id, s_t, cur_attn_sum_low, cur_attn_sum_high;
            
        tf.while_loop(lambda i, token_id, s_t, alpha_sum_low, alpha_sum_high: tf.less(i,self.tokens_length_max - 1), 
                      step, [i, token_id, s_t, alpha_sum_low, alpha_sum_high]);
        # decoded.shape = (batch, seq_length = 89, num_classes)
        logits_sequence = tf.concat(logits_sequence, axis = 1);
        return logits_sequence;

def convert_to_readable(token_id_sequence, id_to_token):

    # convert to readable string
    inputs = token_id_sequence.numpy();
    inputs_shape = inputs.shape;
    flattened = np.reshape(inputs,(-1));
    outputs = list(map(lambda x: id_to_token[x], flattened));
    outputs = np.reshape(outputs,(inputs_shape[0],-1));
    outputs = [''.join(sample) for sample in outputs];
    return outputs;

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    import cv2;
    from train_mathocr import parse_function_generator;
    mathocr = MathOCR((128,128,3));
    optimizer = tf.keras.optimizers.Adam(1e-3);
    checkpoint = tf.train.Checkpoint(model = mathocr, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoint'));
    testset = tf.data.TFRecordDataset('testset.tfrecord').map(parse_function_generator(mathocr.token_to_id[mathocr.PAD], True, True));
    for data, tokens in testset:
        img = (data.numpy() * 255.).astype('uint8');
        cv2.imshow('image',img);
        data = tf.expand_dims(data, 0);
        data = tf.tile(data,(1,1,1,3));
        token_id_sequence, _ = mathocr(data);
        s = convert_to_readable(token_id_sequence, mathocr.id_to_token);
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

