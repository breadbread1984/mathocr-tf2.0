#!/usr/bin/python3

import pickle;
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

def Decoder(code_shape, hidden_dim, num_classes):

    # code.shape = (batch, code h, code w, code channel)
    code = tf.keras.Input(code_shape);
    prev_l1_hidden = tf.keras.Input((hidden_dim,));
    prev_l2_hidden = tf.keras.Input((hidden_dim,));
    
    # 1) get attention
    # prev_l2_hidden_seq.shape = (batch, 1, hidden dim)
    prev_l2_hidden_seq = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(prev_l2_hidden);
    # l1_hidden.shape = (batch, hidden dim)
    l1_hidden = tf.keras.layers.GRU(units = hidden_dim)(prev_l2_hidden_seq, initial_state = prev_l1_hidden);
    # l1_hidden_seq.shape = (batch, seq_length = 1, hidden_dim)
    l1_hidden_seq = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(l1_hidden);
    # l2_hidden.shape = (batch, hidden dim)
    l2_hidden = tf.keras.layers.GRU(units = hidden_dim)(l1_hidden_seq, initial_state = prev_l2_hidden);
    # l2_logits = (batch, attention_dim)
    l2_hidden = tf.keras.layers.Dropout(rate = 0.2)(l2_hidden);
    l2_logits = tf.keras.layers.Dense(units = code.shape[1] * code.shape[2])(l2_hidden);
    # attention = (batch, attention_dim)
    attention = tf.keras.layers.Softmax()(l2_logits);
    attention = tf.keras.layers.Reshape((code.shape[1], code.shape[2], 1,))(attention);

    # 2) get attended code
    # attention.shape = (batch, code h, code w, code channel)
    attention = tf.keras.layers.Lambda(lambda x: tf.tile(x, (1,1,1,code.shape[-1])))(attention);
    # weighted_code.shape = (batch, code h, code w, code channel)
    weighted_code = tf.keras.layers.Multiply()([code, attention]);
    # attended_code.shape = (batch, code channel)
    attended_code = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, [1,2]))(weighted_code);
    attended_code = tf.keras.layers.Dropout(rate = 0.2)(attended_code);

    # 3) decode
    # logits.shape = (batch, num_classes)
    logits = tf.keras.layers.Dense(units = num_classes)(attended_code);
    return tf.keras.Model(inputs = (code, prev_l1_hidden, prev_l2_hidden), outputs = (logits, l1_hidden, l2_hidden));

class MathOCR(tf.keras.Model):

    START = "<SOS>";
    END = "<EOS>";
    PAD = "<PAD>";
    SPECIAL_TOKENS = [START, END, PAD];

    def __init__(self, input_shape = (256,256,1), hidden_dim = 256, tokens_length_max = 90):

        super(MathOCR, self).__init__();
        with open('token_id_map.dat','rb') as f:
            self.token_to_id = pickle.load(f);
            self.id_to_token = pickle.load(f);
        assert len(self.token_to_id) == len(self.id_to_token);

        self.encoder = DenseNet(input_shape[-3:]);
        self.decoder = Decoder(self.encoder.output.shape[1:], hidden_dim, len(self.token_to_id));
        self.embedding = tf.keras.layers.Embedding(input_dim = len(self.token_to_id), output_dim = hidden_dim);
        self.l2_dense = tf.keras.layers.Dense(units = hidden_dim);

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
        l1_hidden = self.l2_dense(tf.math.reduce_mean(code, axis = [1,2]));
        l2_hidden = self.embedding(token_id);

        def step(i, prev_l1_hidden, prev_l2_hidden):
            # predict Ua token
            logits, l1_hidden, l2_hidden = self.decoder([code, prev_l1_hidden, prev_l2_hidden]);
            # token id = (batch, 1)
            _, cur_token_id = tf.math.top_k(logits,1);
            # append token id
            cur_token_id = tf.cast(cur_token_id, dtype = tf.int64);
            token_id_sequence.append(tf.expand_dims(cur_token_id, axis = 1));
            # append logits
            logits_sequence.append(tf.expand_dims(cur_out, axis = 1));
            # increase counter
            i = i + 1;
            return i, l1_hidden, l2_hidden;

        tf.while_loop(lambda i, l1_hidden, l2_hidden: tf.less(i,self.tokens_length_max - 1),
                      step, [i, l1_hidden, l2_hidden]);
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

if __name__ == "__main__":

    assert tf.executing_eagerly();
    mathocr = MathOCR();

