#!/usr/bin/python3

import os;
import pickle;
import tensorflow as tf;
from Model import Encoder, Decoder;

batch_num = 64;
tokens_length_max = 90;
START = "<SOS>";
END = "<EOS>";
PAD = "<PAD>";
SPECIAL_TOKENS = [START, END, PAD];

def parse_function_generator(pad_code, crop = True, transform = True):
    def parse_function(serialized_example):
        context, sequence = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features = {
                'data': tf.io.FixedLenFeature((), dtype = tf.string, default_value = ''),
                'text_length': tf.io.FixedLenFeature((), dtype = tf.int64, default_value = 0),
                'tokens_length': tf.io.FixedLenFeature((), dtype = tf.int64, default_value = 0)
            },
            sequence_features = {
                'text': tf.io.FixedLenSequenceFeature((), dtype = tf.int64),
                'tokens': tf.io.FixedLenSequenceFeature((), dtype = tf.int64)
            }
        );
        # parse
        data = tf.io.decode_raw(context['data'], out_type = tf.uint8);
        data = tf.reshape(data, (256,256,1));
        data = tf.cast(data, dtype = tf.float32);
        text_length = tf.cast(context['text_length'], dtype = tf.int32);
        tokens_length = tf.cast(context['tokens_length'], dtype = tf.int32);
        text = sequence['text'];
        tokens = sequence['tokens'];
        tf.Assert(tf.equal(tf.shape(text)[0],text_length),[tf.shape(text)]);
        tf.Assert(tf.equal(tf.shape(tokens)[0],tokens_length),[tf.shape(tokens)]);
        
        #preprocess
        if crop:
            pos = tf.where(tf.greater(255. - tf.squeeze(data),0.));
            min_yx = tf.reduce_min(pos, axis = 0);
            max_yx = tf.reduce_max(pos, axis = 0);
            hw = max_yx - min_yx;
            data = tf.image.crop_to_bounding_box(data, min_yx[0], min_yx[1], hw[0], hw[1]);
        if transform:
            data = tf.image.resize(data, (128,128))
            
        # pad to fix length to enable batch
        tokens = tf.pad(tokens, paddings = [[0,tokens_length_max - tf.shape(tokens)[0]]], constant_values = pad_code);
        return data, tokens;
    return parse_function;

def main():
    
    # load token
    with open('token_id_map.dat','rb') as f:
        token_to_id = pickle.load(f);
        id_to_token = pickle.load(f);
    assert len(token_to_id) == len(id_to_token);
    # networks
    encoder = Encoder((128,128,1));
    decoder = Decoder(len(token_to_id));
    # load dataset
    trainset = tf.data.TFRecordDataset('trainset.tfrecord').map(parse_function_generator(token_to_id[PAD], True, True)).shuffle(batch_num).batch(batch_num);
    testset = tf.data.TFRecordDataset('testset.tfrecord').map(parse_function_generator(token_to_id[PAD], True, True)).batch(batch_num);
    # checkpoints utilities
    optimizer = tf.keras.optimizers.Adam(1e-3);
    if False == os.path.exists('encoder_checkpoint'): os.mkdir('encoder_checkpoint');
    if False == os.path.exists('decoder_checkpoint'): os.mkdir('decoder_checkpoint');
    encoder_checkpoint = tf.train.Checkpoint(mode = encoder, optimizer = optimizer, optimizer_step = optimizer.iterations);
    decoder_checkpoint = tf.train.Checkpoint(mode = decoder, optimizer = optimizer, optimizer_step = optimizer.iterations);
    encoder_checkpoint.restore(tf.train.latest_checkpoint('encoder_checkpoint'));
    decoder_checkpoint.restore(tf.train.latest_checkpoint('decoder_checkpoint'));
    # log utilities
    avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
    log = tf.summary.create_file_writer('log');
    while True:
        # data.shape = (batch, 128, 128, 1)
        # tokens.shape = (batch, tokens_length_max = 90)
        for data, tokens in trainset:
            with tf.GradientTape() as tape:
                # context tensors
                hidden = None;
                sequence = tf.ones((batch_num,1)) * token_to_id[START];
                decoded_values = tf.TensorArray(dtype = tf.float32, size = tokens_length_max - 1);
                # encode the input image
                low_res, high_res = encoder(data);
                # decode into a sequence of tokens
                for i in tf.range(tokens_length_max - 1):
                    # random choose whether the previous token is from prediction or from ground truth
                    # previous.shape = (batch, 1)
                    previous = tf.cond(
                        tf.less(tf.random.uniform(shape=(), minval = 0, maxval = 1, dtype = tf.float32),0.5),
                        lambda: tokens[:,i:i+1], lambda: sequence[:,-1:]
                    );
                    # predict current token
                    out, hidden = tf.cond(
                        tf.equal(i,0),
                        lambda:decoder(previous, low_res, high_res, reset = True),
                        lambda:decoder(previous, low_res, high_res, hidden = hidden)
                    );
                    # top1_id.shape = (batch, 1)
                    _, top1_id = tf.math.top_k(out,1);
                    # append sequence
                    sequence = tf.concat([sequence, top1_id], axis = -1);
                    decoded_values.apend(i, out);
                # decoded.shape = (batch, seq_length = 89, num_classes)
                decoded = tf.transpose(decoded_values.stack(), perm = (0,2,1));
                # skip the first start token, only use the following ground truth values
                expected = tf.reshape(tokens[:,1:],(-1,tokens_length_max - 1, 1));
                # get loss
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)(decoded, expected);
            avg_loss.update_state(loss);
            if tf.equal(optimizer.iterations % 100, 0):
                with log.as_default():
                    tf.summary.scalar('loss',avg_loss.result(), step = optimizer.iterations);
                print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
                avg_loss.reset_states();
            grads = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables);
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables));
        # save model every epoch
        encoder_checkpoint.save(os.path.join('encoder_checkpoint','ckpt'));
        decoder_checkpoint.save(os.path.join('decoder_checkpoint','ckpt'));
        if loss < 0.01: break;
    #save the network structure with weights
    encoder.save('encoder.h5');
    decoder.save('decoder.h5');
                    
if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
