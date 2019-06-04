#!/usr/bin/python3

import os;
import pickle;
import tensorflow as tf;
from Model import MathOCR;

batch_num = 8;
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
    
    # networks
    mathocr = MathOCR(tokens_length_max = tokens_length_max);
    # load dataset
    trainset = tf.data.TFRecordDataset('trainset.tfrecord').map(parse_function_generator(mathocr.token_to_id[PAD], True, True)).shuffle(batch_num).batch(batch_num);
    testset = tf.data.TFRecordDataset('testset.tfrecord').map(parse_function_generator(mathocr.token_to_id[PAD], True, True)).batch(batch_num);
    # checkpoints utilities
    optimizer = tf.keras.optimizers.Adam(1e-3);
    if False == os.path.exists('checkpoint'): os.mkdir('checkpoint');
    checkpoint = tf.train.Checkpoint(mode = mathocr, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoint'));
    # log utilities
    avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
    log = tf.summary.create_file_writer('checkpoint');
    while True:
        # data.shape = (batch, 128, 128, 1)
        # tokens.shape = (batch, tokens_length_max = 90)
        for data, tokens in trainset:
            with tf.GradientTape() as tape:
                logits, token_id_seq = mathocr.train(data, tokens);
                # skip the first start token, only use the following ground truth values
                expected = tf.cast(tf.reshape(tokens[:,1:],(-1,tokens_length_max - 1, 1)), dtype = tf.float32);
                # get loss
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)(logits, expected);
            avg_loss.update_state(loss);
            if tf.equal(optimizer.iterations % 100, 0):
                with log.as_default():
                    tf.summary.scalar('loss',avg_loss.result(), step = optimizer.iterations);
                print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
                avg_loss.reset_states();
            grads = tape.gradient(loss, mathocr.trainable_variables);
            optimizer.apply_gradients(zip(grads, mathocr.trainable_variables));
        # save model every epoch
        checkpoint.save(os.path.join('checkpoint','ckpt'));
        if loss < 0.01: break;
    #save the network structure with weights
    mathocr.save('mathocr.h5');

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
