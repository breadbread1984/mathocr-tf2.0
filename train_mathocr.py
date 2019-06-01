#!/usr/bin/python3

import pickle;
import tensorflow as tf;
from Model import Encoder, Decoder;

batch_num = 100;
tokens_length_max = 90;
START = "<SOS>";
END = "<EOS>";
PAD = "<PAD>";
SPECIAL_TOKENS = [START, END, PAD];

def parse_function(serialized_example, crop = True, transform = True):
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
    tokens = tf.pad(tokens, paddings = [[0,tokens_length_max - tf.shape(tokens)[0]]], constant_values = -1);
        
    return data, tokens;

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
    trainset = tf.data.TFRecordDataset('trainset.tfrecord').map(parse_function).shuffle(100).batch(batch_num);
    testset = tf.data.TFRecordDataset('testset.tfrecord').map(parse_function).batch(batch_num);
    while True:
        for data, tokens in trainset:
            with tf.GradientTape() as tape:
                low_res, high_res = encoder(data);
                sequence = tf.ones((batch_num,1)) * token_to_id[START];
                hidden = None;
                
                out, hidden = decoder()
            
if __name__ == "__main__":

    main();
