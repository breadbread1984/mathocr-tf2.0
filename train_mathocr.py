#!/usr/bin/python3

import tensorflow as tf;
from Model import Encoder, Decoder;

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
    data = tf.io.decode_raw(context['data'], out_type = tf.uint8);
    data = tf.reshape(data, (256,256,1));
    data = tf.cast(data, dtype = tf.float32);
    text_length = tf.cast(context['text_length'], dtype = tf.int32);
    tokens_length = tf.cast(context['tokens_length'], dtype = tf.int32);
    text = sequence['text'];
    tokens = sequence['tokens'];
    tf.Assert(tf.equal(tf.shape(text)[0],text_length),[tf.shape(text)]);
    tf.Assert(tf.equal(tf.shape(tokens)[0],tokens_length),[tf.shape(tokens)]);
    return data, tokens, text;

def main():
    
    # load dataset
    trainset = tf.data.TFRecordDataset('trainset.tfrecord').map(parse_function).shuffle(100);
    testset = tf.data.TFRecordDataset('testset.tfrecord').map(parse_function);
    while True:
        for data, tokens, _ in trainset:
            # TODO

if __name__ == "__main__":

    main();
