#!/usr/bin/python3

import tensorflow as tf;

def main():

    tf.keras.backend.set_learning_phase(0);
    encoder = tf.keras.models.load_model('encoder.h5', custom_objects={'tf': tf});
    decoder = tf.keras.models.load_model('decoder.h5', custom_objects={'tf': tf});
    tf.saved_model.save(encoder, './encoder/1');
    tf.saved_model.save(decoder, './decoder/1');
    encoder_loaded = tf.saved_model.load('./encoder/1');
    decoder_loaded = tf.saved_model.load('./decoder/1');
    print(encoder_loaded)
    print(decoder_loaded)
    encoder_infer = encoder_loaded.signatures['serving_default'];
    decoder_infer = decoder_loaded.signatures['serving_default'];
    print('====================NOTE==================');
    print('encoder\'s output tensor name is', encoder_infer.structured_outputs);
    print('decoder\'s output tensor name is', decoder_infer.structured_outputs);

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();

