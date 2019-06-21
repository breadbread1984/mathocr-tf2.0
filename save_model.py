#!/usr/bin/python3

import tensorflow as tf;
from Model import MathOCR;

if __name__ == "__main__":

    mathocr = MathOCR();
    optimizer = tf.keras.optimizers.Adam(1e-3);
    checkpoint = tf.train.Checkpoint(model = mathocr, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoint'));
    mathocr.encoder.save('encoder.h5');
    mathocr.decoder.save('decoder.h5');
    mathocr.save_weights('mathocr.h5');
