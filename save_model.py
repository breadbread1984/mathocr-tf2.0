#!/usr/bin/python3

import tensorflow as tf;
from Model import MathOCR;

if __name__ == "__main__":

    mathocr = MathOCR();
    optimizer = tf.keras.optimizers.Adam(1e-3, decay = 1e-4);
    checkpoint = tf.train.Checkpoint(model = mathocr, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoint'));
    mathocr.save_weights('mathocr.h5');
