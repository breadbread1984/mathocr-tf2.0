#!/usr/bin/bash

import cv2;
import numpy as np;
import tensorflow as tf;
from Model import MathOCR, convert_to_readable;

class Predictor(object):

    def __init__(self, input_shape = (128,128,3), weights_path = 'models/mathocr_30800.h5'):

        self.mathocr = MathOCR(input_shape);
        self.mathocr.load_weights(weights_path);

    def predict(self, img):

        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        # smooth
        gray = cv2.GaussianBlur(gray, (5,5), 0);
        # threshold
        _, binary = cv2.threshold(gray, 127,255, cv2.THRESH_BINARY);
        # erosion
        binary = cv2.dilate(binary, kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3), (1,1)), iterations = 1);
        # convert to tensor
        data = tf.stack([binary,] * 3, axis = 2);
        data = tf.cast(data, dtype = tf.float32);
        # preprocess image
        pos = tf.where(tf.greater(255. - tf.squeeze(data),0.));
        min_yx = tf.reduce_min(pos, axis = 0);
        max_yx = tf.reduce_max(pos, axis = 0);
        hw = max_yx - min_yx;
        data = tf.image.crop_to_bounding_box(data, min_yx[0], min_yx[1], hw[0], hw[1]);
        data = tf.image.resize(data, (128,128));
        # add batch dim
        data = tf.expand_dims(data, 0);
        # feed to predictor
        token_id_sequence, _ = self.mathocr(data);
        s = convert_to_readable(token_id_sequence, self.mathocr.id_to_token);
        return s;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    predictor = Predictor();
    img = cv2.imread('pics/handwritting1.jpg');
    assert img is not None, "can't open image!";
    output = predictor.predict(img);
    print(output);

