#!/usr/bin/python3

import csv;
import os;
import cv2;
import tensorflow as tf;
import pickle;

class CrohmeDataset(object):
    
    START = "<SOS>";
    END = "<EOS>";
    PAD = "<PAD>";
    SPECIAL_TOKENS = [START, END, PAD];
    
    def __init__(self, groundtruth, tokens_file, root):
        
        # groundtruth give the corresponce between image path and latex in text.
        # tokens file give the corresponce between latex alphabets to integer.
        # root is where the images are stored.
        self.groundtruth = groundtruth;
        self.token_to_id, self.id_to_token = self.load_vocab(tokens_file);
        self.root = root;
        with open('token_id_map.dat','wb') as f:
            f.write(pickle.dumps(self.token_to_id));
            f.write(pickle.dumps(self.id_to_token));
        
    # There are so many symbols (mostly escape sequences) that are in the test sets but not
    # in the training set.
    def remove_unknown_tokens(self, truth):
        
        # Remove \mathrm and \vtop are only present in the test sets, but not in the
        # training set. They are purely for formatting anyway.
        remaining_truth = truth.replace("\\mathrm", "");
        remaining_truth = remaining_truth.replace("\\vtop", "");
        # \; \! are spaces and only present in 2014's test set
        remaining_truth = remaining_truth.replace("\\;", " ");
        remaining_truth = remaining_truth.replace("\\!", " ");
        remaining_truth = remaining_truth.replace("\\ ", " ");
        # There's one occurrence of \dots in the 2013 test set, but it wasn't present in the
        # training set. It's either \ldots or \cdots in math mode, which are essentially
        # equivalent.
        remaining_truth = remaining_truth.replace("\\dots", "\\ldots");
        # Again, \lbrack and \rbrack where not present in the training set, but they render
        # similar to \left[ and \right] respectively.
        remaining_truth = remaining_truth.replace("\\lbrack", "\\left[");
        remaining_truth = remaining_truth.replace("\\rbrack", "\\right]");
        # Same story, where \mbox = \leavemode\hbox
        remaining_truth = remaining_truth.replace("\\hbox", "\\mbox");
        # There is no reason to use \lt or \gt instead of < and > in math mode. But the
        # training set does. They are not even LaTeX control sequences but are used in
        # MathJax (to prevent code injection).
        remaining_truth = remaining_truth.replace("<", "\\lt");
        remaining_truth = remaining_truth.replace(">", "\\gt");
        # \parallel renders to two vertical bars
        remaining_truth = remaining_truth.replace("\\parallel", "||");
        # Some capital letters are not in the training set...
        remaining_truth = remaining_truth.replace("O", "o");
        remaining_truth = remaining_truth.replace("W", "w");
        remaining_truth = remaining_truth.replace("\\Pi", "\\pi");
        return remaining_truth;

    # Rather ignorant way to encode the truth, but at least it works.
    def encode_truth(self, truth, token_to_id):
        
        truth_tokens = [];
        remaining_truth = self.remove_unknown_tokens(truth).strip();
        while len(remaining_truth) > 0:
            try:
                matching_starts = [
                    [i, len(tok)]
                    for tok, i in token_to_id.items()
                    if remaining_truth.startswith(tok)
                ];
                # Take the longest match
                index, tok_len = max(matching_starts, key=lambda match: match[1]);
                truth_tokens.append(index);
                remaining_truth = remaining_truth[tok_len:].lstrip();
            except ValueError:
                raise Exception("Truth contains unknown token");
        return truth_tokens;

    def load_vocab(self, tokens_file):
        
        with open(tokens_file, "r") as fd:
            reader = csv.reader(fd, delimiter="\t");
            tokens = next(reader);
            tokens.extend(self.SPECIAL_TOKENS);
            token_to_id = {tok: i for i, tok in enumerate(tokens)};
            id_to_token = {i: tok for i, tok in enumerate(tokens)};
            return token_to_id, id_to_token;

    def generate_tfrecord(self, output = 'trainset.tfrecord'):

        writer = tf.io.TFRecordWriter(output);
        with open(self.groundtruth, "r") as fd:
            reader = csv.reader(fd, delimiter = '\t');
            for p, truth in reader:
                img = cv2.imread(os.path.join(self.root, p + ".png"), cv2.IMREAD_GRAYSCALE);
                if img is None:
                    print('can\'t open image ' + os.path.join(self.root, p + ".png"));
                    continue;
                text = truth.encode('ascii');
                tokens = [self.token_to_id[self.START], 
                          *self.encode_truth(truth, self.token_to_id),
                          self.token_to_id[self.END]];
                trainsample = tf.train.SequenceExample(
                    context = tf.train.Features(feature = {
                        'data': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img.tobytes()])),
                        'text_length': tf.train.Feature(int64_list = tf.train.Int64List(value = [len(text)])),
                        'tokens_length': tf.train.Feature(int64_list = tf.train.Int64List(value = [len(tokens)]))
                    }),
                    feature_lists = tf.train.FeatureLists(feature_list={
                        'text': tf.train.FeatureList(feature = [tf.train.Feature(int64_list = tf.train.Int64List(value = [alphabet])) for alphabet in text]),
                        'tokens': tf.train.FeatureList(feature = [tf.train.Feature(int64_list = tf.train.Int64List(value = [token])) for token in tokens])
                    })
                );
                writer.write(trainsample.SerializeToString());
        writer.close();

if __name__ == "__main__":
    
    dataset = CrohmeDataset(
        groundtruth = './CROHME-png/groundtruth_train.tsv',
        tokens_file = './CROHME-png/tokens.tsv',
        root = './CROHME-png/train'
    );
    dataset.generate_tfrecord('trainset.tfrecord');
    dataset = CrohmeDataset(
        groundtruth = './CROHME-png/groundtruth_2016.tsv',
        tokens_file = './CROHME-png/tokens.tsv',
        root = './CROHME-png/test/2016'
    );
    dataset.generate_tfrecord('testset.tfrecord');
