"""Train SVM to distinguish if photos represent the same person with the
bottlenecks extracted from facenet
"""
# MIT License
#
# Copyright (c) 2017 Florent Buisson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import argparse
from sklearn import svm
import numpy as np


# TODO very dirty hash for now :/
def get_embedding(lfw_dir, embeddings, pair0, pair1):
    return embeddings[os.path.join(lfw_dir, pair0, pair0 + '_' + '%04d' % int(pair1) + '.png')]


def get_bottleneck_diff(lfw_dir, embeddings, pair):
    # Get embedding 1
    emb1 = get_embedding(lfw_dir, embeddings, pair[0], pair[1])
    # 2
    emb2 = get_embedding(lfw_dir, embeddings, pair[2], pair[3])

    # TODO if slow, think about doing this array wise with numpy instead of line per line.
    emb_diff = emb1 - emb2
    emb_diff_squared = np.multiply(emb_diff, emb_diff)
    emb_sum = emb1 + emb2
    return np.divide(emb_diff_squared, emb_sum)


def main(args):
    #  Load cached bottlenecks
    #  Create dict from unloaded numpy array ?
    embeddings_np = np.loadtxt("/tmp/embeddings.csv", delimiter=",")
    embeddings = embeddings_np  # TODO dict ?

    # Model variables

    X = []  # bottlenecks diff / L2 for each pair
    Y = []  # 1 for same, 0 for diff

    # Read Same pair file, fetch both bottlenecks and make the diff, add point to np array and corresponding label

    with open(args.same_pairs_file, 'r') as pairs_file:
        for line in pairs_file.readlines()[1:]:
            pair = line.strip().split()
            X.append(get_bottleneck_diff(args.lfw_dir, embeddings, pair))
            Y.append(pair[0] == pair[2])
            assert (pair[0] == pair[2])

    # Do the same with diff pair files

    with open(args.diff_pairs_file, 'r') as pairs_file:
        for line in pairs_file.readlines()[1:]:
            pair = line.strip().split()
            X.append(get_bottleneck_diff(args.lfw_dir, embeddings, pair))
            Y.append(pair[0] == pair[2])

    # Feed the computed points to a linear SVM

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X, Y)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lfw_dir',
        type=str,
        default='/Users/bushi/datasets/face_recognition/lfw/lfw-mtcnnpy_160',
        help='Path to LFW dataset.'
    )
    parser.add_argument(
        '--same_pairs_file',
        type=str,
        default='./same_pairs.txt',
        help='Path to the file identifying pairs of photos of the same person from the LFW dataset.'
    )
    parser.add_argument(
        '--diff_pairs_file',
        type=str,
        default='./diff_pairs.txt',
        help='Path to the file identifying pairs of photos of different persons from the LFW dataset.'
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
