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


def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)


def get_bottleneck_diff(pair):
    pass


def main(args):
    args.same_pairs_file

    args.diff_pairs_file

    X = []  # bottlenecks diff / L2 for each pair
    Y = []  # 1 for same, 0 for diff
    with open('/tmp/same_pairs.txt', 'r') as pairs_file:
        for line in pairs_file.readlines()[1:]:
            pair = line.strip().split()
            X.append(get_bottleneck_diff(pair))
            Y.append(pair[0] == pair[2])
            assert(pair[0] == pair[2])

    with open('/tmp/diff_pairs.txt', 'r') as pairs_file:
        for line in pairs_file.readlines()[1:]:
            pair = line.strip().split()
            X.append(get_bottleneck_diff(pair))
            Y.append(pair[0] == pair[2])

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X, Y)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--same_pairs_file',
        type=str,
        default='/tmp/same_pairs.txt',
        help='Path to the file identifying pairs of photos of the same person from the LFW dataset.'
    )
    parser.add_argument(
        '--diff_pairs_file',
        type=str,
        default='/tmp/diff_pairs.txt',
        help='Path to the file identifying pairs of photos of different persons from the LFW dataset.'
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
