"""Generates pairs file for the LFW dataset
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
import itertools
import random


def listdir_nohidden_gen(path):
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)) and not f.startswith('.'):
            yield os.path.join(path, f)


def pop_random(lst):
    idx = random.randrange(0, len(lst))
    return idx, lst.pop(idx)


def main(args):
    i = 0

    with open('/tmp/diff_pairs.txt', 'w') as pairs_file:
        for dummy in range(0, 40):
            dirs = list(listdir_nohidden_gen(args.lfw_dir))
            while len(dirs) > 0:
                dirs_copy = dirs.copy()
                _, dir1 = pop_random(dirs)
                files1 = os.listdir(dir1)
                while len(files1) > 0 and len(dirs_copy) > 0:
                    file1, _ = pop_random(files1)
                    _, dir2 = pop_random(dirs_copy)
                    files2 = os.listdir(dir2)
                    file2, _ = pop_random(files2)
                    pairs_file.write(
                        os.path.basename(dir1) + " " + str(file1 + 1) + " " + os.path.basename(dir2) + " " + str(
                            file2 + 1) + "\n")
                    i += 1
                    if i % 10000 == 0:
                        print(i)

    print("Producing all pairs for identical peolpe.")
    dirs = list(listdir_nohidden_gen(args.lfw_dir))

    with open('/tmp/same_pairs.txt', 'w') as pairs_file:
        for dir in dirs:
            files_range = range(len(os.listdir(dir)))
            for file1, file2 in itertools.product(files_range, files_range):
                if file1 != file2:
                    pairs_file.write(
                        os.path.basename(dir) + " " + str(file1 + 1) + " " + os.path.basename(dir) + " " + str(
                            file2 + 1) + "\n")
                    i += 1
                    if i % 10000 == 0:
                        print(i)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lfw_dir',
        type=str,
        default='/Users/bushi/datasets/face_recognition/lfw/lfw-mtcnnpy_160',
        help='Path to LFW dataset.'
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
