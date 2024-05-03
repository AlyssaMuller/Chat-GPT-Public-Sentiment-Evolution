#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import pandas as pd
import sklearn.model_selection
import csv


def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(
        root, ext, dirname, basename))
    return basename


def get_data(filename):
    """
    Assumes column 0 is the instance index stored in the
    csv file.  If no such column exists, remove the
    index_col=0 parameter.
    """
    data = pd.read_csv(filename, index_col=0)
    return data


def filter_data(filename):
    file = open(filename, "r")
    og = csv.reader(file, delimiter=',')

    newFilename = filename[:-4]+"_filtered.csv"
    newFile = open(newFilename, "w")
    writer = csv.writer(newFile)

    filtered_rows = [item for item in og if item[4] == "en" and item[3]
                     [0] != "🔥" and (item[3][0] != "R" and item[3][1] != "T")]
    for r in filtered_rows:
        writer.writerow(['', r[3]])
    return newFilename


def split_data(data, train_name, test_name, ratio, seed):
    data_train, data_test = sklearn.model_selection.train_test_split(
        data, test_size=ratio, random_state=seed)
    data_train.to_csv(train_name)
    data_test.to_csv(test_name)
    return


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0], description='Split Data into Training/Testing Sets')
    parser.add_argument('action', default='all',
                        choices=["split", "all"],
                        nargs='?', help="desired action")
    parser.add_argument('--data-file',     '-d', default="",
                        type=str,   help="csv file of data to split")
    #15% test
    parser.add_argument('--test-ratio',    '-r', default=0.15,
                        type=float, help="fraction of data to use as test data")
    parser.add_argument('--train-file',    '-t', default="",    type=str,
                        help="name of file to save training data (default is constructed from input file name)")
    parser.add_argument('--test-file',     '-T', default="",    type=str,
                        help="name of file to save test data (default is constructed from input file name)")
    parser.add_argument('--random-seed',             '-R', default=314159265,
                        type=int, help="random number seed (-1 to use OS entropy)")
    parser.add_argument('--filter',    '-f', default=False,
                        type=bool, help="change if data must be filtered")

    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #

    return my_args


def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=logging.WARN)

    if my_args.random_seed == -1:
        seed = None
    else:
        seed = my_args.random_seed

    if my_args.filter:
        filter_data(my_args.data_file)
    filename = my_args.data_file
    train_file = my_args.train_file
    test_file = my_args.test_file
    if os.path.exists(filename) and os.path.isfile(filename):

        basename = get_basename(filename)
        if train_file and os.path.exists(train_file):
            raise Exception(
                "training data file: {} already exists.".format(train_file))
        if test_file and os.path.exists(test_file):
            raise Exception(
                "testing data file: {} already exists.".format(test_file))

        if not train_file:
            train_file = "{}-train.csv".format(basename)
        if not test_file:
            test_file = "{}-test.csv".format(basename)

        data = get_data(filename)
        split_data(data, train_file, test_file, my_args.test_ratio, seed)
    else:
        print("{} doesn't exist, or is not a normal file.".format(filename))

    return


if __name__ == "__main__":
    main(sys.argv)
