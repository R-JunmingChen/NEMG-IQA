import argparse
import json
import logging
import sys
import os
import math
import shutil
import errno


def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]

    return inner


@singleton
class Context(object):
    def __init__(self):
        self.config = None
        self.logger = None
        self.DEBUG = False

    def init_by_args(self, args):
        # read config file
        self.config = json.load(open(args.config))

        self.DEBUG = args.debug

        # mix with the config file and input args
        ## save
        if args.save_model != "":
            self.config["project"]["save_model"] = args.save_model
        ## load
        if args.load_model != "":
            self.config["project"]["load_model"] = args.load_model
        else:
            self.config["project"]["load_model"] = None

            ## image dir and score file
        if args.image_dir != "":
            self.config["dataset"]["image_dir"] = args.image_dir
        if args.score_file != "":
            self.config["dataset"]["score_file"] = args.score_file

        ## log_file
        if args.log_file != "":
            self.config["project"]["log_file"] = args.log_file

        # create logger
        def _logger(logger_name, log_file_path):
            logger = logging.getLogger(logger_name)
            logger.setLevel(level=logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            handler = logging.FileHandler(log_file_path)
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)

            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)

            logger.addHandler(handler)
            logger.addHandler(console)
            return logger

        self.logger = _logger("IQA", self.config["project"]["log_file"])

    def get_config(self):
        assert self.config is not None, "Can't get config before init by input args"
        return self.config

    def get_logger(self):
        assert self.logger is not None, "Can't get logger before init by input args"
        return self.logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='', type=str, help='Path to input images')
    parser.add_argument('--score_file', default='', type=str, help='Path to input subjective score')
    parser.add_argument('--log_file', default='./save/log.txt', type=str, help='Path to log file')
    parser.add_argument('--config', default='./configuration/csiq_config.json', type=str,
                        help='Path to load config file')
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument('--load_model', default='', type=str,
                        help='Path to load model, if empty, the model would be trained without load previous model')
    parser.add_argument('--save_model', default='', type=str, help='Path to save model')

    args = parser.parse_args()
    return args



def flat(_list):
    def flat_element(l):
        for k in l:
            if not isinstance(k, (list, tuple)):
                yield k
            else:
                yield from flat_element(k)

    return list(flat_element(_list))


def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    """
    if not os.path.exists(path):
        create_dir(path)

    if empty:
        empty_dir(path)


def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def empty_dir(path):
    """
    Delete all files and folders in a directory
    :param path: string, path to directory
    :return: nothing
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Warning: {}'.format(e))


def create_dir(path):
    """
    Creates a directory
    :param path: string
    :return: nothing
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
