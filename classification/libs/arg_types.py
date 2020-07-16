import argparse
import re
from collections import OrderedDict


def arg_boolean(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def arg_tuple(cast_type):
    regex = re.compile(r'\d+\.\d+|\d+')

    def parse_tuple(v):
        vals = regex.findall(v)
        return [cast_type(val) for val in vals]

    return parse_tuple


def arg_list_tuple(cast_type):
    regex = re.compile(r'\([^\)]*\)')
    tuple_parser = arg_tuple(cast_type)

    def parse_list(v):
        tuples = regex.findall(v)
        return [tuple_parser(t) for t in tuples]

    return parse_list


def arg_dict(cast_type, key_type=str):
    regex_pairs = re.compile(r'[^\ ]+=[^\ ]+')
    regex_keyvals = re.compile(r'([^\ ]+)=([^\ ]+)')

    def parse_dict(v):
        d = OrderedDict()
        for keyval in regex_pairs.findall(v):
            key, val = regex_keyvals.match(keyval).groups()
            d.update({key_type(key): cast_type(val)})
        return d

    return parse_dict


def arg_kwargs():
    # kw1name:kw1type=kw1val kw2name:kw2type=kw2val ...
    # eg "step:int=5 gamma:float:0.1" -> {'step': 4, 'gamma': 0.1}
    regex_pairs = re.compile(r'[^\ ]+:[^\ ]+=[^\ ]+')
    regex_vals = re.compile(r'([^\ ]+):([^\ ]+)=([^\ ]+)')

    def parse_kwarg(v):
        d = {}
        for pair in regex_pairs.findall(v):
            key, type, val = regex_vals.match(pair).groups()
            type = eval(type)
            d.update({key: type(val)})
        return d

    return parse_kwarg
