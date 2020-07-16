import inspect
import warnings
import functools

from tensorboardX import SummaryWriter, FileWriter, RecordWriter


def trycatch_method(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            warnings.warn("An error occurred while executing '{}':\n{}"
                          "".format(func.__name__, e), RuntimeWarning)
    return wrapper


def decorate_all_methods(decorator, filter_name=None):
    def apply_decorator(cls):
        for k, f in cls.__dict__.items():
            if inspect.isfunction(f) and (filter_name is None or filter_name(f.__name__)):
                setattr(cls, k, decorator(f))
        return cls
    return apply_decorator


def is_add_method(name):
    return name.startswith("add_")


_decorator = decorate_all_methods(trycatch_method, is_add_method)
SummaryWriter = _decorator(SummaryWriter)
RecordWriter = _decorator(RecordWriter)
FileWriter = _decorator(FileWriter)
