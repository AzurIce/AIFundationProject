import subprocess

# def train(path):
#     res = subprocess.run(['python', path], capture_output=True)
#     print(res)
#
# train('../models/LeNet.py')


import os, sys
from contextlib import contextmanager


@contextmanager
def redirect_stdout(new_stdout):
    saved_stdout, sys.stdout = sys.stdout, new_stdout
    try:
        yield
    finally:
        sys.stdout = saved_stdout
