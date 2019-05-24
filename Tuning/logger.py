# -*- coding:utf-8 -*-

from __future__ import print_function
import time, os, traceback

def get_time():
    current_time = time.time()
    current_ms = str(':') + str(round(current_time * 1000000))[-6:]
    current_time = time.strftime("%Y/%m/%d | %H:%M:%S", time.localtime()) + current_ms
    return current_time

def get_pid():
    return str(os.getpid())

def info(*args):
    print('[ {} | {} ]'.format(get_time(), get_pid()), *args)
    return None

def catch_error(function):
    def wrapper(*argc, **argv):
        try:
            function(*argc, **argv)
        except:
            error_msg = traceback.format_exc()
            info(error_msg)
    return wrapper

if __name__ == '__main__' :
    info('It works?', 'Yes!')