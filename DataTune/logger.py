# -*- coding:utf-8 -*-

from __future__ import print_function
import time, os

def get_current_time() :
    current_time = time.time()
    current_ms = str(':') + str(round(current_time * 1000000))[-6:]
    current_time = time.strftime("%Y/%m/%d | %H:%M:%S", time.localtime()) + current_ms
    return current_time

def get_process() :
    return str(os.getpid())

def info(*args) :
    print('[ {} | {} ]'.format(get_current_time(), get_process()), end=' ')
    print(*args[:-1], end=' ')
    print(args[-1], end='\n')
    return None

if __name__ == '__main__' :
    info('It works?', 'Yes!')