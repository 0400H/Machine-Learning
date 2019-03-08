# -*- coding: UTF-8 -*-

import numpy as np

class estimate_numpy (object) :
    def __init__(self):
        return None

    def get_array_from_log(self, filename = '', keyword = ''):
        data_list = []
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                if ((keyword != '') and (line.find(keyword) != -1)):
                    string_index = line.index(keyword) + len(keyword)
                    string = str(line[string_index: ]).strip('\n').strip('ms.')
                    data_list.append(float(string))
                else :
                    continue
            length = len(data_list)
            data_array = np.array(data_list)
        return data_array, length

    def get_mean_from_data(self, data_array, length = 0):
        if len(data_array) < length :
            print('len of data_array is loss than length')
            return -1
        else :
            return np.sum(data_array[:length]) / length

    def get_max_from_data(self, data_array, length = 0):
        if len(data_array) < length :
            print('len of data_array is loss than length')
            return -1
        else :
            return np.max(data_array[:length])

    def get_min_from_data(self, data_array, length = 0):
        if len(data_array) < length :
            print('len of data_array is loss than length')
            return -1
        else :
            return np.min(data_array[:length])

    def get_sq_from_data(self, data_array, length = 0):
        if len(data_array) < length :
            print('len of data_array is loss than length')
            return -1
        else :
            return np.var(data_array[:length])

    def get_std_from_data(self, data_array, length = 0):
        if len(data_array) < length :
            print('len of data_array is loss than length')
            return -1
        else :
            return np.std(data_array[:length])

pass



if __name__ == '__main__':
    estimate_class = estimate_numpy()
    estimate_data, length = [x for x in range(100)], 100
    # estimate_data, length = estimate_class.get_array_from_log('proto.log', 'forward-backward time:')

    mean_time = estimate_class.get_mean_from_data(estimate_data, length) # get mean time
    max_time = estimate_class.get_max_from_data(estimate_data, length) # get max time
    min_time = estimate_class.get_min_from_data(estimate_data, length) # get min time
    std_time = estimate_class.get_std_from_data(estimate_data, length) # get standard loss time
    sq_time = estimate_class.get_sq_from_data(estimate_data, length) # get square loss time

    print('mean_time: %.3f, max_time: %.3f, min_time: %.3f, std_time: %.6f, sq_time: %.6f'
          % (mean_time, max_time, min_time, std_time, sq_time))