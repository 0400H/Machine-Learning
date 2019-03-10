# -*- coding: UTF-8 -*-

class estimate_string (object) :
    def __init__(self):
        return None

    def get_data_from_log(self, filename = '', keyword = ''):
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
        return data_list, length

    def get_mean_from_data(self, data_list = [], length = 0):
        if len(data_list) < length :
            print('len of data_list is loss than length')
            return -1
        else :
            return sum(data_list[:length]) / length

    def get_max_from_data(self, data_list = [], length = 0):
        if len(data_list) < length :
            print('len of data_list is loss than length')
            return -1
        else :
            return max(data_list[:length])

    def get_min_from_data(self, data_list = [], length = 0):
        if len(data_list) < length :
            print('len of data_list is loss than length')
            return -1
        else :
            return min(data_list[:length])

    def get_sq_from_data(self, data_list = [], length = 0):
        if len(data_list) < length :
            print('len of data_list is loss than length')
            return -1
        else :
            mean_time = self.get_mean_from_data(data_list, length)
            loss_list = [time - mean_time for time in data_list]
            sq_loss = 0
            for loss in loss_list[:length]:
                sq_loss += loss ** 2
            sq_loss = sq_loss / length
            return sq_loss

    def get_std_from_data(self, data_list = [], length = 0):
        if len(data_list) < length :
            print('len of data_list is loss than length')
            return -1
        else :
            sq_loss = self.get_sq_from_data(data_list, length)
            std_loss = sq_loss ** 0.5
            return std_loss

pass



if __name__ == '__main__':
    estimate_class = estimate_string()
    estimate_data, length = [x for x in range(100)], 100
    # estimate_data, length = estimate_class.get_data_from_log('proto.log', 'forward-backward time:')

    mean_time = estimate_class.get_mean_from_data(estimate_data, length) # get mean time
    max_time = estimate_class.get_max_from_data(estimate_data, length) # get max time
    min_time = estimate_class.get_min_from_data(estimate_data, length) # get min time
    std_time = estimate_class.get_std_from_data(estimate_data, length) # get standard loss time
    sq_time = estimate_class.get_sq_from_data(estimate_data, length) # get square loss time

    print('mean_time: %.3f, max_time: %.3f, min_time: %.3f, std_time: %.6f, sq_time: %.6f'
          % (mean_time, max_time, min_time, std_time, sq_time))