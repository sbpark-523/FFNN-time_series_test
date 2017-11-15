import numpy as np

# p_path = "D:/DKE/data/period_classification/주기성_데이터.csv"
# np_path = "D:/DKE/data/period_classification/비주기성_데이터.csv"
# o_path = "D:/DKE/data/period_classification/outline(non_periodic).csv"
# e_path = "D:/DKE/data/period_classification/ECG(periodic).csv"
#
# p_total_list = []
# np_total_list = []


def _reader(path, append_list):
    with open(path, 'r') as ifp:
        for i, line in enumerate(ifp):  # 혹시 나눠서 라인만큼 읽게될때 좋음
            # if i >= 0 & i < 3:
            s_time_series = line.split(',')
            n_time_series = []
            for item in s_time_series:
                if item.__len__() > 0:  # last item
                    n_time_series.append(float(item))
                else:
                    break
            data = n_time_series.copy()
            append_list.append(data)

    return append_list


"""
    ??? -> 1024
    if lenght > 1024 ==> PAA (수정 필요!!!)
"""


def _resize(dataset):
    results = []
    input_length = 1024
    for time_series in dataset:
        # split X and Y
        x_data = time_series[:-2]
        y_data = time_series[-2:]
        x_len = len(x_data)
        if x_len == input_length:
            # print('Fit')
            results.append(time_series)
        elif x_len > input_length:  # PAA
            resized_time_series = []
            width = int(x_len / input_length)
            rest = x_len % input_length
            for i in range(input_length - 1):  # reshaped length -1
                item_list = x_data[i * width:(i + 1) * width]  # sum list
                item = sum(item_list) / float(width)
                resized_time_series.append(item)
            if rest == 0:
                final_list = x_data[-width:]
                final_item = sum(final_list) / width
                resized_time_series.append(final_item)
            else:
                final_list = x_data[-rest:]
                final_item = sum(final_list) / rest
                resized_time_series.append(final_item)
            x_data = resized_time_series
            result = x_data + y_data
            results.append(result)
        elif (x_len < input_length) & (y_data[0] == 1):  # smaller than 1024 and have period
            # concatenate
            # print('True!')
            iter_val = int(input_length / x_len)
            mod_size = input_length % x_len
            # print(mod_size)
            resized_time_series = x_data * iter_val + x_data[:mod_size] + y_data
            results.append(resized_time_series)
        elif (x_len < input_length) & (y_data[0] == 0):
            # inverse PAA
            resized_time_series = []
            # print('False')
            base_width = int(input_length / x_len)
            rest = input_length % x_len
            for i in range(x_len):
                # print('input - rest: ', x_len - rest)
                # print('base width: ', base_width)
                if i < (x_len - rest):
                    # print('base: ',x_data[i])
                    resized_time_series += [x_data[i]] * base_width
                    # print(resized_time_series)
                else:
                    # print('base + 1: ',x_data[i])
                    resized_time_series += [x_data[i]] * (base_width + 1)
            resized_time_series += y_data
            results.append(resized_time_series)

    return results


def _lencheck(target_list):
    for i, time_series in enumerate(target_list):
        if len(time_series) != 1026:
            print(i)


def _shuffleNdivide(p_data, np_data):
    """ p_data, np_data: list"""
    p_data = np.array(p_data)
    np_data = np.array(np_data)

    np.random.shuffle(p_data)
    np.random.shuffle(np_data)

    p_length = len(p_data)
    np_length = len(np_data)

    # divide Train, Validation, Test
    p_training = p_data[0: int(p_length*0.6)].copy()
    p_validation = p_data[int(p_length*0.6): int(p_length*0.8)].copy()
    p_test = p_data[int(p_length*0.8):].copy()

    np_training = np_data[0: int(np_length*0.6)].copy()
    np_validation = np_data[int(np_length*0.6): int(np_length*0.8)].copy()
    np_test = np_data[int(np_length*0.8):].copy()

    print(p_training.shape, np_training.shape)
    training = np.concatenate((p_training, np_training), axis=0)
    validation = np.concatenate((p_validation, np_validation), axis=0)
    test = np.concatenate((p_test, np_test), axis=0)

    np.random.shuffle(training)
    np.random.shuffle(validation)
    np.random.shuffle(test)

    return training[:, :1024], training[:, -2:], validation[:, :1024], validation[:, -2:], test[:, :1024], test[:, -2:]

# read
# _reader(p_path, p_total_list)
# _reader(e_path, p_total_list)
# print(p_total_list.__len__())
# _reader(np_path, np_total_list)
# _reader(o_path, np_total_list)
# print(np_total_list.__len__())
# # # resize
# p_total_list2 = _resize(p_total_list)
# print('---------P len check--------')
# _lencheck(p_total_list2)
# np_total_list2 = _resize(np_total_list)
# print('---------NP len check--------')
# _lencheck(np_total_list2)
# print(np.array(p_total_list2).shape)
# print(np.array(np_total_list2).shape)
# # print('what?')
# _shuffleNdivide(p_total_list2, np_total_list2)