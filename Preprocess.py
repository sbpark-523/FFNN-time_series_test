import numpy as np

p_path = "D:/DKE/data/period_classification/주기성_데이터.csv"
np_path = "D:/DKE/data/period_classification/비주기성_데이터.csv"
o_path = "D:/DKE/data/period_classification/outline(non_periodic).csv"
e_path = "D:/DKE/data/period_classification/ECG(periodic).csv"

p_total_list = []
np_total_list = []
def _reader(path, append_list):

    with open(path, 'r') as ifp:
        for i, line in enumerate(ifp):      # 혹시 나눠서 라인만큼 읽게될때 좋음
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

"""
    ??? -> 1024
    if lenght > 1024 ==> PAA (수정 필요!!!)
"""
def _resize(dataset, isPeriod):
    results = []
    input_length = 10
    for time_series in dataset:
        # split X and Y
        x_data = time_series[:-2]
        y_data = time_series[-2:]
        x_len = len(x_data)
        if x_len == input_length:
            print('Fit')
            results.append(time_series)
        elif x_len > input_length:    # PAA
            resized_time_series = []
            width = int(x_len/input_length)
            for i in range(input_length):    # rest data are split!
                item_list = x_data[i*width:(i+1)*width] # sum list
                item = sum(item_list)/float(width)
                resized_time_series.append(item)
            x_data = resized_time_series
            result = x_data + y_data
            results.append(result)
        elif (x_len < input_length) & isPeriod:     # smaller than 1024 and have period
            # concatenate
            print('True!')
            iter_val = int(input_length/x_len)
            mod_size = input_length%x_len
            print(mod_size)
            resized_time_series = [x_data*iter_val + x_data[:mod_size] + y_data]
            results.append(resized_time_series)
        elif (x_len < input_length) & (isPeriod == False):
            # inverse PAA
            print('False')
            # 2의 지수승이여야 할거같은데..?ㅠㅠㅠㅠ어떡하지..잏ㅎㅎㅎㅎ하핳하하하

    return results

# print(not True)
test = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]]
# print(test)
# result = _resize(test)
# print(result)

myresult = _resize(test, True)
print(myresult)
# read
# _reader(p_path, p_total_list)
# _reader(e_path, p_total_list)
# print(p_total_list.__len__())
# _reader(np_path, np_total_list)
# _reader(o_path, np_total_list)
# print(np_total_list.__len__())
#
# # resize
# p_total_list = _resize(p_total_list, true)
# np_total_list = _resize(np_total_list, false)
# print(np.array(p_total_list).shape)
# print(np.array(np_total_list).shape)

