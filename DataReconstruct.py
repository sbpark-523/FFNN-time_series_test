import numpy as np
import csv
import random
import matplotlib.pyplot as plt

from html5lib.serializer import serialize

"""
 데이터를 읽는다(rad: 10, 30)
 변형시킨다
   1. 512씩 자른다
   2. 256씩 자른다(나머지 zero padding)
   3. 128씩 자른다(나머지 zero padding)
   
   각각 1, 2, 4배 한다
"""

"""파일을 읽고, """
def reader(path):
    time_series = []
    f = open(path, 'r')
    csvReader = csv.reader(f)
    cnt = 0
    for row in csvReader:
        XandY = np.array(row).astype(np.double)
        time_series.append(XandY)
        # if cnt == 0 :
        #     break;
    random.shuffle(time_series)     # Random 하게 섞고 나눠서 return
    time_series = np.round(time_series, 8)

    print(time_series.shape)
    return time_series #time_series[:, :256], time_series[:, -2:]



"""
    p: Periodic
    np: Non Periodic
"""
def setData2(p, nps):
    p_length = len(p)
    np_length = len(nps)

    p_training = p[0 : int(p_length*0.6)].copy()
    p_validation = p[int(p_length*0.6) : int(p_length*0.8)].copy()
    p_test = p[int(p_length*0.8) :].copy()

    np_training = nps[0 : int(np_length*0.6)].copy()
    np_validation = nps[int(np_length*0.6) : int(np_length*0.8)].copy()
    np_test = nps[int(np_length*0.8) :].copy()

    print(p_training.shape, np_training.shape)
    training = np.concatenate((p_training, np_training), axis=0)
    validation = np.concatenate((p_validation, np_validation), axis=0)
    test = np.concatenate((p_test, np_test), axis=0)

    np.random.shuffle(training)
    np.random.shuffle(validation)
    np.random.shuffle(test)

    return training[:, :256], training[:, -2:], validation[:, :256], validation[:, -2:], test[:, :256], test[:, -2:]
    # return training[:, :256], training[:, -2:], validation[:, :256], validation[:, -2:], test[:, :256], test[:, -2:]



def reshape(series_nparray):
    windows= [128]
    disjoint_series = []
    """128씩 자르고 - disjoint"""
    windows128 = series_nparray.shape[1] / 128
    windows128 = round(windows128 + 0.5)        # 올림

    for winSize in windows:
        for i in range(windows128):
            start = i * winSize
            end = start + winSize
            disjoint_series.append(series_nparray[0][start:end])
            print("start: ", start, ", end: ", end, "size of: ", len(disjoint_series))
    return 0

def reshape1(series_array):
    winSize = 256

    windows256 = series_array.shape[1] / winSize
    windows256 = round(windows256 + 0.5)
    disjoint_series = []
    y_data = []

    # windows256 = 2
    # series_array = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    for i in range(windows256):
        origin_noise = []; origin_noise2 = []           # 원본 리스트
        mul2 = []                   # 2배 리스트
        mul2_noise = []; mul2_noise2 = []                 # 2배 + 노이즈
        mul4 = []                                       # 4배 리스트
        mul4_noise = []; mul4_noise2 = []             # 4배 + 노이즈
        start = i * winSize
        end = start + winSize

        split_origin = series_array[0][start:end]       #original data <-일단 지금만 한줄!!
        o_copy = split_origin.copy()

        for item in o_copy:         # 각 데이터를 2배, 4배
            origin_noise.append(item + round(random.gauss(0.0, 0.5), 8))
            origin_noise2.append(item + round(random.gauss(0.0, 0.1), 8))
            mul2.append(item*2)
            mul2_noise.append(item*2 + round(random.gauss(0.0, 0.5), 8))
            mul2_noise2.append(item*2 + round(random.gauss(0.0, 0.1), 8))
            mul4.append(item*4)
            mul4_noise.append(item*4 + round(random.gauss(0.0, 0.5), 8))
            mul4_noise2.append(item*4 + round(random.gauss(0.0, 0.05), 8))

        # add
        disjoint_series.append(split_origin.tolist());          y_data.append([1, 0])
        disjoint_series.append(origin_noise);                   y_data.append([1, 0])
        disjoint_series.append(origin_noise2);                   y_data.append([1, 0])
        disjoint_series.append(mul2);                           y_data.append([1, 0])
        disjoint_series.append(mul2_noise);                     y_data.append([1, 0])
        disjoint_series.append(mul2_noise2);                     y_data.append([1, 0])
        disjoint_series.append(mul4);                           y_data.append([1, 0])
        disjoint_series.append(mul4_noise);                     y_data.append([1, 0])
        disjoint_series.append(mul4_noise2);                     y_data.append([1, 0])

    x_data = np.array(disjoint_series)
    y_data = np.array(y_data)
    return x_data, y_data

#
# X, Y = reader()
# print('result OK')
# print(X.shape)
# print(Y.shape)


# x_origin = reader()
# X_data, Y_data = reshape1(x_origin)
#
# # print('size: ',X_data.__len__())
# # X_array = reader()
# # X_data = np.array(X_array)
# axes = plt.gca()
# axes.set_ylim([-4, 4])
# plt.plot(X_data[0][:-2])
# plt.show()

# print(X_data.shape)
# print(Y_data.shape)