import matplotlib.pyplot as plt

"""
path = 'C:/Users/sbpark/Desktop/real.csv'
line_num = 12

with open(path, 'r') as myfile:
    for i, line in enumerate(myfile):
        if i == line_num:
            print(line)
            series = line.split(',')
            mylist = [float(item) for item in series]
print(mylist)

axes = plt.gca()
# axes.set_ylim([-4, 4])
plt.plot(mylist[:-2], 'r')
plt.show()
"""

def _draw_graph(loss_list, acc_list):
    axes = plt.gca()
    axes.set_ylim([0, max(acc_list)*1.3])
    axes.set_xlim([0, len(loss_list)])
    plt.plot(loss_list, label='loss')
    plt.plot(acc_list, label='accuracy')
    plt.plot([1 for i in range(len(loss_list))], 'r--')
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('epoch')

    plt.title('ReLU')
    plt.show()