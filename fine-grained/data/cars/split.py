import numpy as np
import scipy.io as io

mat = io.loadmat('./cars_annos.mat')
print(type(mat))
print(mat.keys())
print(mat['annotations'][0, 1000])

train = {}
test = {}

x = mat['annotations']
print(x[0, 0][6][0,0])
for i in range(x.shape[1]):
    name = x[0, i][0][0]
    name = name.split('/')[1]
    cls = x[0, i][5][0, 0]
    print(name, cls)
    if x[0, i][6][0, 0] == 0:
        train[name] = cls
    else:
        test[name] = cls

with open('./train_list.txt', 'w') as f:
    for k, v in train.items():
        f.writelines(k + ';' + str(v-1) + '\n')

with open('./test_list.txt', 'w') as f:
    for k, v in test.items():
        f.writelines(k + ';' + str(v-1) + '\n')