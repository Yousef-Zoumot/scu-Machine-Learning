#Import mathematical functions
import numpy as np

import requests

link = "http://www.cse.scu.edu/~yfang/coen140/iris.data"
f = requests.get(link)

print(f.text)

data = f.text

print(data)

data1 = data.split("\n")

print(data1[149])
#
# iris-setosa-training = data1[0:39]
# iris-setosa-test = data1[40:49]
#
# iris-versicolor-training = data1[50:89]
# iris-versicolor-test = data1[90:99]
#
# iris-virginica-training = data1[100:139]
# iris-virginica-test = data1[140:149]

for i in range(0, 150):
    data1[i] = data1[i].split(",")

print(data1[149])
