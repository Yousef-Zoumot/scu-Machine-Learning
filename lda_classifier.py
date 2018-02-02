#Import mathematical functions
import numpy as np

#open text file with data points
text_file = open("dataset_iris.txt", "r")
#split each line
lines = text_file.read().split("\n")
text_file.close()

print(lines)
print(len(lines)) #151
total_data_instances = len(lines) - 1 #150
print(total_data_instances)

#[[0 for x in range(columns)] for y in range(rows)]
dataset_iris = [[float(0) for x in range(5)] for y in range(150)]
for i in range(total_data_instances):
    #split data points of each instance
    #[sepal length, sepal width, pedal length, pedal width, class]
    dataset_iris[i] = lines[i].split(",")
print(dataset_iris[1][0])

#separate data into 80% training and 20% testing
iris_setosa_training = [[float(0) for x in range(5)] for y in range(40)]
iris_setosa_test = [[float(0) for x in range(5)] for y in range(10)]

iris_versicolor_training = [[float(0) for x in range(5)] for y in range(40)]
iris_versicolor_test = [[float(0) for x in range(5)] for y in range(10)]

iris_virginica_training = [[float(0) for x in range(5)] for y in range(40)]
iris_virginica_test = [[float(0) for x in range(5)] for y in range(10)]

setosa_mean = [[float(0) for x in range(1)] for y in range(4)]
versicolor_mean = [[float(0) for x in range(1)] for y in range(4)]
virginica_mean = [[float(0) for x in range(1)] for y in range(4)]

setosa_covariance = [[float(0) for x in range(4)] for y in range(4)]
versicolor_covariance = [[float(0) for x in range(4)] for y in range(4)]
virginica_covariance = [[float(0) for x in range(4)] for y in range(4)]

temp_matrix4by1 = [[float(0) for x in range(1)] for y in range(4)]
temp_matrix1by4 = [[float(0) for x in range(4)] for y in range(1)]

# temp_matrix4by1 = [[1],[1],[1],[1]]
# temp_matrix1by4 = [[1, 2, 3, 4]]
# print(np.dot(temp_matrix4by1, temp_matrix1by4))

# setosa_mean = np.array((4,1)).astype(np.float)
# versicolor_mean = np.array((4,1)).astype(np.float)
# virginica_mean = np.array((4,1)).astype(np.float)

#gather sum for the mean value and divide data accordingly
for i in range(0, 40):
    # setosa_mean[i] += dataset_iris[i].getT()
    setosa_mean[0][0] += float(dataset_iris[i][0])
    setosa_mean[1][0] += float(dataset_iris[i][1])
    setosa_mean[2][0] += float(dataset_iris[i][2])
    setosa_mean[3][0] += float(dataset_iris[i][3])
    iris_setosa_training[i] = dataset_iris[i]

print(setosa_mean)
setosa_mean[0][0] = float(setosa_mean[0][0]) / 40
setosa_mean[1][0] = float(setosa_mean[1][0]) / 40
setosa_mean[2][0] = float(setosa_mean[2][0]) / 40
setosa_mean[3][0] = float(setosa_mean[3][0]) / 40
print(setosa_mean)

for i in range(0, 40):
    temp_matrix4by1[0][0] = float(dataset_iris[i][0]) - setosa_mean[0][0]
    temp_matrix4by1[1][0] = float(dataset_iris[i][1]) - setosa_mean[1][0]
    temp_matrix4by1[2][0] = float(dataset_iris[i][2]) - setosa_mean[2][0]
    temp_matrix4by1[3][0] = float(dataset_iris[i][3]) - setosa_mean[3][0]

    temp_matrix1by4[0][0] = float(dataset_iris[i][0]) - setosa_mean[0][0]
    temp_matrix1by4[0][1] = float(dataset_iris[i][1]) - setosa_mean[1][0]
    temp_matrix1by4[0][2] = float(dataset_iris[i][2]) - setosa_mean[2][0]
    temp_matrix1by4[0][3] = float(dataset_iris[i][3]) - setosa_mean[3][0]

    setosa_covariance += np.dot(temp_matrix4by1, temp_matrix1by4)

setosa_covariance = setosa_covariance / 40
print(setosa_covariance)

for i in range(0, 10):
    iris_setosa_test[i] = dataset_iris[i + 40]

for i in range(0, 40):
    versicolor_mean[0][0] += float(dataset_iris[i + 50][0])
    versicolor_mean[1][0] += float(dataset_iris[i + 50][1])
    versicolor_mean[2][0] += float(dataset_iris[i + 50][2])
    versicolor_mean[3][0] += float(dataset_iris[i + 50][3])
    iris_versicolor_training[i] = dataset_iris[i + 50]

print(versicolor_mean)
versicolor_mean[0][0] = float(versicolor_mean[0][0]) / 40
versicolor_mean[1][0] = float(versicolor_mean[1][0]) / 40
versicolor_mean[2][0] = float(versicolor_mean[2][0]) / 40
versicolor_mean[3][0] = float(versicolor_mean[3][0]) / 40
print(versicolor_mean)

for i in range(0, 10):
    iris_versicolor_test[i] = dataset_iris[i + 90]

for i in range(0, 40):
    virginica_mean[0][0] += float(dataset_iris[i + 100][0])
    virginica_mean[1][0] += float(dataset_iris[i + 100][1])
    virginica_mean[2][0] += float(dataset_iris[i + 100][2])
    virginica_mean[3][0] += float(dataset_iris[i + 100][3])
    iris_virginica_training[i] = dataset_iris[i + 100]

print(virginica_mean)
virginica_mean[0][0] = float(virginica_mean[0][0]) / 40
virginica_mean[1][0] = float(virginica_mean[1][0]) / 40
virginica_mean[2][0] = float(virginica_mean[2][0]) / 40
virginica_mean[3][0] = float(virginica_mean[3][0]) / 40
print(virginica_mean)

for i in range(0, 10):
    iris_virginica_test[i] = dataset_iris[i + 140]

# setosa_training_array = np.array(iris_setosa_training)
#
# setosa_mean_array = np.mean(setosa_training_array, 0)
#
# print(setosa_mean_array)
