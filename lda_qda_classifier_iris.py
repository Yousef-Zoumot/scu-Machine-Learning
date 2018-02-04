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

setosa_mean = np.zeros((4,1))
versicolor_mean = np.zeros((4,1))
virginica_mean = np.zeros((4,1))

setosa_covariance = np.zeros((4,4))
versicolor_covariance = np.zeros((4,4))
virginica_covariance = np.zeros((4,4))

temp_matrix4by1 = np.zeros((4,1))
temp_matrix1by4 = np.zeros((1,4))

# setosa_mean = [[float(0) for x in range(1)] for y in range(4)]
# versicolor_mean = [[float(0) for x in range(1)] for y in range(4)]
# virginica_mean = [[float(0) for x in range(1)] for y in range(4)]
#
# setosa_covariance = [[float(0) for x in range(4)] for y in range(4)]
# versicolor_covariance = [[float(0) for x in range(4)] for y in range(4)]
# virginica_covariance = [[float(0) for x in range(4)] for y in range(4)]
#
# temp_matrix4by1 = [[float(0) for x in range(1)] for y in range(4)]
# temp_matrix1by4 = [[float(0) for x in range(4)] for y in range(1)]

# iris_setosa_training = np.matrix([[float(0) for x in range(5)] for y in range(40)])
# iris_setosa_test = np.matrix([[float(0) for x in range(5)] for y in range(10)])
#
# iris_versicolor_training = np.matrix([[float(0) for x in range(5)] for y in range(40)])
# iris_versicolor_test = np.matrix([[float(0) for x in range(5)] for y in range(10)])
#
# iris_virginica_training =  np.matrix([[float(0) for x in range(5)] for y in range(40)])
# iris_virginica_test = np.matrix([[float(0) for x in range(5)] for y in range(10)])
#
# setosa_mean = np.matrix([[float(0) for x in range(1)] for y in range(4)])
# versicolor_mean = np.matrix([[float(0) for x in range(1)] for y in range(4)])
# virginica_mean = np.matrix([[float(0) for x in range(1)] for y in range(4)])
#
# setosa_covariance = np.matrix([[float(0) for x in range(4)] for y in range(4)])
# versicolor_covariance = np.matrix([[float(0) for x in range(4)] for y in range(4)])
# virginica_covariance = np.matrix([[float(0) for x in range(4)] for y in range(4)])
#
# temp_matrix4by1 = np.matrix([[float(0) for x in range(1)] for y in range(4)])
# temp_matrix1by4 = np.matrix([[float(0) for x in range(4)] for y in range(1)])


# temp_matrix4by1 = [[1],[1],[1],[1]]
# temp_matrix1by4 = [[1, 2, 3, 4]]
# print(np.linalg.det([[1, 2], [3, 4]]))
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

setosa_mean[0][0] = float(setosa_mean[0][0]) / 40
setosa_mean[1][0] = float(setosa_mean[1][0]) / 40
setosa_mean[2][0] = float(setosa_mean[2][0]) / 40
setosa_mean[3][0] = float(setosa_mean[3][0]) / 40
print("setosa_mean = %s\n" %setosa_mean)

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
print("setosa_covariance = %s\n" %setosa_covariance)

for i in range(0, 10):
    iris_setosa_test[i] = dataset_iris[i + 40]

for i in range(0, 40):
    versicolor_mean[0][0] += float(dataset_iris[i + 50][0])
    versicolor_mean[1][0] += float(dataset_iris[i + 50][1])
    versicolor_mean[2][0] += float(dataset_iris[i + 50][2])
    versicolor_mean[3][0] += float(dataset_iris[i + 50][3])
    iris_versicolor_training[i] = dataset_iris[i + 50]

versicolor_mean[0][0] = float(versicolor_mean[0][0]) / 40
versicolor_mean[1][0] = float(versicolor_mean[1][0]) / 40
versicolor_mean[2][0] = float(versicolor_mean[2][0]) / 40
versicolor_mean[3][0] = float(versicolor_mean[3][0]) / 40
print("versicolor_mean = %s\n" %versicolor_mean)

for i in range(0, 40):
    temp_matrix4by1[0][0] = float(dataset_iris[i + 50][0]) - versicolor_mean[0][0]
    temp_matrix4by1[1][0] = float(dataset_iris[i + 50][1]) - versicolor_mean[1][0]
    temp_matrix4by1[2][0] = float(dataset_iris[i + 50][2]) - versicolor_mean[2][0]
    temp_matrix4by1[3][0] = float(dataset_iris[i + 50][3]) - versicolor_mean[3][0]

    temp_matrix1by4[0][0] = float(dataset_iris[i + 50][0]) - versicolor_mean[0][0]
    temp_matrix1by4[0][1] = float(dataset_iris[i + 50][1]) - versicolor_mean[1][0]
    temp_matrix1by4[0][2] = float(dataset_iris[i + 50][2]) - versicolor_mean[2][0]
    temp_matrix1by4[0][3] = float(dataset_iris[i + 50][3]) - versicolor_mean[3][0]

    versicolor_covariance += np.dot(temp_matrix4by1, temp_matrix1by4)

versicolor_covariance = versicolor_covariance / 40
print("versicolor_covariance = %s\n" %versicolor_covariance)

for i in range(0, 10):
    iris_versicolor_test[i] = dataset_iris[i + 90]

for i in range(0, 40):
    virginica_mean[0][0] += float(dataset_iris[i + 100][0])
    virginica_mean[1][0] += float(dataset_iris[i + 100][1])
    virginica_mean[2][0] += float(dataset_iris[i + 100][2])
    virginica_mean[3][0] += float(dataset_iris[i + 100][3])
    iris_virginica_training[i] = dataset_iris[i + 100]

virginica_mean[0][0] = float(virginica_mean[0][0]) / 40
virginica_mean[1][0] = float(virginica_mean[1][0]) / 40
virginica_mean[2][0] = float(virginica_mean[2][0]) / 40
virginica_mean[3][0] = float(virginica_mean[3][0]) / 40
print("virginica_mean = %s\n" %virginica_mean)

for i in range(0, 40):
    temp_matrix4by1[0][0] = float(dataset_iris[i + 100][0]) - virginica_mean[0][0]
    temp_matrix4by1[1][0] = float(dataset_iris[i + 100][1]) - virginica_mean[1][0]
    temp_matrix4by1[2][0] = float(dataset_iris[i + 100][2]) - virginica_mean[2][0]
    temp_matrix4by1[3][0] = float(dataset_iris[i + 100][3]) - virginica_mean[3][0]

    temp_matrix1by4[0][0] = float(dataset_iris[i + 100][0]) - virginica_mean[0][0]
    temp_matrix1by4[0][1] = float(dataset_iris[i + 100][1]) - virginica_mean[1][0]
    temp_matrix1by4[0][2] = float(dataset_iris[i + 100][2]) - virginica_mean[2][0]
    temp_matrix1by4[0][3] = float(dataset_iris[i + 100][3]) - virginica_mean[3][0]

    virginica_covariance += np.dot(temp_matrix4by1, temp_matrix1by4)

virginica_covariance = virginica_covariance / 40
print("virginica_covariance = %s\n" %virginica_covariance)

for i in range(0, 10):
    iris_virginica_test[i] = dataset_iris[i + 140]

covariance_average = (setosa_covariance + versicolor_covariance + virginica_covariance)/3
print("covariance_average = %s\n" %covariance_average)

def prob_data_given_cond(data, mean, covariance):
    data1 = np.matrix(data)
    data1 = np.transpose(data1)
    mean1 = np.matrix(mean)
    covariance1 = np.matrix(covariance)

    return  (1 / (np.sqrt(np.power((2 * np.pi), 4) * np.linalg.det(covariance1)))) * np.exp(-0.5 * np.matmul(np.transpose(data1 - mean1), np.matmul(np.linalg.inv(covariance1), (data1 - mean1))) )
    #(1 / (np.sqrt(np.power((2 * np.pi), 4) * np.linalg.det(covariance)))) *
    #(1 /  np.linalg.det(covariance)) *

print("The following probabilities are the iris_setosa LDA test:")
print("p(x0 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[0][0:4]], setosa_mean, covariance_average))
print("p(x0 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[0][0:4]], versicolor_mean, covariance_average))
print("p(x0 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[0][0:4]], virginica_mean, covariance_average))

print("p(x1 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[1][0:4]], setosa_mean, covariance_average))
print("p(x1 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[1][0:4]], versicolor_mean, covariance_average))
print("p(x1 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[1][0:4]], virginica_mean, covariance_average))

print("p(x2 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[2][0:4]], setosa_mean, covariance_average))
print("p(x2 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[2][0:4]], versicolor_mean, covariance_average))
print("p(x2 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[2][0:4]], virginica_mean, covariance_average))

print("p(x3 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[3][0:4]], setosa_mean, covariance_average))
print("p(x3 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[3][0:4]], versicolor_mean, covariance_average))
print("p(x3 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[3][0:4]], virginica_mean, covariance_average))

print("p(x4 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[4][0:4]], setosa_mean, covariance_average))
print("p(x4 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[4][0:4]], versicolor_mean, covariance_average))
print("p(x4 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[4][0:4]], virginica_mean, covariance_average))

print("p(x5 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[5][0:4]], setosa_mean, covariance_average))
print("p(x5 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[5][0:4]], versicolor_mean, covariance_average))
print("p(x5 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[5][0:4]], virginica_mean, covariance_average))

print("p(x6 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[6][0:4]], setosa_mean, covariance_average))
print("p(x6 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[6][0:4]], versicolor_mean, covariance_average))
print("p(x6 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[6][0:4]], virginica_mean, covariance_average))

print("p(x7 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[7][0:4]], setosa_mean, covariance_average))
print("p(x7 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[7][0:4]], versicolor_mean, covariance_average))
print("p(x7 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[7][0:4]], virginica_mean, covariance_average))

print("p(x8 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[8][0:4]], setosa_mean, covariance_average))
print("p(x8 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[8][0:4]], versicolor_mean, covariance_average))
print("p(x8 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[8][0:4]], virginica_mean, covariance_average))

print("p(x9 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[9][0:4]], setosa_mean, covariance_average))
print("p(x9 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[9][0:4]], versicolor_mean, covariance_average))
print("p(x9 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[9][0:4]], virginica_mean, covariance_average))

#iris_versicolor QDA test
print("The following probabilities are the iris_versicolor LDA test:")
print("p(x0 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[0][0:4]], setosa_mean, covariance_average))
print("p(x0 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[0][0:4]], versicolor_mean, covariance_average))
print("p(x0 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[0][0:4]], virginica_mean, covariance_average))

print("p(x1 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[1][0:4]], setosa_mean, covariance_average))
print("p(x1 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[1][0:4]], versicolor_mean, covariance_average))
print("p(x1 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[1][0:4]], virginica_mean, covariance_average))

print("p(x2 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[2][0:4]], setosa_mean, covariance_average))
print("p(x2 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[2][0:4]], versicolor_mean, covariance_average))
print("p(x2 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[2][0:4]], virginica_mean, covariance_average))

print("p(x3 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[3][0:4]], setosa_mean, covariance_average))
print("p(x3 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[3][0:4]], versicolor_mean, covariance_average))
print("p(x3 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[3][0:4]], virginica_mean, covariance_average))

print("p(x4 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[4][0:4]], setosa_mean, covariance_average))
print("p(x4 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[4][0:4]], versicolor_mean, covariance_average))
print("p(x4 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[4][0:4]], virginica_mean, covariance_average))

print("p(x5 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[5][0:4]], setosa_mean, covariance_average))
print("p(x5 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[5][0:4]], versicolor_mean, covariance_average))
print("p(x5 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[5][0:4]], virginica_mean, covariance_average))

print("p(x6 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[6][0:4]], setosa_mean, covariance_average))
print("p(x6 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[6][0:4]], versicolor_mean, covariance_average))
print("p(x6 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[6][0:4]], virginica_mean, covariance_average))

print("p(x7 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[7][0:4]], setosa_mean, covariance_average))
print("p(x7 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[7][0:4]], versicolor_mean, covariance_average))
print("p(x7 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[7][0:4]], virginica_mean, covariance_average))

print("p(x8 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[8][0:4]], setosa_mean, covariance_average))
print("p(x8 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[8][0:4]], versicolor_mean, covariance_average))
print("p(x8 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[8][0:4]], virginica_mean, covariance_average))

print("p(x9 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[9][0:4]], setosa_mean, covariance_average))
print("p(x9 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[9][0:4]], versicolor_mean, covariance_average))
print("p(x9 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[9][0:4]], virginica_mean, covariance_average))

#iris_virginica QDA test
print("The following probabilities are the iris_virginica LDA test:")
print("p(x0 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[0][0:4]], setosa_mean, covariance_average))
print("p(x0 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[0][0:4]], versicolor_mean, covariance_average))
print("p(x0 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[0][0:4]], virginica_mean, covariance_average))

print("p(x1 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[1][0:4]], setosa_mean, covariance_average))
print("p(x1 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[1][0:4]], versicolor_mean, covariance_average))
print("p(x1 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[1][0:4]], virginica_mean, covariance_average))

print("p(x2 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[2][0:4]], setosa_mean, covariance_average))
print("p(x2 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[2][0:4]], versicolor_mean, covariance_average))
print("p(x2 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[2][0:4]], virginica_mean, covariance_average))

print("p(x3 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[3][0:4]], setosa_mean, covariance_average))
print("p(x3 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[3][0:4]], versicolor_mean, covariance_average))
print("p(x3 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[3][0:4]], virginica_mean, covariance_average))

print("p(x4 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[4][0:4]], setosa_mean, covariance_average))
print("p(x4 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[4][0:4]], versicolor_mean, covariance_average))
print("p(x4 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[4][0:4]], virginica_mean, covariance_average))

print("p(x5 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[5][0:4]], setosa_mean, covariance_average))
print("p(x5 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[5][0:4]], versicolor_mean, covariance_average))
print("p(x5 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[5][0:4]], virginica_mean, covariance_average))

print("p(x6 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[6][0:4]], setosa_mean, covariance_average))
print("p(x6 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[6][0:4]], versicolor_mean, covariance_average))
print("p(x6 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[6][0:4]], virginica_mean, covariance_average))

print("p(x7 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[7][0:4]], setosa_mean, covariance_average))
print("p(x7 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[7][0:4]], versicolor_mean, covariance_average))
print("p(x7 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[7][0:4]], virginica_mean, covariance_average))

print("p(x8 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[8][0:4]], setosa_mean, covariance_average))
print("p(x8 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[8][0:4]], versicolor_mean, covariance_average))
print("p(x8 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[8][0:4]], virginica_mean, covariance_average))

print("p(x9 | u1, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[9][0:4]], setosa_mean, covariance_average))
print("p(x9 | u2, S_avg) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[9][0:4]], versicolor_mean, covariance_average))
print("p(x9 | u3, S_avg) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[9][0:4]], virginica_mean, covariance_average))

#iris_setosa QDA test
print("The following probabilities are the iris_setosa QDA test:")
print("p(x0 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[0][0:4]], setosa_mean, setosa_covariance))
print("p(x0 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[0][0:4]], versicolor_mean, versicolor_covariance))
print("p(x0 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[0][0:4]], virginica_mean, virginica_covariance))

print("p(x1 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[1][0:4]], setosa_mean, setosa_covariance))
print("p(x1 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[1][0:4]], versicolor_mean, versicolor_covariance))
print("p(x1 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[1][0:4]], virginica_mean, virginica_covariance))

print("p(x2 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[2][0:4]], setosa_mean, setosa_covariance))
print("p(x2 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[2][0:4]], versicolor_mean, versicolor_covariance))
print("p(x2 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[2][0:4]], virginica_mean, virginica_covariance))

print("p(x3 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[3][0:4]], setosa_mean, setosa_covariance))
print("p(x3 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[3][0:4]], versicolor_mean, versicolor_covariance))
print("p(x3 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[3][0:4]], virginica_mean, virginica_covariance))

print("p(x4 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[4][0:4]], setosa_mean, setosa_covariance))
print("p(x4 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[4][0:4]], versicolor_mean, versicolor_covariance))
print("p(x4 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[4][0:4]], virginica_mean, virginica_covariance))

print("p(x5 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[5][0:4]], setosa_mean, setosa_covariance))
print("p(x5 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[5][0:4]], versicolor_mean, versicolor_covariance))
print("p(x5 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[5][0:4]], virginica_mean, virginica_covariance))

print("p(x6 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[6][0:4]], setosa_mean, setosa_covariance))
print("p(x6 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[6][0:4]], versicolor_mean, versicolor_covariance))
print("p(x6 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[6][0:4]], virginica_mean, virginica_covariance))

print("p(x7 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[7][0:4]], setosa_mean, setosa_covariance))
print("p(x7 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[7][0:4]], versicolor_mean, versicolor_covariance))
print("p(x7 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[7][0:4]], virginica_mean, virginica_covariance))

print("p(x8 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[8][0:4]], setosa_mean, setosa_covariance))
print("p(x8 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[8][0:4]], versicolor_mean, versicolor_covariance))
print("p(x8 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[8][0:4]], virginica_mean, virginica_covariance))

print("p(x9 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[9][0:4]], setosa_mean, setosa_covariance))
print("p(x9 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_setosa_test[9][0:4]], versicolor_mean, versicolor_covariance))
print("p(x9 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_setosa_test[9][0:4]], virginica_mean, virginica_covariance))

#iris_versicolor QDA test
print("The following probabilities are the iris_versicolor QDA test:")
print("p(x0 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[0][0:4]], setosa_mean, setosa_covariance))
print("p(x0 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[0][0:4]], versicolor_mean, versicolor_covariance))
print("p(x0 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[0][0:4]], virginica_mean, virginica_covariance))

print("p(x1 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[1][0:4]], setosa_mean, setosa_covariance))
print("p(x1 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[1][0:4]], versicolor_mean, versicolor_covariance))
print("p(x1 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[1][0:4]], virginica_mean, virginica_covariance))

print("p(x2 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[2][0:4]], setosa_mean, setosa_covariance))
print("p(x2 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[2][0:4]], versicolor_mean, versicolor_covariance))
print("p(x2 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[2][0:4]], virginica_mean, virginica_covariance))

print("p(x3 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[3][0:4]], setosa_mean, setosa_covariance))
print("p(x3 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[3][0:4]], versicolor_mean, versicolor_covariance))
print("p(x3 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[3][0:4]], virginica_mean, virginica_covariance))

print("p(x4 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[4][0:4]], setosa_mean, setosa_covariance))
print("p(x4 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[4][0:4]], versicolor_mean, versicolor_covariance))
print("p(x4 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[4][0:4]], virginica_mean, virginica_covariance))

print("p(x5 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[5][0:4]], setosa_mean, setosa_covariance))
print("p(x5 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[5][0:4]], versicolor_mean, versicolor_covariance))
print("p(x5 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[5][0:4]], virginica_mean, virginica_covariance))

print("p(x6 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[6][0:4]], setosa_mean, setosa_covariance))
print("p(x6 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[6][0:4]], versicolor_mean, versicolor_covariance))
print("p(x6 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[6][0:4]], virginica_mean, virginica_covariance))

print("p(x7 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[7][0:4]], setosa_mean, setosa_covariance))
print("p(x7 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[7][0:4]], versicolor_mean, versicolor_covariance))
print("p(x7 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[7][0:4]], virginica_mean, virginica_covariance))

print("p(x8 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[8][0:4]], setosa_mean, setosa_covariance))
print("p(x8 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[8][0:4]], versicolor_mean, versicolor_covariance))
print("p(x8 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[8][0:4]], virginica_mean, virginica_covariance))

print("p(x9 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[9][0:4]], setosa_mean, setosa_covariance))
print("p(x9 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_versicolor_test[9][0:4]], versicolor_mean, versicolor_covariance))
print("p(x9 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_versicolor_test[9][0:4]], virginica_mean, virginica_covariance))

#iris_virginica QDA test
print("The following probabilities are the iris_virginica QDA test:")
print("p(x0 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[0][0:4]], setosa_mean, setosa_covariance))
print("p(x0 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[0][0:4]], versicolor_mean, versicolor_covariance))
print("p(x0 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[0][0:4]], virginica_mean, virginica_covariance))

print("p(x1 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[1][0:4]], setosa_mean, setosa_covariance))
print("p(x1 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[1][0:4]], versicolor_mean, versicolor_covariance))
print("p(x1 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[1][0:4]], virginica_mean, virginica_covariance))

print("p(x2 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[2][0:4]], setosa_mean, setosa_covariance))
print("p(x2 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[2][0:4]], versicolor_mean, versicolor_covariance))
print("p(x2 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[2][0:4]], virginica_mean, virginica_covariance))

print("p(x3 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[3][0:4]], setosa_mean, setosa_covariance))
print("p(x3 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[3][0:4]], versicolor_mean, versicolor_covariance))
print("p(x3 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[3][0:4]], virginica_mean, virginica_covariance))

print("p(x4 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[4][0:4]], setosa_mean, setosa_covariance))
print("p(x4 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[4][0:4]], versicolor_mean, versicolor_covariance))
print("p(x4 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[4][0:4]], virginica_mean, virginica_covariance))

print("p(x5 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[5][0:4]], setosa_mean, setosa_covariance))
print("p(x5 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[5][0:4]], versicolor_mean, versicolor_covariance))
print("p(x5 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[5][0:4]], virginica_mean, virginica_covariance))

print("p(x6 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[6][0:4]], setosa_mean, setosa_covariance))
print("p(x6 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[6][0:4]], versicolor_mean, versicolor_covariance))
print("p(x6 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[6][0:4]], virginica_mean, virginica_covariance))

print("p(x7 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[7][0:4]], setosa_mean, setosa_covariance))
print("p(x7 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[7][0:4]], versicolor_mean, versicolor_covariance))
print("p(x7 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[7][0:4]], virginica_mean, virginica_covariance))

print("p(x8 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[8][0:4]], setosa_mean, setosa_covariance))
print("p(x8 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[8][0:4]], versicolor_mean, versicolor_covariance))
print("p(x8 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[8][0:4]], virginica_mean, virginica_covariance))

print("p(x9 | u1, S1) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[9][0:4]], setosa_mean, setosa_covariance))
print("p(x9 | u2, S2) = %s" %prob_data_given_cond([float(i) for i in iris_virginica_test[9][0:4]], versicolor_mean, versicolor_covariance))
print("p(x9 | u3, S3) = %s\n" %prob_data_given_cond([float(i) for i in iris_virginica_test[9][0:4]], virginica_mean, virginica_covariance))

# setosa_training_array = np.array(iris_setosa_training)
#
# setosa_mean_array = np.mean(setosa_training_array, 0)
#
# print(setosa_mean_array)
