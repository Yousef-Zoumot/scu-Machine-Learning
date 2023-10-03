import numpy as np

COLUMNS_FEATURES = 57


def get_Num_Instances(textfile):
    #open text file with data points
    text_file = open(textfile, "r")
    #split each line
    lines = text_file.read().split("\n")
    text_file.close()
    # print(lines)
    # print(len(lines))
    total_data_instances = len(lines) - 1
    # print(total_data_instances)
    return total_data_instances

def get_Data(textfile, num_data_instances):
    #open text file with data points
    text_file = open(textfile, "r")
    #split each line
    lines = text_file.read().split("\n")
    text_file.close()
    # print(lines)
    #[[0 for x in range(columns)] for y in range(rows)]
    dataset = [[float(0) for x in range(COLUMNS_FEATURES)] for y in range(num_data_instances)]
    for i in range(num_data_instances):
        #split data points of each instance
        dataset[i] = lines[i].split(",")
    # print(dataset_ftaps[1][0])
    return dataset


def sigmoid(temp_in):
    return 1 / (1 + np.exp( - temp_in))

def calc_RMSE(Y_new, Y_true, instances):
    rmse = 0
    for i in range(0, instances):
        rmse += np.absolute(Y_new[i] - Y_true[i])
                # rmse += (Y_new[i] - Y_true[i]) * (Y_new[i] - Y_true[i])
    return rmse/instances

def calc_mean(dataset, instances):

    mean = np.zeros((COLUMNS_FEATURES,1))

    for i in range(0, instances):
        for k in range(0, COLUMNS_FEATURES):
            mean[k][0] += float(dataset[i][k])
    for i in range (0, COLUMNS_FEATURES):
        mean[i][0] = float(mean[i][0]) / instances

    return mean

def calc_covariance(dataset, instances, mean):

    covariance = np.zeros((COLUMNS_FEATURES,COLUMNS_FEATURES))

    temp_matrixCOLUMNS_FEATURESby1 = np.zeros((COLUMNS_FEATURES,1))
    temp_matrix1byCOLUMNS_FEATURES = np.zeros((1,COLUMNS_FEATURES))

    for i in range(0, instances):
        for k in range(0, COLUMNS_FEATURES):
            temp_matrixCOLUMNS_FEATURESby1[k][0] = float(dataset[i][k]) - mean[k][0]
            temp_matrix1byCOLUMNS_FEATURES[0][k] = float(dataset[i][k]) - mean[k][0]
        covariance += np.matmul(temp_matrixCOLUMNS_FEATURESby1, temp_matrix1byCOLUMNS_FEATURES)

    covariance = covariance / instances

    return covariance

#standardize features using (feature - mean of feature) / standard deviation of feature
def extract_inputs1(dataset, num_data_instances, num_columns):
    temp_mean = calc_mean(dataset, num_data_instances)
    temp_covariance = calc_covariance(dataset, num_data_instances, temp_mean)

    inputs = np.zeros((num_data_instances, num_columns + 1))

    for i in range(0, num_data_instances):
        for k in range(0, num_columns + 1):
            if(k == num_columns):
                inputs[i][k] = 1
            if(k < 48):
                inputs[i][k] = (float(dataset[i][k]) - temp_mean[k][0]) / temp_covariance[k][0]
            if( k > 47 and k < 54):
                inputs[i][k] = (float(dataset[i][k]) - temp_mean[k][0]) / temp_covariance[k][0]
            if(k == 54):
                inputs[i][k] = (float(dataset[i][k]) - temp_mean[k][0]) / temp_covariance[k][0]
            if(k == 55):
                inputs[i][k] = (float(dataset[i][k]) - temp_mean[k][0]) / temp_covariance[k][0]
            if(k == 56):
                inputs[i][k] = (float(dataset[i][k]) - temp_mean[k][0]) / temp_covariance[k][0]
            # else:
            #     inputs[i][k] = dataset[i][k]

    return inputs

def extract_inputs2(dataset, num_data_instances, num_columns):

    inputs = np.zeros((num_data_instances, num_columns + 1))

    for i in range(0, num_data_instances):
        for k in range(0, num_columns + 1):
            if(k == num_columns):
                inputs[i][k] = 1
            if(k < 48):
                inputs[i][k] = np.log2(float(dataset[i][k]) + 0.1)
            if( k > 47 and k < 54):
                inputs[i][k] = np.log2(float(dataset[i][k]) + 0.1)
            if(k == 54):
                inputs[i][k] = np.log2(float(dataset[i][k]) + 0.1)
            if(k == 55):
                inputs[i][k] = np.log2(float(dataset[i][k]) + 0.1)
            if(k == 56):
                inputs[i][k] = np.log2(float(dataset[i][k]) + 0.1)
            # else:
            #     inputs[i][k] = dataset[i][k]

    return inputs

def extract_inputs3(dataset, num_data_instances, num_columns):
    temp_mean = calc_mean(dataset, num_data_instances)
    temp_covariance = calc_covariance(dataset, num_data_instances, temp_mean)

    inputs = np.zeros((num_data_instances, num_columns + 1))

    for i in range(0, num_data_instances):
        for k in range(0, num_columns + 1):
            if(k == num_columns):
                inputs[i][k] = 1
            if(sigmoid(float(dataset[i][k])) >= 0.5):
                inputs[i][k] = 1
            if(sigmoid(float(dataset[i][k])) < 0.5):
                inputs[i][k] = 0
            # else:
            #     inputs[i][k] = dataset[i][k]

    return inputs

def extract_outputs(dataset, num_data_instances):
    outputs = np.zeros((num_data_instances, 1))

    for i in range(0, num_data_instances):
        outputs[i][0] = dataset[i][57]

    return outputs

def logistic_regression_gradient_descent_training(X, Y, instances, learning_rate, percision, goal):
    #Since we will be generating random numbers, seed them to make them deterministic
    #Give random numbers that are generated
    # the same starting point or "seed" so that we'll get the sam
    # esequence of generated numbers every time we run our program.
    #Useful for debugging
    np.random.seed(1)
    error = 0
    # learning_rate = 0.000000000003039999999999999999999
    # learning_rate = 0.0000000099
    W = np.random.random((COLUMNS_FEATURES + 1, 1))
    W1 = np.random.random((COLUMNS_FEATURES + 1, 1))

    error = calc_RMSE(np.matmul(X, W), Y, instances)
    error2 = calc_RMSE(np.matmul(X, W1), Y, instances)

    variable = 0
    # for k in range(10000):
    while(True):
        error = calc_RMSE(sigmoid(np.matmul(X, W)), Y, instances)
        W = W1
        W1 += learning_rate * (np.matmul(np.transpose(X), (Y - sigmoid(np.matmul(X, W)))))

        error2 = calc_RMSE(sigmoid(np.matmul(X, W)), Y, instances)
        if(variable % 50 == 0):
            # error2 = calc_RMSE(sigmoid(np.matmul(X, W)), Y, instances)
            print("Mean Error of Logistic Regression GD: %f" %error)
            # if(np.absolute(error2 - error) < 0.00000000000000001):
            #     break

        if(np.absolute(error2 - error) < percision or error2 <= goal):
            print("FINAL Mean Error of Logistic Regression GD: %f" %error)
            # print(np.absolute(error2 - error))
            # print(error)
            # print(error2)
            break
        variable += 1

    # print(calc_RMSE(np.matmul(X, W), Y, instances))
    #
    print(sigmoid(np.matmul(X, W1)))
    return sigmoid(np.matmul(X, W1))
    # return W1

dataset_training = get_Data("spam-train", get_Num_Instances("spam-train"))
dataset_testing = get_Data("spam-test", get_Num_Instances("spam-test"))

dataset_outputs_train = extract_outputs(get_Data("spam-train", get_Num_Instances("spam-train")), get_Num_Instances("spam-train"))
dataset_outputs_test = extract_outputs(get_Data("spam-test", get_Num_Instances("spam-test")), get_Num_Instances("spam-test"))


dataset_inputs = extract_inputs1(get_Data("spam-train", get_Num_Instances("spam-train")), get_Num_Instances("spam-train"), COLUMNS_FEATURES)
Ynew_1_train = logistic_regression_gradient_descent_training(dataset_inputs, dataset_outputs_train, get_Num_Instances("spam-train"), 0.0000000099, 0.000000001, 0.085806)
dataset_inputs = extract_inputs1(get_Data("spam-test", get_Num_Instances("spam-test")), get_Num_Instances("spam-test"), COLUMNS_FEATURES)
Ynew_1_test = logistic_regression_gradient_descent_training(dataset_inputs, dataset_outputs_test, get_Num_Instances("spam-test"), 0.0000000099, 0.00000000000001, 0.08000)

dataset_inputs = extract_inputs2(get_Data("spam-train", get_Num_Instances("spam-train")), get_Num_Instances("spam-train"), COLUMNS_FEATURES)
Ynew_2_train = logistic_regression_gradient_descent_training(dataset_inputs, dataset_outputs_train, get_Num_Instances("spam-train"), 0.001, 0.000000000001, 0.053340)
dataset_inputs = extract_inputs1(get_Data("spam-test", get_Num_Instances("spam-test")), get_Num_Instances("spam-test"), COLUMNS_FEATURES)
Ynew_2_test = logistic_regression_gradient_descent_training(dataset_inputs, dataset_outputs_test, get_Num_Instances("spam-test"), 0.000001, 0.0000000000000001, 0.066)

dataset_inputs = extract_inputs3(get_Data("spam-train", get_Num_Instances("spam-train")), get_Num_Instances("spam-train"), COLUMNS_FEATURES)
Ynew3 = logistic_regression_gradient_descent_training(dataset_inputs, dataset_outputs_train, get_Num_Instances("spam-train"), 0.0002, 0.000000000000000000000001, 0.069)
dataset_inputs = extract_inputs1(get_Data("spam-test", get_Num_Instances("spam-test")), get_Num_Instances("spam-test"), COLUMNS_FEATURES)
Ynew_3_test = logistic_regression_gradient_descent_training(dataset_inputs, dataset_outputs_test, get_Num_Instances("spam-test"), 0.00000001, 0.00000000000000001, 0.078)
