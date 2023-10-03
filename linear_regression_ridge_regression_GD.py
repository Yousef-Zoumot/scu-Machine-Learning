import numpy as np

COLUMNS_FEATURES = 95


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
		  dataset[i] = lines[i].split("\t")
	 # print(dataset_ftaps[1][0])
	 return dataset

def calc_RMSE(Y_new, Y_true, instances):
	 rmse = 0
	 for i in range(0, instances):
		  # rmse += np.sqrt((Y_new[i] - Y_true[i]) * (Y_new[i] - Y_true[i]))
          rmse += (Y_new[i] - Y_true[i]) * (Y_new[i] - Y_true[i])

	 return np.sqrt(rmse/instances)

def extract_inputs(dataset, num_data_instances, num_columns):
    inputs = np.zeros((num_data_instances, num_columns + 1))
    for i in range(0, num_data_instances):
        for k in range(0, num_columns + 1):
            if(k == num_columns):
                inputs[i][k] = 1
            else:
                inputs[i][k] = dataset[i][k + 1]
    return inputs

def extract_outputs(dataset, num_data_instances):
	 outputs = np.zeros((num_data_instances, 1))

	 for i in range(0, num_data_instances):
		  outputs[i][0] = dataset[i][0]

	 return outputs

def linear_regression_closed_form_training(X, Y):
	 # W = (X.T X).inverse X.T Y
	 return np.matmul( (np.linalg.inv(np.matmul(np.transpose(X), X))), np.matmul(np.transpose(X), Y))

def linear_regression_closed_form_testing(X, W):
	 # Ynew = W.T * Xnew
	 return np.matmul(X, W)

# def linear_regression_closed_form(X, Y):
#     return linear_regression_closed_form_testing(X, linear_regression_closed_form_training(X, Y))

def ridge_regression_closed_form_training(X, Y, lamd_Da):
	 # Training: W = (X.T * X + lambda*IdentityMatrix).inverse  * X.T * Y
	 #
	 # lambda*IdentityMatrix is a (p +1) x (p+1) matrix
	 return np.matmul( np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X) + (np.identity(96) * lamd_Da)) , np.transpose(X) ) , Y)

def ridge_regression_closed_form_testing(X, W):
	 # Testing: Ynew = W.T * Xnew
	 return np.matmul(X, W)

def linear_regression_gradient_descent_training(X, Y, instances):
    #Since we will be generating random numbers, seed them to make them deterministic
    #Give random numbers that are generated
    # the same starting point or "seed" so that we'll get the sam
    # esequence of generated numbers every time we run our program.
    #Useful for debugging
    np.random.seed(1)
    error = 0
    learning_rate = 0.00001

    W = np.random.random((96, 1))
    W1 = np.random.random((96,1))

    error = calc_RMSE(np.matmul(X, W), Y, instances)
    error2 = calc_RMSE(np.matmul(X, W1), Y, instances)

    variable = 0
    # for k in range(10000):
    while(True):
        error = calc_RMSE(np.matmul(X, W), Y, instances)
        W = W1
        W1 += learning_rate * (np.matmul(np.transpose(X), (Y - np.matmul(X, W))))

        error2 = calc_RMSE(np.matmul(X, W), Y, instances)
        if(variable % 250 == 0):
            print("Mean Squared Error of Linear Regression GD: %f" %error)

        if(error - error2 < 0.000001):
            break
        variable += 1

    print(calc_RMSE(np.matmul(X, W), Y, instances))

    return np.matmul(X, W1)

# def linear linear_regression_gradient_descent_testing():
#
def ridge_regression_gradient_descent_training(X, Y, lamb_da, instances):
    #Since we will be generating random numbers, seed them to make them deterministic
    #Give random numbers that are generated
    # the same starting point or "seed" so that we'll get the sam
    # esequence of generated numbers every time we run our program.
    #Useful for debugging
    np.random.seed(1)
    error = 0
    learning_rate = 0.00001

    W = np.random.random((96, 1))
    W1 = np.random.random((96,1))

    error = calc_RMSE(np.matmul(X, W), Y, instances)
    error2 = calc_RMSE(np.matmul(X, W1), Y, instances)

    variable = 0
    # for k in range(10000):
    while(True):
        error = calc_RMSE(np.matmul(X, W), Y, instances)
        W = W1
        W1 += learning_rate * (np.matmul(np.transpose(X), (Y - np.matmul(X, W))) - lamb_da * W)

        error2 = calc_RMSE(np.matmul(X, W), Y, instances)
        if(variable % 250 == 0):
            print("Mean Squared Error of Ridge Regression GD: %f" %error)

        if(error - error2 < 0.000001):
            break
        variable += 1

    print(calc_RMSE(np.matmul(X, W), Y, instances))

    return np.matmul(X, W1)
#
# def ridge_regression_gradient_descent_testing():
#
def cross_validation(k, X, Y, lamb_da, instances):
    splitz = instances / k
    splitz_percent = 1 / k

    splitz_input_matrix = np.zeros((int(0.8 * instances), COLUMNS_FEATURES +1))
    splitz_output_matrix = np.zeros((int(0.8 * instances), 1))

    splitz_testing_input = np.zeros((int(0.2 * instances), COLUMNS_FEATURES+1))
    splitz_testing_output = np.zeros((int(0.2 * instances) , 1))

    average = 0
    for i in range(k):
        for j in range(instances):
            if(i == 0 and j < int(0.8 * instances)):
                splitz_input_matrix[j] = X[j]
                splitz_output_matrix[j] = Y[j]
                # print(splitz_input_matrix[1])

            if(i == 0 and j >= int(0.8 * instances)):
                splitz_testing_input[j - int(0.8 * instances)] = X[j]
                splitz_testing_output[j - int(0.8 * instances)] = Y[j]

            if(i == 1 and (j < int(0.6 * instances) or j >= int(0.8 * instances))):
                if(j < int(0.6 * instances)):
                    splitz_input_matrix[j] = X[j]
                    splitz_output_matrix[j] = Y[j]
                if(j >= int(0.8 * instances)):
                    splitz_input_matrix[j - int(0.6 * instances)] = X[j]
                    splitz_output_matrix[j- int(0.6 * instances)] = Y[j]

            if(i == 1 and j >= int(0.6 * instances) and j < int(0.8 * instances)):
                splitz_testing_input[j- int(0.6 * instances)] = X[j]
                splitz_testing_output[j- int(0.6 * instances)] = Y[j]

            if(i == 2 and (j < int(0.4 * instances) or j >= int(0.6 * instances))):
                if(j < int(0.4 * instances)):
                    splitz_input_matrix[j] = X[j]
                    splitz_output_matrix[j] = Y[j]
                if(j >= int(0.6 * instances)):
                    splitz_input_matrix[j - int(0.4 * instances)] = X[j]
                    splitz_output_matrix[j- int(0.4 * instances)] = Y[j]

            if(i == 2 and j >= int(0.4 * instances) and j < int(0.6 * instances)):
                splitz_testing_input[j - int(0.4 * instances)] = X[j]
                splitz_testing_output[j - int(0.4 * instances)] = Y[j]

            if(i == 3 and (j < int(0.2 * instances) or j >= int(0.4 * instances))):
                if(j < int(0.2 * instances)):
                    splitz_input_matrix[j] = X[j]
                    splitz_output_matrix[j] = Y[j]
                if(j >= int(0.4 * instances)):
                    splitz_input_matrix[j - int(0.2 * instances)] = X[j]
                    splitz_output_matrix[j - int(0.2 * instances)] = Y[j]

            if(i == 3 and j >= int(0.2 * instances) and j < int(0.4 * instances)):
                splitz_testing_input[j - int(0.2 * instances)] = X[j]
                splitz_testing_output[j - int(0.2 * instances)] = Y[j]

            if(i == 4 and j >= int(0.2 * instances)):
                splitz_input_matrix[j - int(0.2 * instances)] = X[j]
                splitz_output_matrix[j - int(0.2 * instances)] = Y[j]

            if(i == 4 and j < int(0.2 * instances)):
                splitz_testing_input[j] = X[j]
                splitz_testing_output[j] = Y[j]

        # print(splitz_input_matrix)
        # print(splitz_output_matrix)
        # print(splitz_testing_input)
        # print(splitz_testing_output)
        weights = ridge_regression_closed_form_training(splitz_input_matrix, splitz_output_matrix, lamb_da)
        Ynew = ridge_regression_closed_form_testing(splitz_testing_input, weights)
        # print(weights)
        # print(Ynew)
        print("When k is %d and lambda is %f the RMSE is: " %(i + 1,lamb_da))
        print(calc_RMSE(Ynew, splitz_testing_output, int(0.2 * instances)))
        average += calc_RMSE(Ynew, splitz_testing_output, int(0.2 * instances))
    average /= 5
    print("When lambda is %f the AVERAGE RMSE is: " %(lamb_da))
    print(average)
    print("\n")

dataset_training = get_Data("crime_training.txt", get_Num_Instances("crime_training.txt"))
dataset_testing = get_Data("crime_test.txt", get_Num_Instances("crime_test.txt"))

dataset_inputs = extract_inputs(get_Data("crime_training.txt", get_Num_Instances("crime_training.txt")), get_Num_Instances("crime_training.txt"), COLUMNS_FEATURES)
dataset_outputs = extract_outputs(get_Data("crime_training.txt", get_Num_Instances("crime_training.txt")), get_Num_Instances("crime_training.txt"))

#linear_regression_closed_form_training
weights = linear_regression_closed_form_training(dataset_inputs, dataset_outputs)

dataset_testing_inputs = extract_inputs(get_Data("crime_test.txt", get_Num_Instances("crime_test.txt")), get_Num_Instances("crime_test.txt"), COLUMNS_FEATURES)
dataset_testing_outputs = extract_outputs(get_Data("crime_test.txt", get_Num_Instances("crime_test.txt")), get_Num_Instances("crime_test.txt"))

#linear_regression_closed_form_error on test data
temp_inst = get_Num_Instances("crime_test.txt")
Ynew = linear_regression_closed_form_testing(dataset_testing_inputs, weights)
print("The RMSE of closed form linear regression on the test data is: ")
print(calc_RMSE(Ynew, dataset_testing_outputs, temp_inst)) #14.58%

#linear_regression_closed_form_error on training data
temp_inst = get_Num_Instances("crime_training.txt")
Ynew = linear_regression_closed_form_testing(dataset_inputs, weights)
print("The RMSE of closed form linear regression on the training data is: ")
print(calc_RMSE(Ynew, dataset_outputs, temp_inst))#12.76%


weights = ridge_regression_closed_form_training(dataset_testing_inputs, dataset_testing_outputs, 400)

Ynew = ridge_regression_closed_form_testing(dataset_testing_inputs, weights)

temp_inst = get_Num_Instances("crime_test.txt")
print("The RMSE of closed form ridge regression on the test data is: ")
print(calc_RMSE(Ynew, dataset_testing_outputs, temp_inst))#18.95%


temp_inst = get_Num_Instances("crime_training.txt")
lamb_da = 400
for i in range(10):
    cross_validation(5, dataset_inputs, dataset_outputs, lamb_da, temp_inst)
    lamb_da /= 2.0

Ynew = linear_regression_gradient_descent_training(dataset_inputs, dataset_outputs, get_Num_Instances("crime_training.txt"))
#
Ynew = ridge_regression_gradient_descent_training(dataset_inputs, dataset_outputs, 12.5, get_Num_Instances("crime_training.txt"))

Ynew = ridge_regression_gradient_descent_training(dataset_testing_inputs, dataset_testing_outputs, 12.5, get_Num_Instances("crime_test.txt"))

# print(calc_RMSE(Ynew, dataset_outputs))
