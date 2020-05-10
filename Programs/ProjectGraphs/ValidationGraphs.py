import matplotlib.pyplot as plt
import numpy as np
import math
# This program displays all architectures's validation and training error on the same graph...
# It also finds the smallest validation and training errors on each architecture and their corresponding iteration number...


def load_error_data(error_graph_file_name):

    # reading error values and returns them..

    error_graph_file = open(error_graph_file_name, "r")

    iteration_axis = []
    error_axis = []
    validation_error_axis = []
    min_error = 1000000000
    min_validation_error = 10000000000000
    min_index = 0
    while True:
        error = error_graph_file.readline()
        if error == "":
            break
        values = error.split(" ")
        iteration_axis.append(int(values[0]))
        # Train error
        error_value = float(values[1])
        validation_error_value = float(values[2])
        error_axis.append(error_value)
        validation_error_axis.append(validation_error_value)
        if min_error > error_value:
            min_error = error_value



        if min_validation_error > validation_error_value:
            min_validation_error = validation_error_value
            # minimum validation value indx...
            min_index = int(values[0])


    error_graph_file.close()
    return iteration_axis, error_axis, validation_error_axis, min_error, min_validation_error, min_index

    pass

# Trained architectures and their types....
driver_names = {"MLPDriver" : ["MlpRegression", "MlpRegressionRGB", "MlpRegression3Images", "MlpRegressionRGB3Images" ],
                "CNNDriver": ["CNNRegression", "CNNRegressionRGB", "CNNRegression3Image", "CNNRegressionRGB3Image"],
                "RNNDriver": ["RNNRegression", "RNNRegressionRGB", "RNNRegression3Image", "RNNRegressionRGB3Image"],
                "HybridDriver": ["HybridRegressionRGB3Image", "HybridRegression3Image", "HybridRegressionRGB3ImageSingleInput"]}


# Trained architectures and their types....
driver_names2 = {"MLPDriver" : ["MlpRegression", "MlpRegressionRGB", "MlpRegression3Images", "MlpRegressionRGB3Images" ],
                "CNNDriver": ["CNNRegression", "CNNRegressionRGB", "CNNRegression3Image", "CNNRegressionRGB3Image"],
                "RNNDriver": ["RNNRegression", "RNNRegressionRGB", "RNNRegression3Image", "RNNRegressionRGB3Image"],
                "HybridDriver": ["HybridRegressionRGB3Image"]}


train_results = []
validation_results = []
iteration_results = []

min_train_errors = {}
min_validation_errors = {}
min_indexes = {}

# For each architectures we are collecting error data
for drivers in driver_names2.keys():
    for driv in driver_names2[drivers]:
        directory = "../"+drivers+"/"
        road_name = "Road4"
        error_graph_file_name = directory + driv + road_name + "_graph.dat"
        iteration_axis, error_axis, validation_error_axis, min_error, min_validation_error, min_valid_index = load_error_data(error_graph_file_name)
        min_train_errors[driv] = min_error
        min_validation_errors[driv] = min_validation_error
        min_indexes[driv] = min_valid_index

        iteration_results.append(iteration_axis)
        train_results.append(error_axis)
        validation_results.append(validation_error_axis)

        pass

# Displaying the error curves on the same plot
for index in range(len(iteration_results)):

    plt.subplot(211)
    plt.plot(iteration_results[index], train_results[index], label="Error")
    plt.subplot(212)

    plt.plot(iteration_results[index], validation_results[index], label="Error")
print "Min train errors : ", min_train_errors
print "Min validation errors : ", min_validation_errors
print "Min valid indexes : ", min_indexes
plt.show()