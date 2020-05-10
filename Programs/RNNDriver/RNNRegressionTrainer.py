import tensorflow as tf
import numpy as np
import os
import cv2
import pickle
import math
import matplotlib.pyplot as plt



# We will save the trained model into this folder...
# If the folder does not exist, we create one...
if not os.path.isdir("Trained_NN"):
    os.mkdir("Trained_NN")


# We are reading the speed and direction information of the validation set...
# We are storing the data in arrays...
road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type
camera_image_size = 100
# This will hold all the information for the image...
# "image_number": [image number direction_angle normalized_angle speed normalized_speed]
validation_data_dictionary = {}
# Whole direction information will be hold in this to extract statistics of direction information...

validation_direction_data = []
# Whole speed information is stored in this to extract statistics of speed info..

validation_speed_data = []
# We are reading data from file...
data_file = open("../DataCollector/"+location_name+"/data.dat", "r")
for line in data_file:
    f = line.split(" ")
    if len(f) == 5:
        validation_data_dictionary[f[0]] = f
        validation_direction_data.append(int(f[1]))
        validation_speed_data.append(int(f[3]))

data_file.close()

# Statistics of validation direction data...

validation_direction_mean = np.mean(validation_direction_data)
validation_direction_std = np.std(validation_direction_data)
validation_direction_max = np.max(validation_direction_data)
validation_direction_min = np.min(validation_direction_data)

# Statistics of validation speed data...

validation_speed_mean = np.mean(validation_speed_data)
validation_speed_std = np.std(validation_speed_data)
validation_speed_max = np.max(validation_speed_data)
validation_speed_min = np.min(validation_speed_data)


# We are reading direction and speed information of training set...
# Whole data will be hold in this array...
data_dictionary = {}
camera_image_size = 100
# v = (v-min)/(max-min)
# whole direction information of training set will be hold in this array...

direction_data = []
# Whole speed information of training set will be stored in this array...

speed_data = []
# we are reading data file to extract information...
# each line contains image_number (current direction angle) (current angle normalize) (current_speed) ()

road_name = "Road4"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type

# We are reading whole information from file...

data_file = open("../DataCollector/"+location_name+"/data.dat", "r")
for line in data_file:
    f = line.split(" ")
    if len(f) == 5:
        data_dictionary[f[0]] = f
        direction_data.append(int(f[1]))
        speed_data.append(int(f[3]))

# Statistics of direction and speed information of training set

direction_mean = np.mean(direction_data)
direction_std = np.std(direction_data)
direction_max = np.max(direction_data)
direction_min = np.min(direction_data)

speed_mean = np.mean(speed_data)
speed_std = np.std(speed_data)
speed_max = np.max(speed_data)
speed_min = np.min(speed_data)

# When we read first data, it is string type, we are converting it to integer
# For training set....

for img in data_dictionary.keys():
    data_dictionary[img][1] = int(data_dictionary[img][1])# - direction_mean)\
                              #/ (direction_max -direction_min)
    data_dictionary[img][3] = int(data_dictionary[img][3])# - speed_mean) / (speed_max-speed_min)



# For validation set....

for img in validation_data_dictionary.keys():
    validation_data_dictionary[img][1] = int(validation_data_dictionary[img][1])# - direction_mean)\
                             # / (direction_max-direction_min)
    validation_data_dictionary[img][3] = int(validation_data_dictionary[img][3])# - speed_mean) / (speed_max-speed_min)


# We are reading validation set images...

road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type
camera_image_size = 100

validation_image_set = []
# Corresponding angle and direction information will be hold in this
# in form of [[12, 25], [13, 20]]
validation_information_set = []
validation_information_set2 = []
# Whole validation set images will be hold in this array...

validation_whole_image_set = []

# we are loading images...
# getting list of image files...
validation_image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in validation_image_file_list:
    # Only load the GRAY images...

    if dosya.__contains__("GRAYResized"):
        dosya_number = dosya[:-1*len("GRAYResized.png")]
        # read image file

        image_file = cv2.imread("../DataCollector/"+location_name+"/"+dosya)
        image_file = image_file[:, :, 0]
        # Add to array... and reshape for neural network..

        validation_whole_image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # Add corresponding image's angle and speed information... 100x100 = 10000

        validation_information_set.append([validation_data_dictionary[dosya_number][1], validation_data_dictionary[dosya_number][3]])
# Converting numpy array...

validation_whole_image_set = np.array(validation_whole_image_set, dtype=np.float32)



data_file.close()

# We are loading training set images...

image_set = []
# Corresponding speed and direction information....

information_set = []
information_set2 = []
# Whole training set's images will be stored in this...

whole_image_set = []

road_name = "Road4"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type


# Retrieve image list of training set....
image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in image_file_list:
    # Load only gray scale images...

    if dosya.__contains__("GRAYResized"):
        dosya_number = dosya[:-1*len("GRAYResized.png")]
        image_file = cv2.imread("../DataCollector/"+location_name+"/"+dosya)
        # we are using gray scale image of train data...
        image_file = image_file[:, :, 0]
        # add image to array by reshaping it for neural network.. 100x100 = 10000
        whole_image_set.append(np.reshape(image_file, (camera_image_size**2)))

        # angle, speed
        # current image's corresponding target values are loaded to target set...
        # corresponding angle, speed of image...

        information_set.append([data_dictionary[dosya_number][1], data_dictionary[dosya_number][3]])
# Converting to numpy array...

whole_image_set = np.array(whole_image_set, dtype=np.float32)

# To eliminate correlation between samples, we are shuffling the data set
indices = [a for a in range(len(whole_image_set))]
np.random.shuffle(indices)

# We are shuffling dataset, and the target values also must be shuffled as it is done to data set...
whole_image_set = whole_image_set[indices]
gf = []
df = []
for h in indices:
    gf.append(information_set[h])

information_set = gf;


# We will apply mean normalization to training set before supplying it to neural network...
# image_normalized = (current_image - mean_image)/(max_image - min_image)


# To normalize images we are extracting mean of training data set...

mean_image = whole_image_set.mean(axis=0)
std_image = whole_image_set.std(axis=0)

# There are 10000 thousand dimension, for each dimension we are getting the max and min of the
# training set
# Max image of the trainin set...
max_image = whole_image_set.max(axis=0)
min_image = whole_image_set.min(axis=0)
# We are getting different image from max and min images...

diff = max_image - min_image
# Some dimensions might be 0 after subtraction, to mitigate division by zero problem we
# assignning the zero index to max value..
for i in range(len(diff)):
    if abs(diff[i]) < 0.0000001:
        diff[i] = max_image[i]
# We are normalizing the traingin image...

whole_image_set = (whole_image_set - mean_image)
whole_image_set = whole_image_set/(diff)
# We are also normalizing the validation set...

validation_whole_image_set = (validation_whole_image_set-mean_image)/diff

validation_image_set = [np.reshape(validation_whole_image_set[index], (10,1000)) for index in range(0, validation_whole_image_set.shape[0])]


image_set = [ np.reshape(whole_image_set[index], (10, 1000)) for index in range(0, whole_image_set.shape[0])]
#direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min = pickle.load(open("important_data.dat","r"))

# We are stoing the statistics of training set, these infromation will be used by the driver
# programs....
# we are saving statistical information...
pickle.dump([np.max(direction_data), np.min(direction_data),
                    direction_mean, direction_std, np.max(speed_data),
                    np.min(speed_data), speed_mean, speed_std, mean_image, std_image, whole_image_set.max(),
             whole_image_set.min(), diff], open("RNNRegressionTrainer("+road_name+")_important_data.dat", "w"))

print "Data have been loaded"

# We are creating the recurrent neural network in tensorflow...
# After RNN layer one two-hidden layered MLP structure is used to output estimated direction angle and speed
# Placeholders for input data and target values...
# Images will be supplied to this placeholder..
# RNN will use LSTM units...
# Each input size is 1000 and the RNN will be unfolded 10 time steps....
camera_image = tf.placeholder(tf.float32, shape=[None, 10, 1000])
target_values = tf.placeholder(tf.float32, shape=[None, 2])

# RNN layer consists of 256 nodes...
basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=256)
outputs, states = tf.nn.dynamic_rnn(basic_cell, camera_image, dtype=tf.float32)


# We are not using each time step's RNN output, instead we are using the last(10th) step
# of RNN output as input to MLP structure....
# MLP's input layer is the RNN node count...
# MLP's first hidden layer count is 100 nodes...
input_number = 256
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w1 = tf.get_variable("w1", (256, 100), initializer=initializer)
b1 = tf.Variable(tf.zeros(100), name="b1")
fc_output = tf.nn.tanh(tf.matmul(states[1], w1)+b1)


# MLP's second hidden layer count is 50
input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w2 = tf.get_variable("w2", (100, 50), initializer=initializer)
b2 = tf.Variable(tf.zeros(50), name="b2")
fc_output1 = tf.nn.tanh(tf.matmul(fc_output, w2)+b2)

# MLP's output layer is 2 nodes...
input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w3 = tf.get_variable("w3", (50, 2), tf.float32, initializer)
b3 = tf.get_variable("b3", [2], tf.float32, tf.constant_initializer(0))
# At output no nonlienarity is applied...
final_output = tf.matmul(fc_output1, w3) + b3

# Loss minimized is the mean squared difference... 1/30 because the batch size is 30
loss_ = (1/30.0)*tf.squared_difference(final_output, target_values)
summation = tf.reduce_sum(loss_)
# We are using ADAM optimizer while training...

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
optimize = optimizer.minimize(summation)
# Saver for trained model...

train_saver = tf.train.Saver()
# session creation...

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()

# during training we are keeping the validation error to save the model with smallest validation error...

road_name = "Road4.2"
smallest_validation_error = 10000000000

# During training, validation set error and training error is saved to file...

error_graph_file_name = "RNNRegression"+road_name+"_graph.dat"

# arrays required to show error graphs...

iteration_axis = []
error_axis = []
validation_error_axis = []




# If model is saved previously, load it and continue training...
# sometimes, we may need to continue training for previously trained model...
if len([file for file in os.listdir("Trained_NN") if file.__contains__("RNNRegression"+road_name)]) > 0:
    # Load previously saved model...

    train_saver.restore(sess, "Trained_NN/RNNRegression"+road_name)
    # Load the training error value from previous training...

    error_graph_file = open(error_graph_file_name, "r")

    while True:
        error = error_graph_file.readline()
        if error == "":
            break
        values = error.split(" ")
        iteration_axis.append(int(values[0]))
        # Train error
        error_axis.append(float(values[1]))
        validation_error_axis.append(float(values[2]))

    error_iteration_number = max(iteration_axis) + 1
    error_graph_file.close()
    error_graph_file = open(error_graph_file_name, "a")
    smallest_validation_error = min(validation_error_axis)

    print "Pretrained model has been loaded and will be continued to train..."


else:
    # If there is no previously saved model, initialize the neural network with random values...

    error_graph_file = open(error_graph_file_name, "w")
    sess.run(init)
    error_iteration_number = 1


# Iteration number of training...

iteration_count = 50
while iteration_count != 0:

    # after iteration count is completed, the user can continue with new iteration number if he/she wants...
    for iteration in range(iteration_count):
        print "**********Iteration "+str(iteration)+"****************"
        # Train network with whole training set and get the training set error...

        l = 0
        # we are performing 30 sample batch learning...
        for bacht_index in range(0, len(information_set), 30):

            _, loss_value = sess.run([optimize, summation],
                                     feed_dict={camera_image: image_set[bacht_index: bacht_index+30],
                                                target_values: information_set[bacht_index:bacht_index+30]})
            l += loss_value
        # After training, calculate the whole validation set error

        j = 0
        for bacht_index in range(0, len(validation_information_set), 30):
            # we are training with minibatch learning...
            loss_value = sess.run(summation,
                                  feed_dict={camera_image: validation_image_set[bacht_index: bacht_index + 30],
                                             target_values: validation_information_set[bacht_index:bacht_index + 30],
                                             })

            j += loss_value

        # If current validation error is less than the previous validation error
        # Save current model...
        # By doing so, in the end we will have the model saved with the lowest validation error value..

        if j < smallest_validation_error:
            smallest_validation_error = j
            print "Smallest error : ", smallest_validation_error
            print "Model has been saved..."
            train_saver.save(sess, "Trained_NN/RNNRegression" + road_name)

        # Add error values to graphs arrays...

        iteration_axis.append(error_iteration_number)
        error_axis.append(l)
        validation_error_axis.append(j)
        error_graph_file.write(str(error_iteration_number) + " " + str(l) + " " + str(j) + "\n")
        error_iteration_number += 1

        # we are also displaying the train data set error...
        print "Loss : ", l, " Validation error : ", j
        if l < 0.78:
            break
    # Plot the training and validation set errors...

    plt.subplot(211)
    plt.plot(iteration_axis, error_axis, label="Error")
    plt.subplot(212)
    plt.plot(iteration_axis, validation_error_axis, label="Error")
    plt.show()
    # After each training iteration is finished, user may want to continue training...

    # user specifies new iteration number after previous one finishes...
    print "Please enter new iteration count "
    iteration_count = int(raw_input())

error_graph_file.close()

print "Train has been terminated..."
