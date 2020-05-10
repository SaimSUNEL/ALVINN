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
road_name = "Road4"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type
# We are reading whole information from file...

data_file = open("../DataCollector/"+location_name+"/data.dat", "r")
# angle and speed information retrieved...
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


direction_min = min(direction_min, validation_direction_min)
speed_min = min(speed_min, validation_speed_min)


# When we read first data, it is string type, we are converting it to integer
# For training set....

for img in data_dictionary.keys():
    data_dictionary[img][1] = int(data_dictionary[img][1]) # - direction_mean)\
                              #/ (direction_max -direction_min)
    data_dictionary[img][3] = int(data_dictionary[img][3]) #- speed_mean) / (speed_max-speed_min)

# For validation set....
for img in validation_data_dictionary.keys():
    validation_data_dictionary[img][1] = int(validation_data_dictionary[img][1])#-direction_min)# - direction_mean)/direction_std
    validation_data_dictionary[img][3] = int(validation_data_dictionary[img][3])#-speed_min)# - speed_mean) / speed_std




data_file.close()

# We are reading validation set images...

# Corresponding angle and direction information will be hold in this
# in form of [[12, 25], [13, 20]]
validation_information_set = []
validation_information_set2 = []
# Whole validation set images will be hold in this array...

validation_whole_image_set = []
road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type

validation_image_number = 0
# we are loading images...
# getting list of image files...
validation_image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in validation_image_file_list:
    # Only load the RGB images...
    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[: -1*len("Resized.png")]
        # read image file

        image_file = cv2.imread("../DataCollector/"+location_name+"/" + str(validation_image_number) + "Resized.png")
        # Add to array... and reshape for neural network..

        validation_whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))
        # image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # angle, speed
        # Add corresponding image's angle and speed information... 100x100x3 = 30000

        validation_information_set.append([validation_data_dictionary[str(validation_image_number)][1], validation_data_dictionary[str(validation_image_number)][3]])

        validation_image_number += 1
# Converting numpy array...

validation_whole_image_set = np.array(validation_whole_image_set, dtype=np.float32)


# We are loading training set images...
# Corresponding speed and direction information....

information_set = []
information_set2 = []
# Whole training set's images will be stored in this...

whole_image_set = []
road_name = "Road4"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type

image_number = 0
image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in image_file_list:
    # Load only RGB images...

    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[: -1*len("Resized.png")]

        image_file = cv2.imread("../DataCollector/"+location_name+"/" + str(image_number) + "Resized.png")
        # add image to array by reshaping it for neural network.. 100x100x3 = 30000

        whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))
        # image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # angle, speed
        # corresponding angle, speed of image...

        information_set.append([data_dictionary[str(image_number)][1], data_dictionary[str(image_number)][3]])

        image_number += 1
# Converting to numpy array...

whole_image_set = np.array(whole_image_set, dtype=np.float32)


# To normalize images we are extracting mean of training data set...

mean_image = whole_image_set.mean(axis=0)
std_image = whole_image_set.std(axis=0)

# There are 30000 thousand dimension, for each dimension we are getting the max and min of the
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



image_set = [np.reshape(whole_image_set[index], (100, 100, 3))
             for index in range(0, whole_image_set.shape[0])]
validation_image_set = [np.reshape(validation_whole_image_set[index], (100, 100, 3)) for index in range(0, validation_whole_image_set.shape[0])]
direction_max = max(direction_max, validation_direction_max)
speed_max = max(speed_max, validation_speed_max)
speed_min = min(speed_min, validation_speed_min)


# direction_max, direction_min, direction_mean, direction_std, speed_max,
# speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min =
# pickle.load(open("important_data.dat","r"))
# We are stoing the statistics of training set, these infromation will be used by the driver
# programs....
pickle.dump([direction_max, direction_min,
             direction_mean, direction_std, speed_max,
             speed_min, speed_mean, speed_std, mean_image,
             std_image, whole_image_set.max(),
             whole_image_set.min(), diff], open("HybridRegressionRGB3ImageTrainer("+road_name+")_important_data.dat", "w"))
print "max : ", whole_image_set.max()
print "min : ", whole_image_set.min()

print "Data have been loaded"

validation_speed_history = []


# We are supplying 3 consequtive images and last 10 step speed information to neural network..
# We are stacking 3 consequtive images of validation set and 10 step previous speed information

validation_new_image_set = []
validation_new_information_set = []
validation_new_information_set2 = []
# last 10 speed values will be input to LSTM
for image_index in range(9, len(validation_image_set)):
    combinatination = np.array([validation_image_set[image_index], validation_image_set[image_index-1], validation_image_set[image_index-2]], dtype=np.float32)
    combinatination = np.reshape(combinatination, newshape=(100, 100, 9))

    last_ten = []
    for speed_index in range(0, 10):

        last_ten.append(validation_information_set[image_index - speed_index][1])

        pass
    validation_speed_history.append(last_ten)

    validation_new_image_set.append(combinatination)

    validation_new_information_set.append(validation_information_set[image_index])



validation_image_set = np.array(validation_new_image_set)
validation_information_set = validation_new_information_set



# We are supplying 3 consequtive images and last 10 step speed information to neural network..
# We are stacking 3 consequtive images of training set and 10 step previous speed information


speed_history = []

new_image_set = []
new_information_set = []
new_information_set2 = []
for image_index in range(9, len(image_set)):
    combinatination = np.array([image_set[image_index], image_set[image_index-1], image_set[image_index-2]], dtype=np.float32)
    combinatination = np.reshape(combinatination, newshape=(100, 100, 9))
    new_image_set.append(combinatination)
    last_ten = []
    for speed_index in range(0, 10):
        last_ten.append(information_set[image_index - speed_index][1])

        pass
    speed_history.append(last_ten)


    new_information_set.append(information_set[image_index])



image_set = np.array(new_image_set)
information_set = new_information_set


# We are reshaping previous speed informations of training and validation set to be able to
# input it to RNN network, 2 step unfolded each step's input is 5 nodes...
speed_history = [np.reshape(data, (2, 5)) for data in speed_history]
speed_history = np.array(speed_history, dtype=np.float32)

validation_speed_history = [np.reshape(data, (2, 5)) for data in validation_speed_history]

# To eliminate correlation between samples, we are shuffling the data set
indices = [a for a in range(len(image_set))]
np.random.shuffle(indices)
speed_history = speed_history[indices]

# We are shuffling dataset, and the target values also must be shuffled as it is done to data set...
image_set = image_set[indices]
gf = []
df = []
for h in indices:
    gf.append(information_set[h])

information_set = gf;


# Hyrib NN architecture definiton of the system in Tensorflow...
# Three consequitive RBG images are supplied to CNN part...
# Last 10 step speed information is fed to RNN part...
# By using two different kind of input types,
# whole architecture outputs estimates speed and direction angle


# input is 3 stacked consequtive RGB images...
camera_image = tf.placeholder(tf.float32, shape=[None, 100, 100, 9])
# direction and speed target values will be fed to network
target_values = tf.placeholder(tf.float32, [None, 2], "target_values")

# placeholder for last 10 speed input... RNN 2 step unfolding each input is 5 nodes...
speed_history_input = tf.placeholder(tf.float32, shape=[None, 2, 5])

# To process last 10 speed inputs we are using RNN with 2 unfold 5 inputs
with tf.variable_scope('lstm2'):
    # RNN layer consists of 128 nodes...
    basic_cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=128)
    outputs2, states2 = tf.nn.dynamic_rnn(basic_cell2, speed_history_input, dtype=tf.float32)

# fc_output = tf.nn.tanh(tf.matmul(states[1], w1)+b1)





# First convolution with 3 by 3 filters that create 64 feature maps...
input_number = 3 * 3 * 9
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W = tf.get_variable("W", (3, 3, 9, 64), tf.float32, initializer)
b = tf.get_variable("b", [64], tf.float32, tf.constant_initializer(0))

conv1 = tf.nn.conv2d(camera_image, W, strides=[1, 1, 1, 1], padding="SAME")

conv1_out = tf.nn.tanh(tf.nn.bias_add(conv1, b))

# Max pooling for the first convolution layer...
max_pool = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Second convolutional layer 3 by 3 filter produces 32 feature maps...
input_number = 3 * 3 * 64
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W2 = tf.get_variable("W2", (3, 3, 64, 32), tf.float32, initializer)
b2 = tf.get_variable("b2", [32], tf.float32, tf.constant_initializer(0))

conv2 = tf.nn.conv2d(max_pool, W2, strides=[1, 3, 3, 1], padding="SAME")

# Tanh is used an nonlinear function
conv2_out = tf.nn.tanh(tf.nn.bias_add(conv2, b2))


# Max pooling for second convolutional layer...
max_pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# After second convolutional layer we are using a RNN layer ...
# We are reshaping last convolutional output to feed it to RNN
# this RNN is unfolded 9 time steps, and input is 288 nodes...
conv2_normalize = tf.reshape(max_pool2, shape=[-1,9 ,9*32])

# This RNN layer has 50 nodes... After this RNN we are using a MLP architecture to predict
# Only direction angle ...
basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=50)
outputs, states = tf.nn.dynamic_rnn(basic_cell, conv2_normalize, dtype=tf.float32)

# We are only using 9th step's output as input to MLP layer(input layer 50 nodes..)
# MLP layer hidden layer has 100 nodes...
input_number = 50
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w3 = tf.get_variable("w3", (50, 100), initializer=initializer)
b3 = tf.Variable(tf.zeros(100), name="b3")

# We are using tanh at hidden layer of MLP
f1_output = tf.nn.tanh(tf.matmul(states[1], w3) + b3)

# For speed estimation, we are inputting last 10 speed information to an RNN
# Speed information can not be solely determined via the last 10 speed values
# While deciding the speed information, we benefit from images supplied to CNN architecture
# By using output of first hidden layer of MLP architecture on CNN network and last step output of
# RNN to which last 10 speed info is fed, we are estimating the speed information...
speed_concat = tf.concat([f1_output, states2[1]], 1)

# Total concatanation becomes 228 nodes.. and this combination is fed to another MLP architecture...

input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W8 = tf.get_variable("W8", (228, 1), tf.float32, initializer)
b8 = tf.get_variable("b8", [1], tf.float32, tf.constant_initializer(0))

# We are simply performing weighted summation on combination to predict speed info..
speed_output = tf.matmul(speed_concat, W8) + b8


# output of MLP after CNN architecture is estimated direction angle...
input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W4 = tf.get_variable("W4", (100, 1), tf.float32, initializer)
b4 = tf.get_variable("b4", [1], tf.float32, tf.constant_initializer(0))
# We are combining direction and speed outputs into one as other architectures...
final_output = tf.concat([tf.matmul(f1_output, W4) + b4, speed_output], 1)

global_step = tf.Variable(0, name="global_step", trainable=False)



# Loss minimized is the mean squared difference... 1/30 because the batch size is 30
loss_ = (1/30.0)*tf.squared_difference(final_output, target_values)
summation = tf.reduce_sum(loss_)
# We are using ADAM optimizer while training...

optimizer = tf.train.AdamOptimizer(learning_rate=0.000001)
optimize = optimizer.minimize(summation)
train_saver = tf.train.Saver()
# session creation...

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()

# during training we are keeping the validation error to save the model with smallest validation error...

road_name = "Road4.2"
smallest_validation_error = 10000000000

# During training, validation set error and training error is saved to file...

error_graph_file_name = "HybridRegressionRGB3Image"+road_name+"_graph.dat"

# arrays required to show error graphs...

iteration_axis = []
error_axis = []
validation_error_axis = []



# sometimes training procedure is interrupted... To continue it later...
# If model is saved previously, load it and continue training...

if len([file for file in os.listdir("Trained_NN") if file.__contains__("HybridRegressionRGB3Image"+road_name)]) > 0:
    # Load previously saved model...

    train_saver.restore(sess, "Trained_NN/HybridRegressionRGB3Image"+road_name)
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
    print "Pretrained model has been loaded and will be continued to train..."
    smallest_validation_error = min(validation_error_axis)

else:
    # If there is no previously saved model, initialize the neural network with random values...

    error_graph_file = open(error_graph_file_name, "w")
    sess.run(init)
    error_iteration_number = 1



# Iteration number of training...

iteration_count = 10
while iteration_count != 0:

    for iteration in range(iteration_count):
        print "**********Iteration "+str(iteration)+"****************"
        # Train network with whole training set and get the training set error...

        l = 0
        for bacht_index in range(0, len(information_set), 30):
            _, loss_value = sess.run([optimize, summation],
                                     feed_dict={camera_image: image_set[bacht_index: bacht_index+30],
                                                target_values: information_set[bacht_index:bacht_index+30],
                                                speed_history_input: speed_history[bacht_index: bacht_index+30]})
            l += loss_value
        # After training, calculate the whole validation set error

        j = 0
        for bacht_index in range(0, len(validation_information_set), 30):
            # we are training with minibatch learning...
            loss_value = sess.run(summation,
                                  feed_dict={camera_image: validation_image_set[bacht_index: bacht_index + 30],
                                             target_values: validation_information_set[bacht_index:bacht_index + 30],
                                             speed_history_input: validation_speed_history[bacht_index: bacht_index+30]
                                             })

            j += loss_value


        # If current validation error is less than the previous validation error
        # Save current model...
        # By doing so, in the end we will have the model saved with the lowest validation error value..

        if j < smallest_validation_error:
            smallest_validation_error = j
            print "Smallest error : ", smallest_validation_error
            print "Model has been saved..."
            train_saver.save(sess, "Trained_NN/HybridRegressionRGB3Image" + road_name)


        # Add error values to graphs arrays...

        iteration_axis.append(error_iteration_number)
        error_axis.append(l)
        validation_error_axis.append(j)
        error_graph_file.write(str(error_iteration_number) + " " + str(l) + " " + str(j) + "\n")
        error_iteration_number += 1
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

    print "Please enter new iteration count "
    iteration_count = int(raw_input())

error_graph_file.close()

print "Train has been terminated..."

