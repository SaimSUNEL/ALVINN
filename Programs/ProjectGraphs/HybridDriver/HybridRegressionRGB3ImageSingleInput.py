import tensorflow as tf
import numpy as np
import os
import cv2
import pickle
import math
import matplotlib.pyplot as plt
# We are loading statistics file of the architecture
# We had created this file during training...
# Mean image and difference image are required, because we have to normalize test set images...

root_directory = "../../HybridDriver/"
road_name = "Road4"
direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = pickle.load(open(root_directory+"HybridRegressionRGB3ImageSingleInputTrainer("+road_name+")_important_data.dat","r"))

# Loading test set's speed and direction angle

road_name = "Road4"
data_type = "Test"
location_name = "VehicleData"+road_name+data_type
camera_image_size = 100

validation_data_dictionary = {}
validation_direction_data = []
validation_speed_data = []

data_file = open("../../DataCollector/"+location_name+"/data.dat", "r")
for line in data_file:
    f = line.split(" ")
    if len(f) == 5:
        validation_data_dictionary[f[0]] = f
        validation_direction_data.append(int(f[1]))
        validation_speed_data.append(int(f[3]))

data_file.close()

validation_direction_mean = np.mean(validation_direction_data)
validation_direction_std = np.std(validation_direction_data)
validation_direction_max = np.max(validation_direction_data)
validation_direction_min = np.min(validation_direction_data)

validation_speed_mean = np.mean(validation_speed_data)
validation_speed_std = np.std(validation_speed_data)
validation_speed_max = np.max(validation_speed_data)
validation_speed_min = np.min(validation_speed_data)



# Converting string to integer...
for img in validation_data_dictionary.keys():
    validation_data_dictionary[img][1] = int(validation_data_dictionary[img][1])#-direction_min)# - direction_mean)/direction_std
    validation_data_dictionary[img][3] = int(validation_data_dictionary[img][3])#-speed_min)# - speed_mean) / speed_std




data_file.close()

# loading test set images....

validation_information_set = []
validation_information_set2 = []
validation_whole_image_set = []
road_name = "Road4"
data_type = "Test"
location_name = "VehicleData"+road_name+data_type

validation_image_number = 0
validation_image_file_list = os.listdir("../../DataCollector/"+location_name)
for dosya in validation_image_file_list:
    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[: -1*len("Resized.png")]

        image_file = cv2.imread("../../DataCollector/"+location_name+"/" + str(validation_image_number) + "Resized.png")

        validation_whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))
        # image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # angle, speed

        validation_information_set.append([validation_data_dictionary[str(validation_image_number)][1], validation_data_dictionary[str(validation_image_number)][3]])

        validation_image_number += 1

validation_whole_image_set = np.array(validation_whole_image_set, dtype=np.float32)



# Normalizing test images....

validation_whole_image_set = (validation_whole_image_set-mean_image)/diff


# reshaping test images for CNN

validation_image_set = [np.reshape(validation_whole_image_set[index], (100, 100, 3)) for index in range(0, validation_whole_image_set.shape[0])]


validation_speed_history = []

# Stacking three consequtive images into one and 10 last speed values...
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

# Speed history reshaping for RNN architecture...
validation_speed_history = [np.reshape(data, (2, 5)) for data in validation_speed_history]



# definition of trained model....


# input is three RGB image...
camera_image = tf.placeholder(tf.float32, shape=[None, 100, 100, 9])
# direction target values will be fed to network
target_values = tf.placeholder(tf.float32, [None, 2], "target_values")







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

conv2_out = tf.nn.tanh(tf.nn.bias_add(conv2, b2))


# Max pooling for second convolutional layer...
max_pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")



conv2_normalize = tf.reshape(max_pool2, shape=[-1,9 ,9*32])


basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=50)
outputs, states = tf.nn.dynamic_rnn(basic_cell, conv2_normalize, dtype=tf.float32)

input_number = 50
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w3 = tf.get_variable("w3", (50, 100), initializer=initializer)
b3 = tf.Variable(tf.zeros(100), name="b3")

f1_output = tf.nn.tanh(tf.matmul(states[1], w3) + b3)


input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W8 = tf.get_variable("W8", (228, 1), tf.float32, initializer)
b8 = tf.get_variable("b8", [1], tf.float32, tf.constant_initializer(0))



input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W4 = tf.get_variable("W4", (100, 2), tf.float32, initializer)
b4 = tf.get_variable("b4", [2], tf.float32, tf.constant_initializer(0))

final_output = tf.matmul(f1_output, W4) + b4

global_step = tf.Variable(0, name="global_step", trainable=False)




loss_ = (1/30.0)*tf.squared_difference(final_output, target_values)
summation = tf.reduce_sum(loss_)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
optimize = optimizer.minimize(summation)
train_saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()



road_name = "Road4"
# Load trained model...

train_saver.restore(sess, root_directory+"Trained_NN/HybridRegressionRGB3ImageSingleInput"+road_name)
j = 0
# Calculating test error....

for bacht_index in range(0, len(validation_information_set), 30):
    loss_value = sess.run(summation,
                          feed_dict={camera_image: validation_image_set[bacht_index: bacht_index + 30],
                                     target_values: validation_information_set[bacht_index:bacht_index + 30]
                                     })

    j += loss_value

# Displaying test error and storing it to a file for comparison with other architectures...

print "Loss : ", j

dosya = open("../TestResult"+road_name+".txt", "a")
dosya.write("HybridRegressionRGB3ImageSingleInputTest %f" % j + "\n")
dosya.close()
print "Loss : ", j
