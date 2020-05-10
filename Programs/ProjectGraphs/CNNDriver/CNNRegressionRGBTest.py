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

root_directory = "../../CNNDriver/"
road_name = "Road4"
direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = pickle.load(open(root_directory+"CNNRegressionRGBTrainer("+road_name+")_important_data.dat", "r"))






# Loading test set's speed and direction angle

road_name = "Road4"
data_type = "Validation"
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
    validation_data_dictionary[img][1] = int(validation_data_dictionary[img][1])# - direction_mean)\
                              #/ (direction_max-direction_min)
    validation_data_dictionary[img][3] = int(validation_data_dictionary[img][3])# - speed_mean) / (speed_max-speed_min)







road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type
camera_image_size = 100

validation_image_set = []
validation_information_set = []
validation_information_set2 = []
validation_whole_image_set = []

# loading test set images....
validation_image_file_list = os.listdir("../../DataCollector/"+location_name)
for dosya in validation_image_file_list:
    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[:-1*len("Resized.png")]
        image_file = cv2.imread("../../DataCollector/"+location_name+"/"+dosya)

        validation_whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))

        validation_information_set.append([validation_data_dictionary[dosya_number][1], validation_data_dictionary[dosya_number][3]])

validation_whole_image_set = np.array(validation_whole_image_set, dtype=np.float32)


# Normalizing test images....

validation_whole_image_set = (validation_whole_image_set-mean_image)/diff


# reshaping test images for CNN

validation_image_set = [np.reshape(validation_whole_image_set[index], (100, 100, 3)) for index in range(0, validation_whole_image_set.shape[0])]

# definition of trained model....

# we are using a 3 channel (RGB) 100x100 image as input...
camera_image = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
# direction target values will be fed to network
target_values = tf.placeholder(tf.float32, [None, 2], "target_values")


# in the first convolution layer, we are using 3x3 filter and it these filters
# will produce 64 layer future map...

input_number = 3 * 3 * 3
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W = tf.get_variable("W", (3, 3, 3, 64), tf.float32, initializer)
# bias weights of first convolutional layer...
b = tf.get_variable("b", [64], tf.float32, tf.constant_initializer(0))
# leaky tanh activation function is used for first convolution layer...
conv1 = tf.nn.conv2d(camera_image, W, strides=[1, 1, 1, 1], padding="SAME")

conv1_out = tf.nn.tanh(tf.nn.bias_add(conv1, b))

# first pooling layer with 2x2 window is applied after first convolutional layer....
max_pool = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Second convolution layer with 3x3x64 3d filter to create 32 layer feature map in second convolutional layer
input_number = 3 * 3 * 64
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W2 = tf.get_variable("W2", (3, 3, 64, 32), tf.float32, initializer)
# bias weights of second convolutional layer...
b2 = tf.get_variable("b2", [32], tf.float32, tf.constant_initializer(0))

conv2 = tf.nn.conv2d(max_pool, W2, strides=[1, 3, 3, 1], padding="SAME")

# Second convolutional layer's activation fucntion tanh function...
conv2_out = tf.nn.tanh(tf.nn.bias_add(conv2, b2))
# again we are applying another pooling layer for the second convolutional layer with 2x2 window size...
max_pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# After second convolutional layer, we are construncting a feed-forward network... 100x32 for
conv2_normalize = tf.reshape(max_pool2, shape=[-1, 9 * 9 * 32])

input_number = 9 * 9 * 32
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
# feed-forward network's input is second convolution layer... We are constructing a hidden layer of 100 hidde nodes..
W3 = tf.get_variable("W3", (9 * 9 * 32, 100), tf.float32, initializer)
# first feed hidden layer bias....
b3 = tf.get_variable("b3", [100], tf.float32, tf.constant_initializer(0))

# feed-forward's hidden layer activation function is tanh function....
f1_output = tf.nn.tanh(tf.matmul(conv2_normalize, W3) + b3)

# for direction output we are constructing a 2 node output layer...
input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W4 = tf.get_variable("W4", (100, 2), tf.float32, initializer)
b4 = tf.get_variable("b4", [2], tf.float32, tf.constant_initializer(0))
# we are applying softmax to direction output...
final_output = tf.matmul(f1_output, W4) + b4

global_step = tf.Variable(0, name="global_step", trainable=False)




loss_ = (1/30.0)*tf.squared_difference(final_output, target_values)
summation = tf.reduce_sum(loss_)

optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
optimize = optimizer.minimize(summation)



train_saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()


road_name = "Road4"
# Load trained model...

train_saver.restore(sess, root_directory + "Trained_NN/CNNRegressionRGB"+road_name)

# Calculating test error....

j = 0
for bacht_index in range(0, len(validation_information_set), 30):
    loss_value = sess.run(summation,
                          feed_dict={camera_image: validation_image_set[bacht_index: bacht_index + 30],
                                     target_values: validation_information_set[bacht_index:bacht_index + 30],
                                     })

    j += loss_value


# Displaying test error and storing it to a file for comparison with other architectures...
print "Loss : ", j
print "Loss : ", j
dosya = open("../TestResult"+road_name+".txt", "a")
dosya.write("CNNRegressionRGBTest %f" % j + "\n")
dosya.close()