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

root_directory = "../../RNNDriver/"
road_name = "Road4"
direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = pickle.load(open(root_directory+"RNNRegressionRGB3ImageTrainer("+road_name+")_important_data.dat", "r"))



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
    validation_data_dictionary[img][1] = int(validation_data_dictionary[img][1])#-direction_min)# - direction_mean)/direction_std
    validation_data_dictionary[img][3] = int(validation_data_dictionary[img][3])#-speed_min)# - speed_mean) / speed_std




# loading test set images....


validation_information_set = []
validation_information_set2 = []
validation_whole_image_set = []
road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type
# Stacking 3 consequtive images into one for architecture...

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

# reshaping test images for RNN

validation_image_set = [np.reshape(validation_whole_image_set[index], (100, 100, 3)) for index in range(0, validation_whole_image_set.shape[0])]
# Stacking 3 consequtive images into one for architecture...

validation_new_image_set = []
validation_new_information_set = []
validation_new_information_set2 = []
for image_index in range(2, len(validation_image_set)):
    combinatination = np.array([validation_image_set[image_index], validation_image_set[image_index-1], validation_image_set[image_index-2]], dtype=np.float32)
    combinatination = np.reshape(combinatination, newshape=(90, 1000))
    validation_new_image_set.append(combinatination)

    validation_new_information_set.append(validation_information_set[image_index])



validation_image_set = np.array(validation_new_image_set)
validation_information_set = validation_new_information_set



# definition of trained model....

camera_image = tf.placeholder(tf.float32, shape=[None, 90, 1000])
target_values = tf.placeholder(tf.float32, shape=[None, 2])


basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=256)
outputs, states = tf.nn.dynamic_rnn(basic_cell, camera_image, dtype=tf.float32)

input_number = 256
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w1 = tf.get_variable("w1", (256, 100), initializer=initializer)
b1 = tf.Variable(tf.zeros(100), name="b1")
fc_output = tf.nn.tanh(tf.matmul(states[1], w1)+b1)


input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w2 = tf.get_variable("w2", (100, 50), initializer=initializer)
b2 = tf.Variable(tf.zeros(50), name="b2")
fc_output1 = tf.nn.tanh(tf.matmul(fc_output, w2)+b2)


input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w3 = tf.get_variable("w3", (50, 2), tf.float32, initializer)
b3 = tf.get_variable("b3", [2], tf.float32, tf.constant_initializer(0))

final_output = tf.matmul(fc_output1, w3) + b3



loss_ = (1/30.0)*tf.squared_difference(final_output, target_values)
summation = tf.reduce_sum(loss_)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
optimize = optimizer.minimize(summation)


train_saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()


road_name = "Road4"
# Load trained model...

train_saver.restore(sess, root_directory + "Trained_NN/RNNRegressionRGB3Image"+road_name)

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
dosya = open("../TestResult"+road_name+".txt", "a")
dosya.write("RNNRegressionRGB3ImageTest %f" % j + "\n")
dosya.close()
print "Loss : ", j