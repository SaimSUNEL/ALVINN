import tensorflow as tf
import numpy as np
import os
import cv2
import pickle
import math
import matplotlib.pyplot as plt

# We have commented RNNRegressionTrainer.py in detailed manner,
# Here we neglect the most part to comment, because most of the parts are the same...
# In this file same neural network with RNNRegressionTrainer.py is used...
# Only difference is that the input is a RGB image, 100x100x3 input image...
# It is supplied to RNN as 30 timesteps and each time step input is 1000



if not os.path.isdir("Trained_NN"):
    os.mkdir("Trained_NN")



# Loading speed, direction angle info of validation se...
road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type
camera_image_size = 100

validation_data_dictionary = {}
validation_direction_data = []
validation_speed_data = []

data_file = open("../DataCollector/"+location_name+"/data.dat", "r")
for line in data_file:
    f = line.split(" ")
    if len(f) == 5:
        validation_data_dictionary[f[0]] = f
        validation_direction_data.append(int(f[1]))
        validation_speed_data.append(int(f[3]))

data_file.close()

# Statistics of speed, direction angle of validation set
validation_direction_mean = np.mean(validation_direction_data)
validation_direction_std = np.std(validation_direction_data)
validation_direction_max = np.max(validation_direction_data)
validation_direction_min = np.min(validation_direction_data)

validation_speed_mean = np.mean(validation_speed_data)
validation_speed_std = np.std(validation_speed_data)
validation_speed_max = np.max(validation_speed_data)
validation_speed_min = np.min(validation_speed_data)



# Loading speed, angle info of training set.
data_dictionary = {}
camera_image_size = 100
# v = (v-min)/(max-min)
direction_data = []
speed_data = []
# we are reading data file to extract information...
# each line contains image_number (current direction angle) (current angle normalize) (current_speed) ()
road_name = "Road4"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type

data_file = open("../DataCollector/"+location_name+"/data.dat", "r")
# angle and speed information retrieved...
for line in data_file:
    f = line.split(" ")
    if len(f) == 5:
        data_dictionary[f[0]] = f
        direction_data.append(int(f[1]))
        speed_data.append(int(f[3]))

# Statistics of speed, direction angle of training set...
direction_mean = np.mean(direction_data)
direction_std = np.std(direction_data)
direction_max = np.max(direction_data)
direction_min = np.min(direction_data)

speed_mean = np.mean(speed_data)
speed_std = np.std(speed_data)
speed_max = np.max(speed_data)
speed_min = np.min(speed_data)

# Converting string to integer...
for img in data_dictionary.keys():
    data_dictionary[img][1] = int(data_dictionary[img][1])# - direction_mean)\
                              #/ (direction_max -direction_min)
    data_dictionary[img][3] = int(data_dictionary[img][3]) #- speed_mean) / (speed_max-speed_min)


for img in validation_data_dictionary.keys():
    validation_data_dictionary[img][1] = int(validation_data_dictionary[img][1])# - direction_mean)\
                              #/ (direction_max-direction_min)
    validation_data_dictionary[img][3] = int(validation_data_dictionary[img][3])# - speed_mean) / (speed_max-speed_min)



data_file.close()


# Loading validation set images...
road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type
camera_image_size = 100

validation_image_set = []
validation_information_set = []
validation_information_set2 = []
validation_whole_image_set = []

# we are loading images...
validation_image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in validation_image_file_list:
    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[:-1*len("Resized.png")]
        image_file = cv2.imread("../DataCollector/"+location_name+"/"+dosya)

        validation_whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))

        validation_information_set.append([validation_data_dictionary[dosya_number][1], validation_data_dictionary[dosya_number][3]])

validation_whole_image_set = np.array(validation_whole_image_set, dtype=np.float32)






# Loading training set images...

information_set = []
information_set2 = []
whole_image_set = []
# we are loading images...
road_name = "Road4"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type

image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in image_file_list:
    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[: -1*len("Resized.png")]

        image_file = cv2.imread("../DataCollector/"+location_name+"/" + dosya)
        # we are loading RGB image (3 channel)
        whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))
        # image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # angle, speed

        information_set.append([data_dictionary[dosya_number][1], data_dictionary[dosya_number][3]])


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

# Normalizing validation and training set...

mean_image = whole_image_set.mean(axis=0)
std_image = whole_image_set.std(axis=0)

max_image = whole_image_set.max(axis=0)
min_image = whole_image_set.min(axis=0)

diff = max_image - min_image

for i in range(len(diff)):
    if abs(diff[i]) < 0.0000001:
        diff[i] = max_image[i]

whole_image_set = (whole_image_set - mean_image)
whole_image_set = whole_image_set/(diff)

validation_whole_image_set = (validation_whole_image_set-mean_image)/diff


image_set = [np.reshape(whole_image_set[index], (30, 1000))
             for index in range(0, whole_image_set.shape[0])]

validation_image_set = [np.reshape(validation_whole_image_set[index], (30, 1000)) for index in range(0, validation_whole_image_set.shape[0])]


# direction_max, direction_min, direction_mean, direction_std, speed_max,
# speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min =
# pickle.load(open("important_data.dat","r"))

# we are saving statistical information...
pickle.dump([np.max(direction_data), np.min(direction_data),
             direction_mean, direction_std, np.max(speed_data),
             np.min(speed_data), speed_mean, speed_std, mean_image,
             std_image, whole_image_set.max(),
             whole_image_set.min(), diff], open("RNNRegressionRGBTrainer("+road_name+")_important_data.dat", "w"))

print "Data have been loaded"

# RNN architecture definition
# 30 times unfolded, and each input size is 1000
camera_image = tf.placeholder(tf.float32, shape=[None, 30, 1000])
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

optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
optimize = optimizer.minimize(summation)


train_saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()



road_name = "Road4.2"
smallest_validation_error = 10000000000

error_graph_file_name = "RNNRegressionRGB"+road_name+"_graph.dat"


iteration_axis = []
error_axis = []
validation_error_axis = []



# sometimes, we may need to continue training for previously trained model...
if len([file for file in os.listdir("Trained_NN") if file.__contains__("RNNRegressionRGB"+road_name)]) > 0:
    train_saver.restore(sess, "Trained_NN/RNNRegressionRGB"+road_name)
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
    print "samlle : ", smallest_validation_error
else:
    error_graph_file = open(error_graph_file_name, "w")
    sess.run(init)
    error_iteration_number = 1



iteration_count = 40
while iteration_count != 0:
    # we are starting with 100 iteration at the beginning...
    # after this is completed, the user can continue with new iteration number if he/she wants...

    for iteration in range(iteration_count):
        print "**********Iteration "+str(iteration)+"****************"

        l = 0
        # we are performing 30 sample batch learning...
        for bacht_index in range(0, len(information_set), 30):
            _, loss_value = sess.run([optimize, summation],
                                     feed_dict={camera_image: image_set[bacht_index: bacht_index+30],
                                                target_values: information_set[bacht_index:bacht_index+30]})
            l += loss_value

        j = 0
        for bacht_index in range(0, len(validation_information_set), 30):
            # we are training with minibatch learning...
            loss_value = sess.run(summation,
                                  feed_dict={camera_image: validation_image_set[bacht_index: bacht_index + 30],
                                             target_values: validation_information_set[bacht_index:bacht_index + 30],
                                             })

            j += loss_value

        if j < smallest_validation_error:
            smallest_validation_error = j
            print "Smallest error : ", smallest_validation_error
            print "Model has been saved..."
            train_saver.save(sess, "Trained_NN/RNNRegressionRGB" + road_name)

        iteration_axis.append(error_iteration_number)
        error_axis.append(l)
        validation_error_axis.append(j)
        error_graph_file.write(str(error_iteration_number) + " " + str(l) + " " + str(j) + "\n")
        error_iteration_number += 1
        if l < 0.78:
            break
        # we are also displaying the train data set error...
        print "Loss : ", l, " Validation : ", j
    plt.subplot(211)
    plt.plot(iteration_axis, error_axis, label="Error")
    plt.subplot(212)
    plt.plot(iteration_axis, validation_error_axis, label="Error")
    plt.show()

    # user specifies new iteration number after previous one finishes...
    print "Please enter new iteration count "
    iteration_count = int(raw_input())

error_graph_file.close()

print "Train has been terminated..."

