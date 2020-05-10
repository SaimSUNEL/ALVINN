import tensorflow as tf
import cv2
import os
import numpy as np
import pickle
from tensorflow.python.client import device_lib
import math
import matplotlib.pyplot as plt

# We have commented MLPRegressionTrainer.py in detailed manner,
# Here we neglect the most part to comment, because most of the parts are the same...
# In this file same neural network with MLPRegressionTrainer.py is used..
# Only difference is that the input is 3 RGB images, 90000 input layer...





# the directory to which trained model is saved should exist...
if not os.path.isdir("Trained_NN"):
    os.mkdir("Trained_NN")

# Validation speed, direction angle are being loaded....

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

# Statistics of speed, direction angle of validation set...
validation_direction_mean = np.mean(validation_direction_data)
validation_direction_std = np.std(validation_direction_data)
validation_direction_max = np.max(validation_direction_data)
validation_direction_min = np.min(validation_direction_data)

validation_speed_mean = np.mean(validation_speed_data)
validation_speed_std = np.std(validation_speed_data)
validation_speed_max = np.max(validation_speed_data)
validation_speed_min = np.min(validation_speed_data)




# Training speed, direction angle are being loaded....

data_dictionary = {}
camera_image_size = 100
# v = (v-min)/(max-min)
direction_data = []
speed_data = []
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

# Statistics of speed, direction angle of training set....
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



# Converting string to integer...
for img in data_dictionary.keys():
    data_dictionary[img][1] = int(data_dictionary[img][1]) # - direction_mean)\
                              #/ (direction_max -direction_min)
    data_dictionary[img][3] = int(data_dictionary[img][3]) #- speed_mean) / (speed_max-speed_min)

# Validation data target normalization
for img in validation_data_dictionary.keys():
    validation_data_dictionary[img][1] = int(validation_data_dictionary[img][1])#-direction_min)# - direction_mean)/direction_std
    validation_data_dictionary[img][3] = int(validation_data_dictionary[img][3])#-speed_min)# - speed_mean) / speed_std

data_file.close()


# Validation set images are being loaded...
validation_information_set = []
validation_information_set2 = []
validation_whole_image_set = []
road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type

validation_image_number = 0
validation_image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in validation_image_file_list:
    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[: -1*len("Resized.png")]

        image_file = cv2.imread("../DataCollector/"+location_name+"/" + str(validation_image_number) + "Resized.png")

        validation_whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))
        # image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # angle, speed

        validation_information_set.append([validation_data_dictionary[str(validation_image_number)][1], validation_data_dictionary[str(validation_image_number)][3]])

        validation_image_number += 1

validation_whole_image_set = np.array(validation_whole_image_set, dtype=np.float32)


# Training set images are being loaded...

information_set = []
information_set2 = []
whole_image_set = []
road_name = "Road4"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type

image_number = 0
image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in image_file_list:
    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[: -1*len("Resized.png")]

        image_file = cv2.imread("../DataCollector/"+location_name+"/" + str(image_number) + "Resized.png")

        whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))
        # image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # angle, speed

        information_set.append([data_dictionary[str(image_number)][1], data_dictionary[str(image_number)][3]])

        image_number += 1

whole_image_set = np.array(whole_image_set, dtype=np.float32)


# Training and validation set normalizations...

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



image_set = [np.reshape(whole_image_set[index], (100*100*3))
             for index in range(0, whole_image_set.shape[0])]
validation_image_set = [np.reshape(validation_whole_image_set[index], (100*100*3)) for index in range(0, validation_whole_image_set.shape[0])]
direction_max = max(direction_max, validation_direction_max)
speed_max = max(speed_max, validation_speed_max)
speed_min = min(speed_min, validation_speed_min)


# direction_max, direction_min, direction_mean, direction_std, speed_max,
# speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min =
# pickle.load(open("important_data.dat","r"))

pickle.dump([direction_max, direction_min,
             direction_mean, direction_std, speed_max,
             speed_min, speed_mean, speed_std, mean_image,
             std_image, whole_image_set.max(),
             whole_image_set.min(), diff], open("MLPRegressionRGB3ImageTrainer("+road_name+")_important_data.dat", "w"))
print "max : ", whole_image_set.max()
print "min : ", whole_image_set.min()

print "Data have been loaded"
exit()
# 3 sequential images of validation set are stacked into one...

validation_new_image_set = []
validation_new_information_set = []
validation_new_information_set2 = []
for image_index in range(2, len(validation_image_set)):
    combinatination = np.array([validation_image_set[image_index], validation_image_set[image_index-1], validation_image_set[image_index-2]], dtype=np.float32)
    combinatination = np.reshape(combinatination, newshape=(90000))
    validation_new_image_set.append(combinatination)

    validation_new_information_set.append(validation_information_set[image_index])



validation_image_set = np.array(validation_new_image_set)
validation_information_set = validation_new_information_set





# 3 sequential images of training set are stacked into one...


new_image_set = []
new_information_set = []
new_information_set2 = []
for image_index in range(2, len(image_set)):
    combinatination = np.array([image_set[image_index], image_set[image_index-1], image_set[image_index-2]], dtype=np.float32)
    combinatination = np.reshape(combinatination, newshape=(90000))
    new_image_set.append(combinatination)

    new_information_set.append(information_set[image_index])



image_set = np.array(new_image_set)
information_set = new_information_set







# To eliminate correlation between samples, we are shuffling the data set
indices = [a for a in range(len(image_set))]
np.random.shuffle(indices)

# We are shuffling dataset, and the target values also must be shuffled as it is done to data set...
image_set = image_set[indices]
gf = []
df = []
for h in indices:
    gf.append(information_set[h])

information_set = gf;

print "Data have been loaded"

# Input layer is 90000 nodes
camera_image = tf.placeholder(tf.float32, [None, (camera_image_size**2)*9], "camera_image")
target_values = tf.placeholder(tf.float32, [None, 2], "target_values")

initrange = math.sqrt(2.0/(camera_image_size**2)*9)

w1 = tf.get_variable("w1", ((camera_image_size**2)*9, 352), initializer=tf.random_uniform_initializer(-initrange, initrange))
b1 = tf.Variable(tf.zeros(352), name="b1")

layer_1_output = tf.nn.tanh(tf.matmul(camera_image, w1) + b1)

initrange = math.sqrt(2.0/(352))
w4 = tf.get_variable("w4", (352, 2), initializer=tf.random_uniform_initializer(-initrange, initrange))
b4 = tf.Variable(tf.zeros(2), name="b4")

final_output = tf.matmul(layer_1_output, w4) + b4

loss_ = (1/30.0)*tf.squared_difference(final_output, target_values)
summation = tf.reduce_sum(loss_)

loss_sum = tf.summary.scalar("Lost", summation)


# we are using adam optimizer for adaptive learning rate...
optimizer = tf.train.AdamOptimizer(learning_rate = 0.000001)
optimize = optimizer.minimize(summation)

# To save our trained network to file ...
train_saver = tf.train.Saver()
# session creation....
sess = tf.Session()
# before running the graph all the variables must be initialized....
init = tf.global_variables_initializer()



road_name = "Road4.2"
smallest_validation_error = 10000000000

error_graph_file_name = "MlpRegressionRGB3Images"+road_name+"_graph.dat"


iteration_axis = []
error_axis = []
validation_error_axis = []





# sometimes trained network is trained for better accuracy...
# currently it does not load previously trained network...
# we are initializing the network...
if len([file for file in os.listdir("Trained_NN") if file.__contains__("MlpRegressionRGB3Images"+road_name)]) >0:
    train_saver.restore(sess, "Trained_NN/MlpRegressionRGB3Images"+road_name)
    # sess.run(init)
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
    print "Smallest error : ", smallest_validation_error




    print "Pretrained model has been loaded and will be continued to train..."

else:
    error_graph_file = open(error_graph_file_name, "w")
    sess.run(init)
    error_iteration_number = 1


# Training the nn...
iteration_count = 10

while iteration_count != 0:


    for iteration in range(iteration_count):
        print "**********Iteration "+str(iteration)+"****************"

        l = 0
        # we are training our network with 30 sample batches..
        for bacht_index in range(0, len(information_set), 30):
            # we are training with minibatch learning...
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
            train_saver.save(sess, "Trained_NN/MlpRegressionRGB3Images" + road_name)

        # we are displaying data set error...
        print "Loss : ", l, " Validation error : ", j
        iteration_axis.append(error_iteration_number)
        error_axis.append(l)
        validation_error_axis.append(j)
        error_graph_file.write(str(error_iteration_number) + " " + str(l) + " " + str(j) + "\n")
        error_iteration_number += 1
        if l < 0.78:
            break

    # After previous iteration count is exhausted, user enters a new iteration count to continue
    # training
    # if users enters 0 zero the training process is terminated
    plt.subplot(211)
    plt.plot(iteration_axis, error_axis, label="Error")
    plt.subplot(212)
    plt.plot(iteration_axis, validation_error_axis, label="Error")
    plt.show()

    print "Please enter new iteration count "
    iteration_count= int(raw_input())

error_graph_file.close()
print "Train has been terminated..."
