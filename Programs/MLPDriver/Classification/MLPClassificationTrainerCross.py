import tensorflow as tf
import cv2
import os
import numpy as np
import pickle
import math
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

# the directory to which trained model is saved should exist...
if not os.path.isdir("Trained_NN"):
    os.mkdir("Trained_NN")

data_dictionary = {}
camera_image_size = 100
# v = (v-min)/(max-min)
direction_data = []
speed_data = []

# we are reading data file to extract information...
# each line contains image_number (current direction angle) (current angle normalize) (current_speed) ()

road_name = "Road6"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type


data_file = open("../DataCollector/"+location_name+"/data.dat", "r")
for line in data_file:
    f = line.split(" ")
    if len(f) == 5:
        data_dictionary[f[0]] = f
        direction_data.append(int(f[1]))
        speed_data.append(int(f[3]))

direction_mean = np.mean(direction_data)
direction_std = np.std(direction_data)
direction_max = np.max(direction_data)
direction_min = np.min(direction_data)

speed_mean = np.mean(speed_data)
speed_std = np.std(speed_data)
speed_max = np.max(speed_data)
speed_min = np.min(speed_data)

print "Train max min direction - ", direction_min, " ", direction_max
print "Train max min speed - ", speed_min, " ", speed_max


# We are performing classification on speed and direction
# Each speed value corresponds a node in output...
# For simplicty, we are shifting the values to start it from 0...
# While choosing the correct angle, we are subtracting these values from final output indexes...

direction_min = direction_min
speed_min = speed_min


for img in data_dictionary.keys():
    data_dictionary[img][1] = (int(data_dictionary[img][1])-direction_min)# - direction_mean)/direction_std
    data_dictionary[img][3] = (int(data_dictionary[img][3])-speed_min)# - speed_mean) / speed_std

data_file.close()


road_name = "Road6"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type



image_set = []
information_set = []
information_set2 = []
whole_image_set = []

# we are loading images...
image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in image_file_list:
    if dosya.__contains__("GRAYResized"):
        dosya_number = dosya[:-1*len("GRAYResized.png")]
        image_file = cv2.imread("../DataCollector/"+location_name+"/"+dosya)
        image_file = image_file[:, :, 0]
        whole_image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # angle, speed
        # we also loading corresponding target values...
        information_set.append(data_dictionary[dosya_number][1])# , int(data_dictionary[dosya_number][3])])
        information_set2.append(data_dictionary[dosya_number][3])

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
    df.append(information_set2[h])
information_set = gf;
information_set2 = df;




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


image_set = [ whole_image_set[index] for index in range(0, whole_image_set.shape[0])]


# We are saving statistical information to a file for other applications...
pickle.dump([np.max(direction_data), np.min(direction_data),
                    direction_mean, direction_std, np.max(speed_data),
                    np.min(speed_data), speed_mean, speed_std, mean_image, std_image, whole_image_set.max(),
             whole_image_set.min(), diff], open("MLPClassificationTrainer("+road_name+")_important_data.dat","w"))


print "Data have been loaded"

# The input layer ...
# the input size is 10000(100x100)... grayscale image...
camera_image = tf.placeholder(tf.float32, [None, camera_image_size**2], "camera_image")

# these two places holders keep the true values of speed and direction information of current image
target_values = tf.placeholder(tf.int32, [None], "target_values")
target_values2 = tf.placeholder(tf.int32, [None], "target_values2")

input_ = camera_image_size**2
init_range = math.sqrt(3.0/input_)

# first hidden layer 10000 input - 50 first hidden layer...
w1 = tf.get_variable("w1", (camera_image_size**2, 50),
                     initializer=tf.random_uniform_initializer(-init_range, init_range))
# bias weights of first hidden layer...
b1 = tf.Variable(tf.zeros(50), name="b1")

# First hidden layer activation function is tanh...
layer_1_output = tf.nn.tanh(tf.matmul(camera_image, w1) + b1)


direction_size = int(np.max(direction_data) - np.min(direction_data))
direction_size = direction_max - direction_min

input_ = 50
init_range = math.sqrt(3.0/input_)
# Second hidden layer definition, 50 first hidden layer - 32 output layer for direction information...
w4 = tf.get_variable("w4", (50, direction_size), initializer=tf.random_uniform_initializer(-init_range, init_range))
# bias of output layer...
b4 = tf.Variable(tf.zeros(direction_size), name="b4")
speed_size = int(np.max(speed_data)-np.min(speed_data))
speed_size = int(speed_max - speed_min)

# Second hidden layer definition, 50 first hidden layer - 25 output layer for speed information..
w5 = tf.get_variable("w5", (50, speed_size), initializer=tf.random_uniform_initializer(-init_range, init_range))
b5 = tf.Variable(tf.zeros(speed_size), name="b5")


# final output for direction information, we are applying softmax...
final_output = tf.nn.softmax((tf.matmul(layer_1_output, w4)+b4))
# final output for speed information, we are applying softmax...
final_output2 = tf.nn.softmax((tf.matmul(layer_1_output, w5)+b5))

# because we are aproaching the problem as classification problem...
# so we need to represent target information as one hot vector...
# we are creating the related graph elements for tensorflow...
one_hot = tf.one_hot(target_values, direction_size, on_value=1.0, off_value=0.0, dtype=tf.float32)
one_hot2 = tf.one_hot(target_values2, speed_size, on_value=1.0, off_value=0.0, dtype=tf.float32)

# error for speed information, cross entropy is used...
loss_s = one_hot2 * tf.log(final_output2)
entropys = -tf.reduce_sum(loss_s, axis=1)
entropy2s = tf.reduce_sum(entropys)

# error for direction information, cross entropy loss is used...
loss_ = one_hot * tf.log(final_output)
entropy = -tf.reduce_sum(loss_, axis=1)
entropy2 = tf.reduce_sum(entropy)
# So the total error is the summation of speed and direction error...
finn = entropy2 + entropy2s



# we are using adam optimizer for adaptive learning rate...
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
optimize = optimizer.minimize(finn)

# To save our trained network to file ...
train_saver = tf.train.Saver()
# session creation....
sess = tf.Session()
# before running the graph all the variables must be initialized....
init = tf.global_variables_initializer()

error_graph_file_name = "MlpClassification"+road_name+"_graph.dat"


iteration_axis = []
error_axis = []
validation_error_axis = []



# sometimes trained network is trained for better accuracy...
# currently it does not load previously trained network...
# we are initializing the network...
if len([file for file in os.listdir("Trained_NN") if file.__contains__("MlpClassification"+road_name)]) >0:
    train_saver.restore(sess, "Trained_NN/MlpClassification"+road_name)
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

    error_iteration_number = max(iteration_axis)+1
    error_graph_file.close()
    error_graph_file = open(error_graph_file_name, "a")


    print "Pretrained model has been loaded and will be continued to train..."

else:
    error_graph_file = open(error_graph_file_name, "w")
    sess.run(init)
    error_iteration_number = 1


image_set = np.array(image_set, dtype=np.float32)
information_set = np.array(information_set, dtype=np.int32)
information_set2 = np.array(information_set2, dtype=np.int32)


for cross_index in range(5):

    bank_size = len(image_set)/5
    data_set_indices = [a for a in range(len(image_set))]
    test_indices = [f for f in data_set_indices if cross_index*bank_size<f< (cross_index+1)*bank_size]
    train_indices = [r for r in data_set_indices if not r in test_indices]


    train_data = image_set[train_indices]
    train_information_set = information_set[train_indices]
    train_information_set2 = information_set2[train_indices]

    test_data = image_set[test_indices]
    test_information_set = information_set[test_indices]
    test_information_set2 = information_set2[test_indices]



    iteration_count = 10

    # a
    while iteration_count != 0:

            # # We are performing 1000 iteration at beginning...
            for iteration in range(iteration_count):
                print "**********Iteration "+str(iteration)+"****************"

                l = 0
                BATCH_SIZE = 10
                # we are training our network with 30 sample batches..
                for bacht_index in range(0, len(train_information_set), BATCH_SIZE):
                    # we are training with minibatch learning...
                    _, loss_value = sess.run([optimize, finn],
                                             feed_dict={camera_image: train_data[bacht_index: bacht_index+BATCH_SIZE],
                                                        target_values: train_information_set[bacht_index:bacht_index+BATCH_SIZE],
                                                        target_values2: train_information_set2[bacht_index:bacht_index+BATCH_SIZE]})
                    l += loss_value

                j = 0
                for bacht_index in range(0, len(test_information_set), BATCH_SIZE):
                    # we are training with minibatch learning...
                    loss_value = sess.run(finn,
                                             feed_dict={camera_image: test_data[bacht_index: bacht_index + BATCH_SIZE],
                                                        target_values: test_information_set[
                                                                       bacht_index:bacht_index + BATCH_SIZE],
                                                        target_values2: test_information_set2[
                                                                        bacht_index:bacht_index + BATCH_SIZE]})
                    j += loss_value

                # we are displaying data set error...
                print "Loss : ", l
                iteration_axis.append(error_iteration_number)
                error_axis.append(l)
                validation_error_axis.append(j)
                error_graph_file.write(str(error_iteration_number)+" "+ str(l)+" "+ str(j) + "\n")
                error_iteration_number += 1


            # After previous iteration count is exhausted, user enters a new iteration count to continue
            # training
            # if users enters 0 zero the training process is terminated
            plt.plot(iteration_axis, error_axis, label="Error")
            plt.plot(iteration_axis, validation_error_axis, label="Error")
            plt.show()
            print "Please enter new iteration count "
            iteration_count= int(raw_input())


error_graph_file.close()

# Trained model is saved to disk...
print "Train has been terminated..."
train_saver.save(sess, "Trained_NN/MlpClassification"+road_name)