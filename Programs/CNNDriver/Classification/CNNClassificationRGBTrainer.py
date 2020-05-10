import tensorflow as tf
import numpy as np
import os
import cv2
import pickle
import math
import matplotlib.pyplot as plt

# the directory to which trained model is saved should exist...
if not os.path.isdir("Trained_NN"):
    os.mkdir("Trained_NN")


# Validation data set is being loaded....

road_name = "Road4"
data_type = "Test"
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


validation_direction_mean = np.mean(validation_direction_data)
validation_direction_std = np.std(validation_direction_data)
validation_direction_max = np.max(validation_direction_data)
validation_direction_min = np.min(validation_direction_data)

validation_speed_mean = np.mean(validation_speed_data)
validation_speed_std = np.std(validation_speed_data)
validation_speed_max = np.max(validation_speed_data)
validation_speed_min = np.min(validation_speed_data)





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

direction_mean = np.mean(direction_data)
direction_max = np.max(direction_data)
direction_min = np.min(direction_data)

direction_std = np.std(direction_data)

speed_mean = np.mean(speed_data)
speed_max = np.max(speed_data)
speed_min = np.min(speed_data)

speed_std = np.std(speed_data)
# We are performing classification on speed and direction
# Each speed value corresponds a node in output...
# For simplicty, we are shifting the values to start it from 0...
# While choosing the correct angle, we are subtracting these values from final output indexes...

direction_min = min(direction_min, validation_direction_min)
speed_min = min(speed_min, validation_speed_min)




for img in data_dictionary.keys():
    data_dictionary[img][1] = (int(data_dictionary[img][1])-direction_min)  # - direction_mean)/direction_std
    data_dictionary[img][3] = (int(data_dictionary[img][3])-speed_min)  # - speed_mean) / speed_std

# Validation data target normalization
for img in validation_data_dictionary.keys():
    validation_data_dictionary[img][1] = (int(validation_data_dictionary[img][1])-direction_min)# - direction_mean)/direction_std
    validation_data_dictionary[img][3] = (int(validation_data_dictionary[img][3])-speed_min)# - speed_mean) / speed_std



data_file.close()




validation_information_set = []
validation_information_set2 = []
validation_whole_image_set = []
road_name = "Road4"
data_type = "Test"
location_name = "VehicleData"+road_name+data_type

# we are loading images...
validation_image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in validation_image_file_list:
    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[: -1*len("Resized.png")]

        image_file = cv2.imread("../DataCollector/"+location_name+"/" + dosya)
        # we are loading RGB image (3 channel)
        validation_whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))
        # image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # angle, speed

        # current image's corresponding target values are loaded to target set...
        validation_information_set.append(data_dictionary[dosya_number][1])  # , int(data_dictionary
        # [dosya_number][3])])
        validation_information_set2.append(data_dictionary[dosya_number][3])

validation_whole_image_set = np.array(validation_whole_image_set, dtype=np.float32)










information_set = []
information_set2 = []
whole_image_set = []
road_name = "Road4"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type

# we are loading images...
image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in image_file_list:
    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[: -1*len("Resized.png")]

        image_file = cv2.imread("../DataCollector/"+location_name+"/" + dosya)
        # we are loading RGB image (3 channel)
        whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))
        # image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # angle, speed

        # current image's corresponding target values are loaded to target set...
        information_set.append(data_dictionary[dosya_number][1])  # , int(data_dictionary
        # [dosya_number][3])])
        information_set2.append(data_dictionary[dosya_number][3])

whole_image_set = np.array(whole_image_set, dtype=np.float32)

# To eliminate correlation between samples, we are shuffling the data set
indices = [a for a in range(len(whole_image_set)-1)]
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

# statistical information of dataset is stored in a file, these information
# is later loaded by driver programs...


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


image_set = [np.reshape(whole_image_set[index], (100, 100, 3))
             for index in range(0, whole_image_set.shape[0])]

validation_image_set = [np.reshape(validation_whole_image_set[index], (100, 100, 3)) for index in range(0, validation_whole_image_set.shape[0])]

direction_max = max(direction_max, validation_direction_max)
speed_max = max(speed_max, validation_speed_max)
speed_min = min(speed_min, validation_speed_min)

# direction_max, direction_min, direction_mean, direction_std, speed_max,
# speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min =
# pickle.load(open("important_data.dat","r"))

# we are saving statistical information...
pickle.dump([direction_max, direction_min,
             direction_mean, direction_std, speed_max,
             speed_min, speed_mean, speed_std, mean_image,
             std_image, whole_image_set.max(),
             whole_image_set.min(), diff], open("CNNClassificationRGBTrainer("+road_name+")_important_data.dat", "w"))

print "Data have been loaded"

# we are using a 3 channel (RGB) 100x100 image as input...
camera_image = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
# direction target values will be fed to network
target_values = tf.placeholder(tf.int32, shape=[None])
# speed target values will be fed to network via this placeholder...
target_values2 = tf.placeholder(tf.int32, shape=[None])


# in the first convolution layer, we are using 3x3 filter and it these filters
# will produce 64 layer future map...

input_number = 3 * 3 * 3
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W = tf.get_variable("W", (3, 3, 3, 64), tf.float32, initializer)
# bias weights of first convolutional layer...
b = tf.get_variable("b", [64], tf.float32, tf.constant_initializer(0))
# leaky relu activation function is used for first convolution layer...
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

# Second convolutional layer's activation fucntion leaky relu function...
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

# for direction output we are constructing a 32 node output layer...

direction_size = direction_max - direction_min

input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W4 = tf.get_variable("W4", (100, direction_size), tf.float32, initializer)
b4 = tf.get_variable("b4", [direction_size], tf.float32, tf.constant_initializer(0))
# we are applying softmax to direction output...
final_output = tf.nn.softmax((tf.matmul(f1_output, W4) + b4))

speed_size = speed_max - speed_min

# for speed output we are constructing a 25 node output layer...
input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W5 = tf.get_variable("W5", (100, speed_size), tf.float32, initializer)
b5 = tf.get_variable("b5", [speed_size], tf.float32, tf.constant_initializer(0))
# we are applying softmax for output of speed output...
final_output2 = tf.nn.softmax((tf.matmul(f1_output, W5) + b5))

global_step = tf.Variable(0, name="global_step", trainable=False)



# we are performing classification so for both speed and direction targets we need construct one hot vector...

one_hot = tf.one_hot(target_values, direction_size, on_value=1.0, off_value=0.0, dtype=tf.float32)
one_hot2 = tf.one_hot(target_values2, speed_size, on_value=1.0, off_value=0.0, dtype=tf.float32)
# we are calculating speed information error with cross entropy loss

loss_s = one_hot2 * tf.log(final_output2)
entropys = -tf.reduce_sum(loss_s, axis=1)
entropy2s = tf.reduce_sum(entropys)

# we are calculating direction information error...
loss_ = one_hot * tf.log(final_output)
entropy = -tf.reduce_sum(loss_, axis=1)
entropy2 = tf.reduce_sum(entropy)
# The total error is summation of direction and speed errors...
finn = entropy2 + entropy2s

# We are using adam optimer...
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
optimize = optimizer.minimize(finn)
# we will save the trained network to file...
train_saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()


error_graph_file_name = "CNNClassificationRGB"+road_name+"_graph.dat"


iteration_axis = []
error_axis = []
validation_error_axis = []




# sometimes, we may need to continue training for previously trained model...
if len([file for file in os.listdir("Trained_NN") if file.__contains__("CNNClassificationRGB"+road_name)]) > 0:
    train_saver.restore(sess, "Trained_NN/CNNClassificationRGB"+road_name)
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
    print "Pretrained model has been loaded and will be continued to train..."

else:
    error_graph_file = open(error_graph_file_name, "w")
    sess.run(init)
    error_iteration_number = 1




iteration_count = 10
while iteration_count != 0:
    # we are starting with 100 iteration at the beginning...
    # after this is completed, the user can continue with new iteration number if he/she wants...

    for iteration in range(iteration_count):
        print "**********Iteration "+str(iteration)+"****************"

        l = 0
        # we are performing 30 sample batch learning...
        for bacht_index in range(0, len(information_set), 30):
            _, loss_value = sess.run([optimize, finn],
                                     feed_dict={camera_image: image_set[bacht_index: bacht_index+30],
                                                target_values: information_set[bacht_index:bacht_index+30],
                                                target_values2: information_set2[bacht_index:bacht_index+30]})
            l += loss_value

        j = 0
        s = 0
        for bacht_index in range(0, len(validation_information_set), 30):
            # we are training with minibatch learning...
            loss_value, dir_output, spe_output = sess.run([finn, final_output, final_output2],
                                                          feed_dict={camera_image: validation_image_set[
                                                                                   bacht_index: bacht_index + 30],
                                                                     target_values: validation_information_set[
                                                                                    bacht_index:bacht_index + 30],
                                                                     target_values2: validation_information_set2[
                                                                                     bacht_index:bacht_index + 30]})

            s += loss_value
            res_dir = np.argmax(dir_output, axis=1)
            # print "Dir : ", res_dir
            res_spe = np.argmax(spe_output, axis=1)
            # print "Speed : ", res_spe

            cor_dir = np.array(validation_information_set[bacht_index: bacht_index + 30], dtype=np.float32)
            cor_spe = np.array(validation_information_set2[bacht_index: bacht_index + 30])

            # print "Correct dir : ", validation_information_set[bacht_index: bacht_index+30]
            # print "Correct speed : ", validation_information_set2[bacht_index: bacht_index+30]
            l1 = np.sum(np.square(cor_dir - res_dir))
            l2 = np.sum(np.square(cor_spe - res_spe))
            # print "Los : ", l1+l2

            j += l1 + l2




            # we are also displaying the train data set error...
        print "Loss : ", l

        iteration_axis.append(error_iteration_number)
        error_axis.append(l)
        validation_error_axis.append(j)
        error_graph_file.write(str(error_iteration_number) + " " + str(l) + " " + str(j) + "\n")
        error_iteration_number += 1

    plt.plot(iteration_axis, error_axis, label="Error")
    plt.plot(iteration_axis, validation_error_axis, label="Error")
    plt.show()
    # user specifies new iteration number after previous one finishes...
    print "Please enter new iteration count "
    iteration_count = int(raw_input())


error_graph_file.close()
# We are saving the trained model to disk...
print "Train has been terminated..."
train_saver.save(sess, "Trained_NN/CNNClassificationRGB"+road_name)

