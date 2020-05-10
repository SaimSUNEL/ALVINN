import tensorflow as tf
import numpy as np
import os
import cv2
import pickle
import math

if not os.path.isdir("Trained_NN"):
    os.mkdir("Trained_NN")

data_dictionary = {}
camera_image_size = 100
# v = (v-min)/(max-min)
direction_data = []
speed_data = []
data_file = open("../DataCollector/VehicleDataRoad3/data.dat", "r")
# angle and speed information retrieved...
for line in data_file:
    f = line.split(" ")
    if len(f) == 5:
        data_dictionary[f[0]] = f
        direction_data.append(int(f[1]))
        speed_data.append(int(f[3]))

direction_mean = np.mean(direction_data)

direction_std = np.std(direction_data)

speed_mean = np.mean(speed_data)
speed_std = np.std(speed_data)

for img in data_dictionary.keys():
    data_dictionary[img][1] = (int(data_dictionary[img][1])+6)  # - direction_mean)/direction_std
    data_dictionary[img][3] = (int(data_dictionary[img][3])-5)  # - speed_mean) / speed_std

data_file.close()

information_set = []
information_set2 = []
whole_image_set = []
image_number = 0
image_file_list = os.listdir("../DataCollector/VehicleDataRoad3")
np.random.shuffle(image_file_list)
for dosya in image_file_list:
    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[: -1*len("Resized.png")]

        image_file = cv2.imread("../DataCollector/VehicleDataRoad3/" +
                                dosya)

        whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))
        # image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # angle, speed
        information_set.append(data_dictionary[dosya_number][1])  # , int(data_dictionary
        # [dosya_number][3])])
        information_set2.append(data_dictionary[dosya_number][3])
        image_number += 1





whole_image_set = np.array(whole_image_set, dtype=np.float32)

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
image_set = [np.reshape(whole_image_set[index], (3, 10000))
             for index in range(0, whole_image_set.shape[0])]
# direction_max, direction_min, direction_mean, direction_std, speed_max,
# speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min =
# pickle.load(open("important_data.dat","r"))

pickle.dump([np.max(direction_data), np.min(direction_data),
             direction_mean, direction_std, np.max(speed_data),
             np.min(speed_data), speed_mean, speed_std, mean_image,
             std_image, whole_image_set.max(),
             whole_image_set.min(), diff], open("RNNClassificationRGBTrainer_important_data.dat", "w"))
print "max : ", whole_image_set.max()
print "min : ", whole_image_set.min()

print "Data have been loaded"

# 100 input 100 stepss...
camera_image = tf.placeholder(tf.float32, shape=[None, 3, 10000])
target_values = tf.placeholder(tf.int32, shape=[None])
target_values2 = tf.placeholder(tf.int32, shape=[None])

basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=50)
outputs, states = tf.nn.dynamic_rnn(basic_cell, camera_image, dtype=tf.float32)

input_number = 50
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w1 = tf.get_variable("w1", (50, 50), initializer=initializer)
b1 = tf.Variable(tf.zeros(50), name="b1")
fc_output = tf.nn.leaky_relu(tf.matmul(states[1], w1)+b1)

input_number = 50
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w2 = tf.get_variable("w2", (50, 25), tf.float32, initializer)
b2 = tf.get_variable("b2", [25], tf.float32, tf.constant_initializer(0))


fc2_output = tf.nn.leaky_relu(tf.matmul(fc_output, w2)+b2)

input_number = 25
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w3 = tf.get_variable("w3", (25, 32), tf.float32, initializer)
b3 = tf.get_variable("b3", [32], tf.float32, tf.constant_initializer(0))

final_output = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(fc2_output, w3) + b3))

input_number = 25
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W4 = tf.get_variable("W4", (25, 25), tf.float32, initializer)
b4 = tf.get_variable("b4", [25], tf.float32, tf.constant_initializer(0))

final_output2 = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(fc2_output, W4) + b4))

one_hot = tf.one_hot(target_values, 32, on_value=1.0, off_value=0.0, dtype=tf.float32)
one_hot2 = tf.one_hot(target_values2, 25, on_value=1.0, off_value=0.0, dtype=tf.float32)

loss_s = one_hot2 * tf.log(final_output2)
entropys = -tf.reduce_sum(loss_s, axis=1)
entropy2s = tf.reduce_sum(entropys)

loss_ = one_hot * tf.log(final_output)
entropy = -tf.reduce_sum(loss_, axis=1)
entropy2 = tf.reduce_sum(entropy)
finn = entropy2 + entropy2s

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
optimize = optimizer.minimize(finn)
train_saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()

if len([file_ for file_ in os.listdir("Trained_NN") if file_.__contains__("RNNClassificationRGB")]) > 0:
    # train_saver.restore(sess, "Trained_NN/RNNClassificationRGB")
    sess.run(init)
    print "Pretrained model has been loaded and will be continued to train..."

else:
    sess.run(init)


bacht_index = 0
fin = diff = loss_value = None

iteration_count = 10
while iteration_count != 0:

    for iteration in range(iteration_count):
        print "**********Iteration "+str(iteration)+"****************"

        l1 = 0
        for bacht_index in range(0, len(information_set), 30):
            _, loss_value = sess.run([optimize, finn],
                                     feed_dict={camera_image: image_set[bacht_index: bacht_index+30],
                                                target_values: information_set[bacht_index:bacht_index+30],
                                                target_values2: information_set2[bacht_index:bacht_index+30]})
            l1 += loss_value
        print "Loss : ", l1

    print "Please enter new iteration count "
    iteration_count = int(raw_input())


print "Train has been terminated..."
train_saver.save(sess, "Trained_NN/RNNClassificationRGB")
