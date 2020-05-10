import tensorflow as tf
import cv2
import os
import numpy as np
import math
import pickle
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

# We are loading statistics file of the architecture
# We had created this file during training...
# Mean image and difference image are required, because we have to normalize test set images...

root_directory = "../../MLPDriver/"
road_name = "Road4"
direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = pickle.load(open(root_directory+"MLPRegressionRGBTrainer("+road_name+")_important_data.dat","r"))


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
    validation_data_dictionary[img][1] = int(validation_data_dictionary[img][1]) #- direction_mean)\
                              #/ (direction_max-direction_min)
    validation_data_dictionary[img][3] = int(validation_data_dictionary[img][3]) #- speed_mean) / (speed_max-speed_min)


# loading test set images....

road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type
camera_image_size = 100

validation_image_set = []
validation_information_set = []
validation_information_set2 = []
validation_whole_image_set = []

# we are loading images...
validation_image_file_list = os.listdir("../../DataCollector/"+location_name)
for dosya in validation_image_file_list:
    if dosya.__contains__("Resized") and not dosya.__contains__("GRAY"):
        dosya_number = dosya[:-1*len("Resized.png")]
        image_file = cv2.imread("../../DataCollector/"+location_name+"/"+dosya)
        image_file = image_file
        validation_whole_image_set.append(np.reshape(image_file, (camera_image_size**2)*3))

        validation_information_set.append([validation_data_dictionary[dosya_number][1], validation_data_dictionary[dosya_number][3]])

validation_whole_image_set = np.array(validation_whole_image_set, dtype=np.float32)



# Normalizing test images....
validation_whole_image_set = (validation_whole_image_set-mean_image)/diff



# reshaping test images for MLP

validation_image_set = [validation_whole_image_set[index] for index in range(0, validation_whole_image_set.shape[0])]


# definition of trained model....

camera_image = tf.placeholder(tf.float32, [None, (camera_image_size**2)*3], "camera_image")
target_values = tf.placeholder(tf.float32, [None, 2], "target_values")

# first layer

initrange = math.sqrt(2.0/((camera_image_size**2)*3))

w1 = tf.get_variable("w1", ((camera_image_size**2)*3, 352), initializer=tf.random_uniform_initializer(-initrange, initrange))
b1 = tf.Variable(tf.zeros(352), name="b1")

layer_1_output = tf.nn.tanh(tf.matmul(camera_image, w1) + b1)

initrange = math.sqrt(2.0/(352))
w4 = tf.get_variable("w4", (352, 2), initializer=tf.random_uniform_initializer(-initrange, initrange))
b4 = tf.Variable(tf.zeros(2), name="b4")

final_output = tf.matmul(layer_1_output, w4) + b4

loss_ = (1/30.0)*tf.squared_difference(final_output, target_values)
summation = tf.reduce_sum(loss_)

loss_sum = tf.summary.scalar("Lost", summation)


train_saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()




road_name = "Road4"
# Load trained model...

train_saver.restore(sess, root_directory+"Trained_NN/MlpRegressionRGB"+road_name)


# Calculating test error....
j = 0
for bacht_index in range(0, len(validation_information_set), 30):
    loss_value = sess.run(summation,
                          feed_dict={camera_image: validation_image_set[bacht_index: bacht_index + 30],
                                     target_values: validation_information_set[bacht_index:bacht_index + 30],
                                     })



    j += loss_value
# Displaying test error and storing it to a file for comparison with other architectures...

dosya = open("../TestResult"+road_name+".txt", "a")
dosya.write("MLPRegressionRGBTest %f" % j + "\n")
dosya.close()

print "Loss : ", j
