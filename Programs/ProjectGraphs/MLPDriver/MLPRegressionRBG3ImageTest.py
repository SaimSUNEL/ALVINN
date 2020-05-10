import tensorflow as tf
import cv2
import os
import numpy as np
import pickle
from tensorflow.python.client import device_lib
import math
import matplotlib.pyplot as plt

# We are loading statistics file of the architecture
# We had created this file during training...
# Mean image and difference image are required, because we have to normalize test set images...


root_directory = "../../MLPDriver/"
road_name = "Road4"
direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = pickle.load(open(root_directory+"MLPRegressionRGB3ImagesTrainer("+road_name+")_important_data.dat","r"))


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



data_file.close()
# Converting string to integer...

for img in validation_data_dictionary.keys():
    validation_data_dictionary[img][1] = int(validation_data_dictionary[img][1])# - direction_mean)\
                             # / (direction_max-direction_min)
    validation_data_dictionary[img][3] = int(validation_data_dictionary[img][3])# - speed_mean) / (speed_max-speed_min)





validation_image_set = []
validation_information_set = []
validation_information_set2 = []
validation_whole_image_set = []
road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type


# loading test set images....
validation_image_file_list = os.listdir("../../DataCollector/"+location_name)
validation_image_file_list = [ int(a[:-1*len("Resized.png")]) for a in validation_image_file_list if a.__contains__("Resized") and not a.__contains__("GRAY")]
# images should be ordered... because two consequence images will be packed...
validation_image_file_list.sort()

print validation_image_file_list
# Stacking 3 consequtive images into one for architecture...

for dosya in range(2, len(validation_image_file_list)):

    # current and previous images are packed...
    dosya_number = dosya - 1 #+89
    dosya_number2 = dosya #+89

    dosya_number3 = dosya - 2

    image_file = cv2.imread("../../DataCollector/"+location_name+"/"+str(dosya_number)+"Resized.png")
    image_file = image_file

    image_file2 = cv2.imread("../../DataCollector/"+location_name+"/"+str(dosya_number2)+"Resized.png")
    image_file2 = image_file2
    image_file3 = cv2.imread("../../DataCollector/"+location_name+"/"+str(dosya_number3)+"Resized.png")
    image_file3 = image_file3

    combination = np.concatenate((np.reshape(image_file2, (1, (camera_image_size**2)*3)),
                                  np.reshape(image_file, (1, (camera_image_size**2)*3)),
                                  np.reshape(image_file3, (1, (camera_image_size ** 2)*3)))).reshape(90000)

    validation_whole_image_set.append(combination)

    # the corresponding target information(speed and direction) is added to target set
    # information_set.append(data_dictionary[str(dosya_number2)][1])# , int(data_dictionary[dosya_number][3])])
    # information_set2.append(data_dictionary[str(dosya_number2)][3])

    validation_information_set.append([validation_data_dictionary[str(dosya_number2)][1],
                            validation_data_dictionary[str(dosya_number2)][3]])


validation_whole_image_set = np.array(validation_whole_image_set, dtype=np.float32)



# Normalizing test images....

validation_whole_image_set = (validation_whole_image_set-mean_image)/diff
# reshaping test images for MLP
validation_image_set = [validation_whole_image_set[index] for index in range(0, validation_whole_image_set.shape[0])]

# definition of trained model....


camera_image = tf.placeholder(tf.float32, [None, (camera_image_size**2)*9], "camera_image")
target_values = tf.placeholder(tf.float32, [None, 2], "target_values")

# first layer

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

road_name = "Road4"
# Load trained model...

train_saver.restore(sess, root_directory+"Trained_NN/MlpRegressionRGB3Images"+road_name)


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
dosya.write("MLPRegressionRGB3ImageTest %f" % j + "\n")
dosya.close()

print "Loss : ", j
