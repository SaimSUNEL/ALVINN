import tensorflow as tf
import cv2
import os
import numpy as np
import math
import pickle
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt


# We will save the trained model into this folder...
# If the folder does not exist, we create one...
if not os.path.isdir("Trained_NN"):
    os.mkdir("Trained_NN")


# We are reading the speed and direction information of the validation set...
# We are storing the data in arrays...
road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type
camera_image_size = 100

# This will hold all the information for the image...
# "image_number": [image number direction_angle normalized_angle speed normalized_speed]
validation_data_dictionary = {}
# Whole direction information will be hold in this to extract statistics of direction information...
validation_direction_data = []
# Whole speed information is stored in this to extract statistics of speed info..
validation_speed_data = []
# We are reading data from file...
data_file = open("../DataCollector/"+location_name+"/data.dat", "r")
for line in data_file:
    f = line.split(" ")
    if len(f) == 5:
        validation_data_dictionary[f[0]] = f
        validation_direction_data.append(int(f[1]))
        validation_speed_data.append(int(f[3]))

data_file.close()



# Statistics of validation direction data...
validation_direction_mean = np.mean(validation_direction_data)
validation_direction_std = np.std(validation_direction_data)
validation_direction_max = np.max(validation_direction_data)
validation_direction_min = np.min(validation_direction_data)

# Statistics of validation speed data...
validation_speed_mean = np.mean(validation_speed_data)
validation_speed_std = np.std(validation_speed_data)
validation_speed_max = np.max(validation_speed_data)
validation_speed_min = np.min(validation_speed_data)


# We are reading direction and speed information of training set...
# Whole data will be hold in this array...
data_dictionary = {}
camera_image_size = 100
# whole direction information of training set will be hold in this array...
direction_data = []
# Whole speed information of training set will be stored in this array...
speed_data = []
road_name = "Road4"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type
# We are reading whole information from file...
data_file = open("../DataCollector/"+location_name+"/data.dat", "r")
for line in data_file:
    f = line.split(" ")
    if len(f) == 5:
        data_dictionary[f[0]] = f
        direction_data.append(int(f[1]))
        speed_data.append(int(f[3]))




# Statistics of direction and speed information of training set
direction_mean = np.mean(direction_data)
direction_std = np.std(direction_data)
direction_max = np.max(direction_data)
direction_min = np.min(direction_data)

speed_mean = np.mean(speed_data)
speed_std = np.std(speed_data)
speed_max = np.max(speed_data)
speed_min = np.min(speed_data)


direction_max = max(direction_max, validation_direction_max)
direction_min = min(direction_min, validation_direction_min)


# When we read first data, it is string type, we are converting it to integer
# For training set....
for img in data_dictionary.keys():
    data_dictionary[img][1] = int(data_dictionary[img][1]) #- direction_mean)\
                              #/ (direction_max-direction_min)
    data_dictionary[img][3] = int(data_dictionary[img][3]) #- speed_mean) / (speed_max-speed_min)

data_file.close()

# For validation set....
for img in validation_data_dictionary.keys():
    validation_data_dictionary[img][1] = int(validation_data_dictionary[img][1]) #- direction_mean)\
                              #/ (direction_max-direction_min)
    validation_data_dictionary[img][3] = int(validation_data_dictionary[img][3]) #- speed_mean) / (speed_max-speed_min)

# We are reading validation set images...
road_name = "Road4"
data_type = "Validation"
location_name = "VehicleData"+road_name+data_type
camera_image_size = 100


# Whole validation set images will be hold in this array...
validation_whole_image_set = []
# Corresponding angle and direction information will be hold in this
# in form of [[12, 25], [13, 20]]
validation_information_set = []


# we are loading images...
# getting list of image files...
validation_image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in validation_image_file_list:
    # Only load the GRAY images...
    if dosya.__contains__("GRAYResized"):
        dosya_number = dosya[:-1*len("GRAYResized.png")]
        # read image file
        image_file = cv2.imread("../DataCollector/"+location_name+"/"+dosya)
        image_file = image_file[:, :, 0]
        # Add to array... and reshape for neural network..
        validation_whole_image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # Add corresponding image's angle and speed information... 100x100 = 10000
        validation_information_set.append([validation_data_dictionary[dosya_number][1], validation_data_dictionary[dosya_number][3]])

# Converting numpy array...
validation_whole_image_set = np.array(validation_whole_image_set, dtype=np.float32)




# We are loading training set images...
road_name = "Road4"
data_type = "Train"
location_name = "VehicleData"+road_name+data_type

# Whole training set's images will be stored in this...
whole_image_set = []
# Corresponding speed and direction information....
information_set = []
# Retrieve image list of training set....
image_file_list = os.listdir("../DataCollector/"+location_name)
for dosya in image_file_list:
    # Load only gray scale images...
    if dosya.__contains__("GRAYResized"):
        dosya_number = dosya[:-1*len("GRAYResized.png")]
        image_file = cv2.imread("../DataCollector/"+location_name+"/"+dosya)
        image_file = image_file[:, :, 0]
        # add image to array by reshaping it for neural network.. 100x100 = 10000
        whole_image_set.append(np.reshape(image_file, (camera_image_size**2)))
        # corresponding angle, speed of image...
        information_set.append([data_dictionary[dosya_number][1], data_dictionary[dosya_number][3]])

# Converting to numpy array...
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

# We will apply mean normalization to training set before supplying it to neural network...
# image_normalized = (current_image - mean_image)/(max_image - min_image)


# To normalize images we are extracting mean of training data set...
mean_image = whole_image_set.mean(axis=0)
std_image = whole_image_set.std(axis=0)

# There are 10000 thousand dimension, for each dimension we are getting the max and min of the
# training set
# Max image of the trainin set...
max_image = whole_image_set.max(axis=0)
min_image = whole_image_set.min(axis=0)

# We are getting different image from max and min images...
diff = max_image - min_image

# Some dimensions might be 0 after subtraction, to mitigate division by zero problem we
# assignning the zero index to max value..
for i in range(len(diff)):
    if abs(diff[i]) < 0.0000001:
        diff[i] = max_image[i]

# We are normalizing the traingin image...
whole_image_set = (whole_image_set - mean_image)
whole_image_set = whole_image_set/(diff)
# We are also normalizing the validation set...
validation_whole_image_set = (validation_whole_image_set-mean_image)/diff



image_set = [whole_image_set[index] for index in range(0, whole_image_set.shape[0])]
validation_image_set = [validation_whole_image_set[index] for index in range(0, validation_whole_image_set.shape[0])]

# We are stoing the statistics of training set, these infromation will be used by the driver
# programs....
pickle.dump([direction_max, direction_min, direction_mean, direction_std, speed_max,
             speed_min, speed_mean, speed_std,
             mean_image, std_image, max_image,
             min_image, diff], open("MLPRegressionTrainer("+road_name+")_important_data.dat", "w"))


print "Data have been loaded"

# We are creating the neural network in tensorflow...
# Placeholders for input data and target values...
# Images will be supplied to this placeholder..
camera_image = tf.placeholder(tf.float32, [None, camera_image_size**2], "camera_image")
# Target values of corresponding image are supplied to this placeholder...
target_values = tf.placeholder(tf.float32, [None, 2], "target_values")

# input layer 10000 nodes...
# first hidden layer 352 nodes...
# Weight matrix and bias of input-first hidden layer...
initrange = math.sqrt(2.0/(camera_image_size**2))
w1 = tf.get_variable("w1", (camera_image_size**2, 352), initializer=tf.random_uniform_initializer(-initrange, initrange))
b1 = tf.Variable(tf.zeros(352), name="b1")
# Tanh is applied to first hidden layer...
layer_1_output = tf.nn.tanh(tf.matmul(camera_image, w1) + b1)

# First hidden layer- output layer weight matrix..
# Output lalyer 2 nodes, first is the predicted direction output, second is the predicted speed output
initrange = math.sqrt(2.0/(352))
w4 = tf.get_variable("w4", (352, 2), initializer=tf.random_uniform_initializer(-initrange, initrange))
b4 = tf.Variable(tf.zeros(2), name="b4")
# No activation function at output layer...
final_output = tf.matmul(layer_1_output, w4) + b4

# Loss minimized is the mean squared difference... 1/30 because the batch size is 30
loss_ = (1/30.0)*tf.squared_difference(final_output, target_values)
summation = tf.reduce_sum(loss_)


loss_sum = tf.summary.scalar("Lost", summation)

# We are using ADAM optimizer while training...
optimizer = tf.train.AdamOptimizer(learning_rate=0.000001)
optimize = optimizer.minimize(summation)
# Saver for trained model...

train_saver = tf.train.Saver()
# session creation...
sess = tf.Session()
file_write = tf.summary.FileWriter("Trained_NN/", tf.get_default_graph())
init = tf.global_variables_initializer()

# during training we are keeping the validation error to save the model with smallest validation error...
smallest_validation_error = 10000000000

road_name = "Road4.2"

# During training, validation set error and training error is saved to file...
error_graph_file_name = "MlpRegression"+road_name+"_graph.dat"

# arrays required to show error graphs...
iteration_axis = []
error_axis = []
validation_error_axis = []

# sometimes training procedure is interrupted... To continue it later...
# If model is saved previously, load it and continue training...
if len([file1 for file1 in os.listdir("Trained_NN") if file1.__contains__("MlpRegression"+road_name)]) > 0:
    # Load previously saved model...
    train_saver.restore(sess, "Trained_NN/MlpRegression"+road_name)

    # Load the training error value from previous training...
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
    # If there is no previously saved model, initialize the neural network with random values...

    error_graph_file = open(error_graph_file_name, "w")
    sess.run(init)
    error_iteration_number = 1






# Iteration number of training...
rr = 0
iteration_count = 7
while iteration_count != 0:
    for iteration in range(iteration_count):
        print "**********Iteration "+str(iteration)+"****************"

        # Train network with whole training set and get the training set error...
        l = 0
        for bacht_index in range(0, len(information_set), 30):
            _, loss_value, sum_str = sess.run([optimize, summation, loss_sum],
                                     feed_dict={camera_image: image_set[bacht_index: bacht_index+30],
                                                target_values: information_set[bacht_index:bacht_index+30]})

            l += loss_value
            # print "Summary : ", sum_str
        # After training, calculate the whole validation set error
        j = 0
        for bacht_index in range(0, len(validation_information_set), 30):
            # we are training with minibatch learning...
            loss_value = sess.run(summation,
                                  feed_dict={camera_image: validation_image_set[bacht_index: bacht_index + 30],
                                             target_values: validation_information_set[bacht_index:bacht_index + 30],
                                             })



            j += loss_value

        # If current validation error is less than the previous validation error
        # Save current model...
        # By doing so, in the end we will have the model saved with the lowest validation error value..
        if j < smallest_validation_error:
            smallest_validation_error = j
            print "Smallest error : ", smallest_validation_error
            print "Model has been saved..."
            train_saver.save(sess, "Trained_NN/MlpRegression" + road_name)


        print "Loss : ", l, " Validation error : ", j
        rr += 1

        # Add error values to graphs arrays...

        file_write.add_summary(sum_str, rr)
        iteration_axis.append(error_iteration_number)
        error_axis.append(l)
        validation_error_axis.append(j)
        error_graph_file.write(str(error_iteration_number) + " " + str(l) + " " + str(j) + "\n")
        error_iteration_number += 1
        if l < 0.78:
            break

    # Plot the training and validation set errors...
    plt.subplot(211)
    plt.plot(iteration_axis, error_axis, label="Error")
    plt.subplot(212)
    plt.plot(iteration_axis, validation_error_axis, label="Error")
    plt.show()

    # After each training iteration is finished, user may want to continue training...
    print "Please enter new iteration count "
    iteration_count = int(raw_input())


error_graph_file.close()
file_write.close()
print "Train has been terminated..."

