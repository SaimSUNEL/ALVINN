import numpy as np
import math
import matplotlib.pyplot as plt

# This program creates histogram for speed, direction angle data for validation, train and test set...


# We are reading Test set's speed and direction angles data from file..
road_name = "Road4"
data_type = "Test"
location_name = "VehicleData"+road_name+data_type
camera_image_size = 100

test_data_dictionary = {}
test_direction_data = []
test_speed_data = []

data_file = open("../DataCollector/"+location_name+"/data.dat", "r")
for line in data_file:
    f = line.split(" ")
    if len(f) == 5:
        test_data_dictionary[f[0]] = f
        test_direction_data.append(int(f[1]))
        test_speed_data.append(int(f[3]))

data_file.close()




# We are reading Validation set's speed and direction angles data from file..

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





# We are reading Train set's speed and direction angles from file..

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
for line in data_file:
    f = line.split(" ")
    if len(f) == 5:
        data_dictionary[f[0]] = f
        direction_data.append(int(f[1]))
        speed_data.append(int(f[3]))


# Converting string data to integer for each data set...
for img in data_dictionary.keys():
    data_dictionary[img][1] = int(data_dictionary[img][1]) #- direction_mean)\
                              #/ (direction_max-direction_min)
    data_dictionary[img][3] = int(data_dictionary[img][3]) #- speed_mean) / (speed_max-speed_min)

data_file.close()

for img in validation_data_dictionary.keys():
    validation_data_dictionary[img][1] = int(validation_data_dictionary[img][1])# - direction_mean)\
                             # / (direction_max-direction_min)
    validation_data_dictionary[img][3] = int(validation_data_dictionary[img][3])# - speed_mean) / (speed_max-speed_min)


for img in test_data_dictionary.keys():
    test_data_dictionary[img][1] = int(test_data_dictionary[img][1])# - direction_mean)\
                             # / (direction_max-direction_min)
    test_data_dictionary[img][3] = int(test_data_dictionary[img][3])# - speed_mean) / (speed_max-speed_min)




# we are putting all direction and speed data into two separate arrays...
# for each training set...
test_train_direction_set = []
test_train_speed_set = []

for img in test_data_dictionary.keys():
    test_train_direction_set.append(test_data_dictionary[img][1])
    test_train_speed_set.append(test_data_dictionary[img][3])





train_direction_set = []
train_speed_set = []

for img in data_dictionary.keys():
    train_direction_set.append(data_dictionary[img][1])
    train_speed_set.append(data_dictionary[img][3])

validation_train_direction_set = []
validation_train_speed_set = []

for img in validation_data_dictionary.keys():
    validation_train_direction_set.append(validation_data_dictionary[img][1])
    validation_train_speed_set.append(validation_data_dictionary[img][3])




# We are displaying histograms from data...

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].set_title('Train Direction')
axs[0].hist(train_direction_set, bins=60)
axs[1].set_title('Train Speed')
axs[1].hist(train_speed_set, bins=30)

fig2, axs2 = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs2[0].set_title('Validation Direction')
axs2[0].hist(validation_train_direction_set, bins=60)
axs2[1].set_title('Validation Speed')
axs2[1].hist(validation_train_speed_set, bins=30)


fig2, axs2 = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs2[0].set_title('Test Direction')
axs2[0].hist(test_train_direction_set, bins=60)
axs2[1].set_title('Test Speed')
axs2[1].hist(test_train_speed_set, bins=30)



plt.show()
