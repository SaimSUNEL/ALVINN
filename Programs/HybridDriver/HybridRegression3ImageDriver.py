import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow.python.client import device_lib
import rospy
import math
from sensor_msgs.msg import Image
import cv2
import threading
import Tkinter
import cv_bridge
from geometry_msgs.msg import Twist
import numpy as np
import os
from threading import Lock
import pickle
import time
import random

image_mutex = Lock()

vehicle_image = None
road_name = "Road4"

camera_image_size = 100



# input is 3 stacked consequtive RGB images...
camera_image = tf.placeholder(tf.float32, shape=[None, 100, 100, 9])
# direction and speed target values will be fed to network
target_values = tf.placeholder(tf.float32, [None, 2], "target_values")

# placeholder for last 10 speed input... RNN 2 step unfolding each input is 5 nodes...
speed_history_input = tf.placeholder(tf.float32, shape=[None, 2, 5])

# To process last 10 speed inputs we are using RNN with 2 unfold 5 inputs
with tf.variable_scope('lstm2'):
    # RNN layer consists of 128 nodes...
    basic_cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=128)
    outputs2, states2 = tf.nn.dynamic_rnn(basic_cell2, speed_history_input, dtype=tf.float32)

# fc_output = tf.nn.tanh(tf.matmul(states[1], w1)+b1)





# First convolution with 3 by 3 filters that create 64 feature maps...
input_number = 3 * 3 * 9
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W = tf.get_variable("W", (3, 3, 9, 64), tf.float32, initializer)
b = tf.get_variable("b", [64], tf.float32, tf.constant_initializer(0))

conv1 = tf.nn.conv2d(camera_image, W, strides=[1, 1, 1, 1], padding="SAME")

conv1_out = tf.nn.tanh(tf.nn.bias_add(conv1, b))

# Max pooling for the first convolution layer...
max_pool = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Second convolutional layer 3 by 3 filter produces 32 feature maps...
input_number = 3 * 3 * 64
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W2 = tf.get_variable("W2", (3, 3, 64, 32), tf.float32, initializer)
b2 = tf.get_variable("b2", [32], tf.float32, tf.constant_initializer(0))

conv2 = tf.nn.conv2d(max_pool, W2, strides=[1, 3, 3, 1], padding="SAME")

# Tanh is used an nonlinear function
conv2_out = tf.nn.tanh(tf.nn.bias_add(conv2, b2))


# Max pooling for second convolutional layer...
max_pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# After second convolutional layer we are using a RNN layer ...
# We are reshaping last convolutional output to feed it to RNN
# this RNN is unfolded 9 time steps, and input is 288 nodes...
conv2_normalize = tf.reshape(max_pool2, shape=[-1,9 ,9*32])

# This RNN layer has 50 nodes... After this RNN we are using a MLP architecture to predict
# Only direction angle ...
basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=50)
outputs, states = tf.nn.dynamic_rnn(basic_cell, conv2_normalize, dtype=tf.float32)

# We are only using 9th step's output as input to MLP layer(input layer 50 nodes..)
# MLP layer hidden layer has 100 nodes...
input_number = 50
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w3 = tf.get_variable("w3", (50, 100), initializer=initializer)
b3 = tf.Variable(tf.zeros(100), name="b3")

# We are using tanh at hidden layer of MLP
f1_output = tf.nn.tanh(tf.matmul(states[1], w3) + b3)

# For speed estimation, we are inputting last 10 speed information to an RNN
# Speed information can not be solely determined via the last 10 speed values
# While deciding the speed information, we benefit from images supplied to CNN architecture
# By using output of first hidden layer of MLP architecture on CNN network and last step output of
# RNN to which last 10 speed info is fed, we are estimating the speed information...
speed_concat = tf.concat([f1_output, states2[1]], 1)

# Total concatanation becomes 228 nodes.. and this combination is fed to another MLP architecture...

input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W8 = tf.get_variable("W8", (228, 1), tf.float32, initializer)
b8 = tf.get_variable("b8", [1], tf.float32, tf.constant_initializer(0))

# We are simply performing weighted summation on combination to predict speed info..
speed_output = tf.matmul(speed_concat, W8) + b8


# output of MLP after CNN architecture is estimated direction angle...
input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W4 = tf.get_variable("W4", (100, 1), tf.float32, initializer)
b4 = tf.get_variable("b4", [1], tf.float32, tf.constant_initializer(0))
# We are combining direction and speed outputs into one as other architectures...
final_output = tf.concat([tf.matmul(f1_output, W4) + b4, speed_output], 1)



direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = \
    pickle.load(open("HybridRegressionRGB3ImageTrainer("+road_name+")_important_data.dat","r"))
mean_image = np.reshape(mean_image, (100, 100, 3))
diff_image = np.reshape(diff, (100, 100, 3))

train_saver = tf.train.Saver()
sess = tf.Session()

train_saver.restore(sess, "Trained_NN/HybridRegressionRGB3Image"+road_name)
image_file = cv2.imread("../DataCollector/VehicleDataRoad5/"+"232Resized.png")
# v = (v-min)/(max-min)
#curr_speed = out[0][1] * float((speed_max - speed_min)) +speed_min
def vehicle_image_callback(data):
    global vehicle_image
    image_mutex.acquire()
    try:
        vehicle_image = bridge.imgmsg_to_cv2(data, "bgr8")  # "32FC1") #"passthrough") #"bgr8")
    except cv_bridge.CvBridgeError as e:
        print(e)
    image_mutex.release()

    cv2.imshow("Video", vehicle_image)
    cv2.waitKey(100)

queue = [0, 0, 0]
speed_queue = [ random.randint(13, 15) for i in range(10)]
image_count = 0
def drive_thread():
    global msg, vehicle_image, sess, queue, image_count, speed_queue
    global vehicle_control_publisher
    while True:
        image_mutex.acquire()
        if vehicle_image is not None:
            if image_count < 2:
                img = cv2.resize(vehicle_image, (100, 100))
                img = (img - mean_image)/diff_image# /(img_max-img_min)
                queue[2] = queue[1]
                queue[1] = queue[0]
                queue[0] = img
                image_count += 1
            else:
                start_time = time.time()
                img = cv2.resize(vehicle_image, (100, 100))
                img = (img - mean_image)/diff_image  # /(img_max-img_min)
                queue[2] = queue[1]
                queue[1] = queue[0]
                queue[0] = img
                image_count += 1
                combination = np.array([queue], dtype=np.float32)
                combination = np.reshape(combination, (100, 100, 9))

                reshaped_speed_queue = np.reshape(speed_queue, (2, 5))
                out = sess.run(final_output,
                                     feed_dict={camera_image: [combination],
                                                speed_history_input: [reshaped_speed_queue]})
                # v = (v-min)/(max-min)
                # print "out : ", out
                curr_angle = out[0][0]
                curr_speed = out[0][1]

                #print "Output : ", curr_angle, ":", curr_speed
                msg.linear.x = float(curr_angle)*math.pi/720.0
                msg.linear.y = 0
                msg.angular.z = float(curr_speed)/10.0
                vehicle_control_publisher.publish(msg)
                print "Time : ", time.time()-start_time


                for ind in range(9, 0, -1):
                    speed_queue[ind] = speed_queue[ind-1]
                speed_queue[0] = curr_speed


        image_mutex.release()


        threading._sleep(0.2)


# create trackbars for color change
bridge = cv_bridge.CvBridge()
msg = Twist()
msg.linear.x = 0
msg.linear.y = 0
msg.linear.z = 0
msg.angular.x = 0
msg.angular.y = 0
msg.angular.z = 0




rospy.init_node("HybridRegressionRGB3ImageDriver", anonymous=False)

vehicle_image_subscriber = rospy.Subscriber("/vehicle_cam/image_raw", Image, vehicle_image_callback)
vehicle_control_publisher = rospy.Publisher("/vehicle_model/cmd_vel", Twist, queue_size=1)

print ("Connected to ROS")

threading.Thread(target=drive_thread).start()

rospy.spin()







