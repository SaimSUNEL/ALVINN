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
image_mutex = Lock()

road_name = "Road4"

vehicle_image = None
# dir max min :  26   -6
# spee max min  30 5

direction_max, direction_min = 26, -6
speed_max, speed_min = 30, 5
camera_image_size = 100



direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = pickle.load(open("CNNClassificationTrainer("+road_name+")_important_data.dat","r"))








# we are using a one channel (gray scale) 100x100 image...
camera_image = tf.placeholder(tf.float32, shape=[None, 100, 100, 1])
# direction target values will be fed to network
target_values = tf.placeholder(tf.int32, shape=[None])
# speed target values will be fed to network via this placeholder...
target_values2 = tf.placeholder(tf.int32, shape=[None])


# in the first convolution layer, we are using 3x3 filter and it these filters
# will produce 64 layer future map...
input_ = 3 * 3 * 1
initializer = tf.random_uniform_initializer(-1, 1)
W = tf.get_variable("W", (3, 3, 1, 64), tf.float32, initializer)
# bias weights of first convolutional layer...
b = tf.get_variable("b", [64], tf.float32, tf.constant_initializer(0))
conv1 = tf.nn.conv2d(camera_image, W, strides=[1, 1, 1, 1], padding="SAME")

# leaky relu activation function is used for first convolution layer...
conv1_out = tf.nn.leaky_relu(tf.nn.bias_add(conv1, b))

# first pooling layer with 2x2 window is applied after first convolutional layer....
max_pool = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Second convolution layer with 3x3x64 3d filter to create 32 layer feature map in second convolutional layer
input_ = 3 * 3 * 64
initializer = tf.random_uniform_initializer(-1, 1)
W2 = tf.get_variable("W2", (3, 3, 64, 32), tf.float32, initializer)
# bias weights of second convolutional layer...
b2 = tf.get_variable("b2", [32], tf.float32, tf.constant_initializer(0))

conv2 = tf.nn.conv2d(max_pool, W2, strides=[1, 3, 3, 1], padding="SAME")

# Second convolutional layer's activation fucntion leaky relu function...
conv2_out = tf.nn.leaky_relu(tf.nn.bias_add(conv2, b2))

# again we are applying another pooling layer for the second convolutional layer with 2x2 window size...
max_pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# After second convolutional layer, we are construncting a feed-forward network... 100x32 for
conv2_normalize = tf.reshape(max_pool2, shape=[-1, 9 * 9 * 32])
input_ = 3 * 3 * 32
initializer = tf.random_uniform_initializer(-1, 1)
# feed-forward network's input is second convolution layer... We are constructing a hidden layer of 100 hidde nodes..
W3 = tf.get_variable("W3", (9 * 9 * 32, 100), tf.float32, initializer)
# first feed hidden layer bias....
b3 = tf.get_variable("b3", [100], tf.float32, tf.constant_initializer(0))

# feed-forward's hidden layer activation function is tanh function....
f1_output = tf.nn.tanh(tf.matmul(conv2_normalize, W3) + b3)

# for direction output we are constructing a 32 node output layer...
input_ = 500
direction_size = int(direction_max - direction_min)

initializer = tf.random_uniform_initializer(-1, 1)
W4 = tf.get_variable("W4", (100, direction_size), tf.float32, initializer)
b4 = tf.get_variable("b4", [direction_size], tf.float32, tf.constant_initializer(0))
# we are applying softmax to direction output...
final_output = tf.nn.softmax((tf.matmul(f1_output, W4) + b4))

speed_size = int(speed_max-speed_min)
# for speed output we are constructing a 25 node output layer...
W5 = tf.get_variable("W5", (100, speed_size), tf.float32, initializer)
b5 = tf.get_variable("b5", [speed_size], tf.float32, tf.constant_initializer(0))
# we are applying softmax for output of speed output...
final_output2 = tf.nn.softmax((tf.matmul(f1_output, W5) + b5))

train_saver = tf.train.Saver()
sess = tf.Session()

train_saver.restore(sess, "Trained_NN/CNNClassification"+road_name)
image_file = cv2.imread("../DataCollector/VehicleDataRoad4Validation/"+"394GRAYResized.png")
gr  = cv2.resize(cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY), (100, 100))
img = np.reshape(cv2.cvtColor(cv2.resize(image_file, (100, 100)), cv2.COLOR_BGR2GRAY), (10000))
img = img - mean_image
img /= diff
img = np.reshape(img, (100, 100, 1))

out = sess.run(final_output, feed_dict={
    camera_image: [img]})
# v = (v-min)/(max-min)
print "out : ", out
print np.argmax(out)
print direction_min
curr_angle = int(np.argmax(out)) +direction_min
#curr_speed = out[0][1] * float((speed_max - speed_min)) +speed_min

print "Output : ", curr_angle, ":"#curr_speed



def vehicle_image_callback(data):
    global vehicle_image
    image_mutex.acquire()
    try:
        vehicle_image = bridge.imgmsg_to_cv2(data, "bgr8")  # "32FC1") #"passthrough") #"bgr8")
    except cv_bridge.CvBridgeError as e:
        print(e)
    image_mutex.release()

    cv2.imshow("Video", vehicle_image)
    cv2.waitKey(30)


def drive_thread():
    global msg, vehicle_image, sess
    global vehicle_control_publisher
    while True:
        image_mutex.acquire()
        if vehicle_image is not None:
            img = np.reshape(cv2.cvtColor(cv2.resize(vehicle_image, (100, 100)), cv2.COLOR_BGR2GRAY), (10000))
            img = (img - mean_image)/diff# /(img_max-img_min)
            img = np.reshape(img, (100, 100, 1))
            out, out2 = sess.run([final_output, final_output2], feed_dict={camera_image: [img]})
            # v = (v-min)/(max-min)
            #print "out : ", out
            curr_angle = int(np.argmax(out)) + direction_min
            curr_speed = int(np.argmax(out2))+speed_min

            #print "Output : ", curr_angle, ":", curr_speed

            msg.linear.x = float(curr_angle)*math.pi/720.0
            msg.linear.y = 0
            msg.angular.z = float(curr_speed)/10.0
            vehicle_control_publisher.publish(msg)


        image_mutex.release()


        threading._sleep(0.3)


# create trackbars for color change
bridge = cv_bridge.CvBridge()
msg = Twist()
msg.linear.x = 0
msg.linear.y = 0
msg.linear.z = 0
msg.angular.x = 0
msg.angular.y = 0
msg.angular.z = 0




rospy.init_node("CNNClassificationDriver", anonymous=False)

vehicle_image_subscriber = rospy.Subscriber("/vehicle_cam/image_raw", Image, vehicle_image_callback)
vehicle_control_publisher = rospy.Publisher("/vehicle_model/cmd_vel", Twist, queue_size=1)

print ("Connected to ROS")

threading.Thread(target=drive_thread).start()

rospy.spin()







