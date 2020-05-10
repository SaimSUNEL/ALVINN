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


direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff_image = \
    pickle.load(open("MLPRegressionTrainer("+road_name+")_important_data.dat","r"))


camera_image_size = 100


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



train_saver = tf.train.Saver()
sess = tf.Session()

train_saver.restore(sess, "Trained_NN/MlpRegression"+road_name)
image_file = cv2.imread("../DataCollector/VehicleDataRoad4Train/"+"357GRAYResized.png")
gr  = cv2.resize(cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY), (100, 100))
img = np.reshape(cv2.cvtColor(cv2.resize(image_file, (100, 100)), cv2.COLOR_BGR2GRAY), (10000))
img = (img - mean_image)/(diff_image)
out = sess.run(final_output, feed_dict={
    camera_image: [img]})
# v = (v-min)/(max-min)
print "out : ", out
curr_angle = out[0][0]
curr_speed = out[0][1]

print "direction Output : ", curr_angle, ":"#curr_speed
print "speed : ", curr_speed, ":"#curr_speed




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


def drive_thread():
    global msg, vehicle_image, sess
    global vehicle_control_publisher
    while True:
        image_mutex.acquire()
        if vehicle_image is not None:
            img = np.reshape(cv2.cvtColor(cv2.resize(vehicle_image, (100, 100)), cv2.COLOR_BGR2GRAY), (10000))
            img = (img - mean_image)/(diff_image)
            out = sess.run(final_output, feed_dict={camera_image: [img]})
            # v = (v-min)/(max-min)

            curr_angle = out[0][0] #* float((direction_max - direction_min)) +direction_mean
            curr_speed = out[0][1] #* float((speed_max - speed_min)) +speed_mean

            #print "Output : ", curr_angle, ":", curr_speed

            msg.linear.x = float(curr_angle)*math.pi/720.0
            msg.linear.y = 0
            msg.angular.z = float(curr_speed)/10.0
            vehicle_control_publisher.publish(msg)


        image_mutex.release()


        threading._sleep(1.0)


# create trackbars for color change
bridge = cv_bridge.CvBridge()
msg = Twist()
msg.linear.x = 0
msg.linear.y = 0
msg.linear.z = 0
msg.angular.x = 0
msg.angular.y = 0
msg.angular.z = 0




rospy.init_node("MLPRegressionDriver", anonymous=False)

vehicle_image_subscriber = rospy.Subscriber("/vehicle_cam/image_raw", Image, vehicle_image_callback)
vehicle_control_publisher = rospy.Publisher("/vehicle_model/cmd_vel", Twist, queue_size=1)

print ("Connected to ROS")

threading.Thread(target=drive_thread).start()

rospy.spin()







