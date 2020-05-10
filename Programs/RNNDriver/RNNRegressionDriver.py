import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow.python.client import device_lib
# import rospy
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

camera_image_size = 100

# We are creating the recurrent neural network in tensorflow...
# After RNN layer one two-hidden layered MLP structure is used to output estimated direction angle and speed
# Placeholders for input data and target values...
# Images will be supplied to this placeholder..
# RNN will use LSTM units...
# Each input size is 1000 and the RNN will be unfolded 10 time steps....
camera_image = tf.placeholder(tf.float32, shape=[None, 10, 1000])
target_values = tf.placeholder(tf.float32, shape=[None, 2])

# RNN layer consists of 256 nodes...
basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=256)
outputs, states = tf.nn.dynamic_rnn(basic_cell, camera_image, dtype=tf.float32)


# We are not using each time step's RNN output, instead we are using the last(10th) step
# of RNN output as input to MLP structure....
# MLP's input layer is the RNN node count...
# MLP's first hidden layer count is 100 nodes...
input_number = 256
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w1 = tf.get_variable("w1", (256, 100), initializer=initializer)
b1 = tf.Variable(tf.zeros(100), name="b1")
fc_output = tf.nn.tanh(tf.matmul(states[1], w1)+b1)


# MLP's second hidden layer count is 50
input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w2 = tf.get_variable("w2", (100, 50), initializer=initializer)
b2 = tf.Variable(tf.zeros(50), name="b2")
fc_output1 = tf.nn.tanh(tf.matmul(fc_output, w2)+b2)

# MLP's output layer is 2 nodes...
input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w3 = tf.get_variable("w3", (50, 2), tf.float32, initializer)
b3 = tf.get_variable("b3", [2], tf.float32, tf.constant_initializer(0))
# At output no nonlienarity is applied...
final_output = tf.matmul(fc_output1, w3) + b3


direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = \
    pickle.load(open("RNNRegressionTrainer("+road_name+")_important_data.dat","r"))
mean_image = np.reshape(mean_image, (10, 1000))
diff_image = np.reshape(diff, (10, 1000));
train_saver = tf.train.Saver()
sess = tf.Session()

train_saver.restore(sess, "Trained_NN/RNNRegression"+road_name)
image_file = cv2.imread("../DataCollector/VehicleDataRoad3/"+"232Resized.png")
img = cv2.cvtColor(cv2.resize(image_file, (10, 1000)), cv2.COLOR_BGR2GRAY)
img = (img - mean_image)/(diff_image)
out = sess.run(final_output, feed_dict={
    camera_image: [img]})
# v = (v-min)/(max-min)
print "out : ", out
curr_angle = out[0][0]
curr_speed = out[0][1]

print "direction Output : ", curr_angle, ":"#curr_speed
print "speed : ", curr_speed, ":"#curr_speed

# exit(0)
exit()

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
            img = cv2.cvtColor(cv2.resize(vehicle_image, (100, 100)), cv2.COLOR_BGR2GRAY).reshape((10, 1000))
            print img.shape
            #img = img[:, :, 0]
            img = (img - mean_image)/diff_image# /(img_max-img_min)
            out = sess.run(final_output, feed_dict={camera_image: [img]})
            # v = (v-min)/(max-min)
            #print "out : ", out
            curr_angle = out[0][0]
            curr_speed = out[0][1]

            #print "Output : ", curr_angle, ":", curr_speed
            msg.linear.x = float(curr_angle)*math.pi/720.0
            msg.linear.y = 0
            msg.angular.z = float(curr_speed)/10.0
            vehicle_control_publisher.publish(msg)


        image_mutex.release()


        threading._sleep(1)


# create trackbars for color change
bridge = cv_bridge.CvBridge()
msg = Twist()
msg.linear.x = 0
msg.linear.y = 0
msg.linear.z = 0
msg.angular.x = 0
msg.angular.y = 0
msg.angular.z = 0




rospy.init_node("RNNRegressionDriver", anonymous=False)

vehicle_image_subscriber = rospy.Subscriber("/vehicle_cam/image_raw", Image, vehicle_image_callback)
vehicle_control_publisher = rospy.Publisher("/vehicle_model/cmd_vel", Twist, queue_size=1)

print ("Connected to ROS")

threading.Thread(target=drive_thread).start()

rospy.spin()







