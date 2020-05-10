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
image_mutex = Lock()

road_name = "Road4"

vehicle_image = None
# dir max min :  26   -6
# spee max min  30 5

direction_max, direction_min = 26, -6
speed_max, speed_min = 30, 5
camera_image_size = 100


direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = pickle.load(open("MLPClassification2ImagesTrainer("+road_name+")_important_data.dat","r"))




# The input layer ...
# the input size is 20000(100x100)... grayscale image...
camera_image = tf.placeholder(tf.float32, [None, 2*(camera_image_size**2)], "camera_image")

# these two places holders keep the true values of speed and direction information of current image
target_values = tf.placeholder(tf.int32, [None], "target_values")
target_values2 = tf.placeholder(tf.int32, [None], "target_values2")

input_ = 2*(camera_image_size**2)
init_range = math.sqrt(2.0/input_)

# first hidden layer 20000 input - 50 first hidden layer...
w1 = tf.get_variable("w1", (2*(camera_image_size**2), 50),
                     initializer=tf.random_uniform_initializer(-init_range, init_range))
# bias weights of first hidden layer...
b1 = tf.Variable(tf.zeros(50), name="b1")

# First hidden layer activation function is tanh...
layer_1_output = tf.nn.tanh(tf.matmul(camera_image, w1) + b1)

input_ = 50
init_range = math.sqrt(2.0/input_)

data_size = int(direction_max-direction_min)

# Second hidden layer definition, 50 first hidden layer - 32 output layer for direction information...
w4 = tf.get_variable("w4", (50, data_size), initializer=tf.random_uniform_initializer(-init_range, init_range))
# bias of output layer...
b4 = tf.Variable(tf.zeros(data_size), name="b4")

speed_size = int(speed_max-speed_min)

# Second hidden layer definition, 50 first hidden layer - 25 output layer for speed information..
w5 = tf.get_variable("w5", (50, speed_size), initializer=tf.random_uniform_initializer(-1.0, 1.0))
b5 = tf.Variable(tf.zeros(speed_size), name="b5")


# final output for direction information, we are applying softmax...
final_output = tf.nn.softmax((tf.matmul(layer_1_output, w4)+b4))
# final output for speed information, we are applying softmax...
final_output2 = tf.nn.softmax((tf.matmul(layer_1_output, w5)+b5))

train_saver = tf.train.Saver()
sess = tf.Session()

train_saver.restore(sess, "Trained_NN/MLPClassification2Images"+road_name)
image_file = cv2.imread("../DataCollector/VehicleDataRoad4Validation/"+"231GRAYResized.png")
image_file = image_file[:, :, 0]

image_file2 = cv2.imread("../DataCollector/VehicleDataRoad7/"+"232GRAYResized.png")
image_file2 = image_file2[:, :, 0]
combination = np.concatenate((np.reshape(image_file2, (1, camera_image_size ** 2)),
                              np.reshape(image_file, (1, camera_image_size ** 2)))).reshape(20000)


img = combination - mean_image
img /= diff




out = sess.run(final_output, feed_dict={
    camera_image: [img]})
# v = (v-min)/(max-min)
print "out : ", out
curr_angle = int(np.argmax(out)) + direction_min
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
    cv2.waitKey(100)


def drive_thread():
    global msg, vehicle_image, sess
    global vehicle_control_publisher
    first_image_taken = False
    first_image = None
    while True:

        image_mutex.acquire()
        if vehicle_image is not None:
            if first_image_taken == False:
                first_image_taken = True
                img = np.reshape(cv2.cvtColor(cv2.resize(vehicle_image, (100, 100)), cv2.COLOR_BGR2GRAY), (10000))
                first_image = img.copy()
                image_mutex.release()
                continue

            img = np.reshape(cv2.cvtColor(cv2.resize(vehicle_image, (100, 100)), cv2.COLOR_BGR2GRAY), (10000))
            combination = np.concatenate((img,
                                          first_image)).reshape(20000)

            imgt = (combination - mean_image)/diff# /(img_max-img_min)
            out, out2 = sess.run([final_output, final_output2], feed_dict={camera_image: [imgt]})
            # v = (v-min)/(max-min)
            #print "out : ", out
            curr_angle = int(np.argmax(out)) + direction_min
            curr_speed = int(np.argmax(out2))+ speed_min

            #print "Output : ", curr_angle, ":", curr_speed

            msg.linear.x = float(curr_angle)*math.pi/720.0
            msg.linear.y = 0
            msg.angular.z = float(curr_speed)/10.0
            vehicle_control_publisher.publish(msg)

            first_image = img


        image_mutex.release()


        threading._sleep(0.5)


# create trackbars for color change
bridge = cv_bridge.CvBridge()
msg = Twist()
msg.linear.x = 0
msg.linear.y = 0
msg.linear.z = 0
msg.angular.x = 0
msg.angular.y = 0
msg.angular.z = 0




rospy.init_node("MLPClassification2ImagesDriver", anonymous=False)

vehicle_image_subscriber = rospy.Subscriber("/vehicle_cam/image_raw", Image, vehicle_image_callback)
vehicle_control_publisher = rospy.Publisher("/vehicle_model/cmd_vel", Twist, queue_size=1)

print ("Connected to ROS")

threading.Thread(target=drive_thread).start()

rospy.spin()







