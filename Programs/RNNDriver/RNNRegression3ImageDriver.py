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

camera_image_size = 100


# Same RNN architecture with exception that it is unfolded 30 time for 3 grayscale images...
camera_image = tf.placeholder(tf.float32, shape=[None, 30, 1000])
target_values = tf.placeholder(tf.float32, shape=[None, 2])


basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=256)
outputs, states = tf.nn.dynamic_rnn(basic_cell, camera_image, dtype=tf.float32)

input_number = 256
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w1 = tf.get_variable("w1", (256, 100), initializer=initializer)
b1 = tf.Variable(tf.zeros(100), name="b1")
fc_output = tf.nn.tanh(tf.matmul(states[1], w1)+b1)


input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w2 = tf.get_variable("w2", (100, 50), initializer=initializer)
b2 = tf.Variable(tf.zeros(50), name="b2")
fc_output1 = tf.nn.tanh(tf.matmul(fc_output, w2)+b2)


input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w3 = tf.get_variable("w3", (50, 2), tf.float32, initializer)
b3 = tf.get_variable("b3", [2], tf.float32, tf.constant_initializer(0))

final_output = tf.matmul(fc_output1, w3) + b3


direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = \
    pickle.load(open("RNNRegression3ImageTrainer("+road_name+")_important_data.dat","r"))
mean_image = np.reshape(mean_image, (100, 100, 3))
diff_image = np.reshape(diff, (100, 100, 3))
train_saver = tf.train.Saver()
sess = tf.Session()

train_saver.restore(sess, "Trained_NN/RNNRegression3Image"+road_name)
image_file = cv2.imread("../DataCollector/VehicleDataRoad3/"+"232Resized.png")
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
image_count = 0
def drive_thread():
    global msg, vehicle_image, sess, queue, image_count
    global vehicle_control_publisher
    while True:
        image_mutex.acquire()
        if vehicle_image is not None:
            if image_count < 2:
                img = cv2.cvtColor(cv2.resize(vehicle_image, (100, 100)), cv2.COLOR_BGR2GRAY)
                img = (img - mean_image)/diff_image # /(img_max-img_min)
                queue[2] = queue[1]
                queue[1] = queue[0]
                queue[0] = img
                image_count += 1
            else:
                img = cv2.cvtColor(cv2.resize(vehicle_image, (100, 100)), cv2.COLOR_BGR2GRAY)
                img = (img - mean_image)/diff_image  # /(img_max-img_min)
                queue[2] = queue[1]
                queue[1] = queue[0]
                queue[0] = img
                image_count += 1
                combination = np.array([queue], dtype=np.float32)
                combination = np.reshape(combination, (90, 1000))
                out  = sess.run(final_output,
                                     feed_dict={camera_image: [combination]})
                # v = (v-min)/(max-min)
                # print "out : ", out

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




rospy.init_node("RNNRegression3ImageDriver", anonymous=False)

vehicle_image_subscriber = rospy.Subscriber("/vehicle_cam/image_raw", Image, vehicle_image_callback)
vehicle_control_publisher = rospy.Publisher("/vehicle_model/cmd_vel", Twist, queue_size=1)

print ("Connected to ROS")

threading.Thread(target=drive_thread).start()

rospy.spin()







