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

vehicle_image = None
# dir max min :  26   -6
# spee max min  30 5

direction_max, direction_min = 26, -6
speed_max, speed_min = 30, 5
camera_image_size = 100

# 100 input 100 stepss...
camera_image = tf.placeholder(tf.float32, shape=[None, 3, 10000])
target_values = tf.placeholder(tf.int32, shape=[None])
target_values2 = tf.placeholder(tf.int32, shape=[None])

basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=50)
outputs, states = tf.nn.dynamic_rnn(basic_cell, camera_image, dtype=tf.float32)

input_number = 50
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w1 = tf.get_variable("w1", (50, 50), initializer=initializer)
b1 = tf.Variable(tf.zeros(50), name="b1")
fc_output = tf.nn.leaky_relu(tf.matmul(states[1], w1)+b1)

input_number = 50
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w2 = tf.get_variable("w2", (50, 25), tf.float32, initializer)
b2 = tf.get_variable("b2", [25], tf.float32, tf.constant_initializer(0))


fc2_output = tf.nn.leaky_relu(tf.matmul(fc_output, w2)+b2)

input_number = 25
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
w3 = tf.get_variable("w3", (25, 32), tf.float32, initializer)
b3 = tf.get_variable("b3", [32], tf.float32, tf.constant_initializer(0))

final_output = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(fc2_output, w3) + b3))

input_number = 25
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W4 = tf.get_variable("W4", (25, 25), tf.float32, initializer)
b4 = tf.get_variable("b4", [25], tf.float32, tf.constant_initializer(0))

final_output2 = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(fc2_output, W4) + b4))

one_hot = tf.one_hot(target_values, 32, on_value=1.0, off_value=0.0, dtype=tf.float32)
one_hot2 = tf.one_hot(target_values2, 25, on_value=1.0, off_value=0.0, dtype=tf.float32)

loss_s = one_hot2 * tf.log(final_output2)
entropys = -tf.reduce_sum(loss_s, axis=1)
entropy2s = tf.reduce_sum(entropys)

loss_ = one_hot * tf.log(final_output)
entropy = -tf.reduce_sum(loss_, axis=1)
entropy2 = tf.reduce_sum(entropy)
finn = entropy2 + entropy2s

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
optimize = optimizer.minimize(finn)

direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = pickle.load(open("RNNClassificationRGBTrainer_important_data.dat","r"))
mean_image = np.reshape(mean_image, (3, 10000))
diff_image = np.reshape(diff, (3, 10000))

train_saver = tf.train.Saver()
sess = tf.Session()

train_saver.restore(sess, "Trained_NN/RNNClassificationRGB")
image_file = cv2.imread("../DataCollector/VehicleDataRoad3/"+"358Resized.png")
image_file = np.reshape(image_file, (3, 10000))
image_file = (image_file - mean_image)/diff_image

ang, speed = sess.run([final_output, final_output2], feed_dict={camera_image: [image_file]})
print "Ang :", np.argmax(ang) - 6
print "Speed :", np.argmax(speed)+5


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


def drive_thread():
    global msg, vehicle_image, sess
    global vehicle_control_publisher
    while True:
        image_mutex.acquire()
        if vehicle_image is not None:
            img = cv2.resize(vehicle_image, (100, 100))
            img = np.reshape(img, (3, 10000))
            # print img.shape
            # img = img[:, :, 0]
            img = (img - mean_image)/diff_image  # /(img_max-img_min)
            out, out2 = sess.run([final_output, final_output2], feed_dict={camera_image: [img]})
            # v = (v-min)/(max-min)
            # print "out : ", out
            curr_angle = int(np.argmax(out))-6
            curr_speed = int(np.argmax(out2))+5
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




rospy.init_node("RNNClassificationRGBDriver", anonymous=False)

vehicle_image_subscriber = rospy.Subscriber("/vehicle_cam/image_raw", Image, vehicle_image_callback)
vehicle_control_publisher = rospy.Publisher("/vehicle_model/cmd_vel", Twist, queue_size=1)

print ("Connected to ROS")

threading.Thread(target=drive_thread).start()

rospy.spin()







