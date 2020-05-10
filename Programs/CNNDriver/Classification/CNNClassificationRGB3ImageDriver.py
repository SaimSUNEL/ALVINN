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

direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = pickle.load(open("CNNClassificationRGB3ImageTrainer("+road_name+")_important_data.dat","r"))


camera_image = tf.placeholder(tf.float32, shape=[None, 100, 100, 9])
target_values = tf.placeholder(tf.int32, shape=[None])
target_values2 = tf.placeholder(tf.int32, shape=[None])




input_number = 3 * 3 * 3
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W = tf.get_variable("W", (3, 3, 9, 64), tf.float32, initializer)
b = tf.get_variable("b", [64], tf.float32, tf.constant_initializer(0))

conv1 = tf.nn.conv2d(camera_image, W, strides=[1, 1, 1, 1], padding="SAME")

conv1_out = tf.nn.leaky_relu(tf.nn.bias_add(conv1, b))
max_pool = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

input_number = 3 * 3 * 64
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W2 = tf.get_variable("W2", (3, 3, 64, 32), tf.float32, initializer)
b2 = tf.get_variable("b2", [32], tf.float32, tf.constant_initializer(0))

conv2 = tf.nn.conv2d(max_pool, W2, strides=[1, 3, 3, 1], padding="SAME")

conv2_out = tf.nn.leaky_relu(tf.nn.bias_add(conv2, b2))
max_pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

conv2_normalize = tf.reshape(max_pool2, shape=[-1, 9 * 9 * 32])

input_number = 9 * 9 * 32
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W3 = tf.get_variable("W3", (9 * 9 * 32, 100), tf.float32, initializer)
b3 = tf.get_variable("b3", [100], tf.float32, tf.constant_initializer(0))

f1_output = tf.nn.tanh(tf.matmul(conv2_normalize, W3) + b3)

direction_size = direction_max - direction_min

input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W4 = tf.get_variable("W4", (100, direction_size), tf.float32, initializer)
b4 = tf.get_variable("b4", [direction_size], tf.float32, tf.constant_initializer(0))

final_output = tf.nn.softmax((tf.matmul(f1_output, W4) + b4))

speed_size = speed_max - speed_min

input_number = 100
init_range = math.sqrt(2.0 / input_number)
initializer = tf.random_uniform_initializer(-init_range, init_range)
W5 = tf.get_variable("W5", (100, speed_size), tf.float32, initializer)
b5 = tf.get_variable("b5", [speed_size], tf.float32, tf.constant_initializer(0))

final_output2 = tf.nn.softmax((tf.matmul(f1_output, W5) + b5))





mean_image = np.reshape(mean_image, (100, 100, 3))
diff_image = np.reshape(diff, (100, 100, 3))
train_saver = tf.train.Saver()
sess = tf.Session()

train_saver.restore(sess, "Trained_NN/CNNClassificationRGB3Image"+road_name)
image_file = cv2.imread("../DataCollector/VehicleDataRoad4Validation/"+"232Resized.png")
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
import time
def drive_thread():
    global msg, vehicle_image, sess, queue, image_count
    global vehicle_control_publisher

    while True:
        image_mutex.acquire()
        start = time.time()
        if vehicle_image is not None:
            if image_count < 2:
                img = cv2.resize(vehicle_image, (100, 100))
                img = (img - mean_image)/diff_image# /(img_max-img_min)
                queue[2] = queue[1]
                queue[1] = queue[0]
                queue[0] = img
                image_count += 1
            else:
                img = cv2.resize(vehicle_image, (100, 100))
                img = (img - mean_image)/diff_image  # /(img_max-img_min)
                queue[2] = queue[1]
                queue[1] = queue[0]
                queue[0] = img
                image_count += 1
                combination = np.array([queue], dtype=np.float32)
                combination = np.reshape(combination, (100, 100, 9))
                out, out2 = sess.run([final_output, final_output2],
                                     feed_dict={camera_image: [combination]})
                # v = (v-min)/(max-min)
                # print "out : ", out
                curr_angle = int(np.argmax(out))-6
                curr_speed = int(np.argmax(out2))+5
                #print "Output : ", curr_angle, ":", curr_speed
                msg.linear.x = float(curr_angle)*math.pi/720.0
                msg.linear.y = 0
                msg.angular.z = float(curr_speed)/10.0
                vehicle_control_publisher.publish(msg)
                end_time = time.time()
                print "Calculation time : ", end_time- start

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




rospy.init_node("CNNClassificationRGB3ImageDriver", anonymous=False)

vehicle_image_subscriber = rospy.Subscriber("/vehicle_cam/image_raw", Image, vehicle_image_callback)
vehicle_control_publisher = rospy.Publisher("/vehicle_model/cmd_vel", Twist, queue_size=1)

print ("Connected to ROS")

threading.Thread(target=drive_thread).start()

rospy.spin()







