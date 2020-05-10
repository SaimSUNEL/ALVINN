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

camera_image_size = 100

direction_max, direction_min, direction_mean, direction_std, speed_max, speed_min, speed_mean, speed_std, mean_image, std_image, img_max, img_min, diff = \
    pickle.load(open("MLPRegressionRGB3ImageTrainer("+road_name+")_important_data.dat","r"))

print "mean shape : ", mean_image.shape
print "diff shape : ", diff.shape




# Input layer is 90000 nodes
camera_image = tf.placeholder(tf.float32, [None, (camera_image_size**2)*9], "camera_image")
target_values = tf.placeholder(tf.float32, [None, 2], "target_values")

initrange = math.sqrt(2.0/(camera_image_size**2)*9)

w1 = tf.get_variable("w1", ((camera_image_size**2)*9, 352), initializer=tf.random_uniform_initializer(-initrange, initrange))
b1 = tf.Variable(tf.zeros(352), name="b1")

layer_1_output = tf.nn.tanh(tf.matmul(camera_image, w1) + b1)

initrange = math.sqrt(2.0/(352))
w4 = tf.get_variable("w4", (352, 2), initializer=tf.random_uniform_initializer(-initrange, initrange))
b4 = tf.Variable(tf.zeros(2), name="b4")

final_output = tf.matmul(layer_1_output, w4) + b4


train_saver = tf.train.Saver()
sess = tf.Session()

train_saver.restore(sess, "Trained_NN/MlpRegressionRGB3Images"+road_name)


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
                img = np.reshape(cv2.resize(vehicle_image, (100, 100)), (30000))
                img = (img - mean_image)/diff# /(img_max-img_min)
                queue[2] = queue[1]
                queue[1] = queue[0]
                queue[0] = img
                image_count += 1
            else:
                start_time = time.time()
                img = np.reshape(cv2.resize(vehicle_image, (100, 100)), (30000))
                img = (img - mean_image)/diff  # /(img_max-img_min)
                queue[2] = queue[1]
                queue[1] = queue[0]
                queue[0] = img
                image_count += 1
                combination = np.array([queue], dtype=np.float32)
                combination = np.reshape(combination, (90000))
                out = sess.run(final_output,
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
                print "Time : ", time.time()-start_time

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




rospy.init_node("MLPRegressionRGB3ImagesDriver", anonymous=False)

vehicle_image_subscriber = rospy.Subscriber("/vehicle_cam/image_raw", Image, vehicle_image_callback)
vehicle_control_publisher = rospy.Publisher("/vehicle_model/cmd_vel", Twist, queue_size=1)

print ("Connected to ROS")

threading.Thread(target=drive_thread).start()

rospy.spin()
100, 100, 9






