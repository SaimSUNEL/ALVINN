# Program stores direction angle, and speed commands sent to vehicle and images taken from
# vehicle camera...


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
image_mutex = Lock()
file_lock = Lock()

data_file = None
last_image_number = 0
file_isopen = False

# We are storing the collected data in this folder, if it does not exist,
# create one...
if not os.path.isdir("VehicleData"):
    os.mkdir("VehicleData")
    # Direction angle and speed data will be stored with image number to this file...
    data_file = open("VehicleData/data.dat", "a")
    file_isopen = True
else:
    # If directory already exists, find the largest numbered image,
    # we will continue adding from last number
    file_list = os.listdir("VehicleData")
    numbers = [int(val[:-4]) for val in file_list if val.__contains__(".png") and not val.__contains__("Resized") and not val.__contains__("GRAY")]
    if len(numbers) == 0:
        last_image_number = 0
    else:
        last_image_number = max(numbers)
    # Adding information to data storage file..
    data_file = open("VehicleData/data.dat", "a")
    file_isopen = True


#  mutex.acquire()
#  mutex.release()
vehicle_image = None


# This function receives images from ROS and stores it in program.....
# mutex are used to avoid race condition...
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


def nothing():
    pass



# this part creates a GUI thread consisting of scroll bars for angle and speed...
# The scrollbars can be controlled via keys...
# w speed up, s speed down
# a left turn, d right turn...
# recording can be started and stopped at any time by pressing r and t keys..
tkmaster = None
speed_scroll = None
angle_scroll = None
record_status_var, record_status_label = None, None

recording_active = False


def control_gui():
    global tkmaster, speed_scroll, angle_scroll, record_status_var, record_status_label, recording_active
    global file_isopen, data_file

    def keyboard_event(event):
        global file_isopen, data_file, recording_active
        print "Speed key : ", event.char
        if event.char == "w":
            if speed_scroll.get() != "30":
                speed_scroll.set(speed_scroll.get()+1)
        elif event.char == "s":
            if speed_scroll.get() != "5":
                speed_scroll.set(speed_scroll.get()-1)
        elif event.char == "a":
            angle_scroll.set(angle_scroll.get()-1)
        elif event.char == "d":
            angle_scroll.set(angle_scroll.get()+1)

        elif event.char == "r":
            file_lock.acquire()
            if not file_isopen:
                data_file = open("VehicleData/data.dat", "a")
                file_isopen = True
                print "File open again..."
            else:
                print "File is already open..."

            recording_active = True
            record_status_var.set("Recording")
            file_lock.release()

        elif event.char == "t":
            file_lock.acquire()
            if file_isopen:
                data_file.close()
                file_isopen = False
                print "File closed"
            else:
                print "File is already closed..."

            recording_active = False
            record_status_var.set("Not Recording...")
            file_lock.release()

    tkmaster = Tkinter.Tk()
    frame = Tkinter.Frame(tkmaster)
    frame.pack()
    speed_scroll = Tkinter.Scale(frame, from_=5, to=30, orient=Tkinter.HORIZONTAL)

    speed_scroll.pack()

    record_status_var = Tkinter.StringVar()
    record_status_label = Tkinter.Label(frame, textvariable=record_status_var, relief=Tkinter.RAISED)

    record_status_var.set("Not Recording...")
    record_status_label.pack()

    angle_scroll = Tkinter.Scale(frame, from_=-90, to=90, orient=Tkinter.HORIZONTAL)

    angle_scroll.pack()
    tkmaster.bind("<Key>", keyboard_event)

    Tkinter.mainloop()



# This part sends speed and direction angle info to vehicle also saves images and current speed
# speed info to file...
# We are collecting data 2 times in a second
def collect_thread():
    global last_image_number, file_isopen, data_file
    global msg
    global vehicle_control_publisher, recording_active
    while True:
        if angle_scroll is not None and speed_scroll is not None:
            # print "direction angle : ", angle_scroll.get()
            # print "vehicle speed : ", speed_scroll.get()

            msg.linear.x = float(angle_scroll.get())*math.pi/720.0
            msg.linear.y = 0
            msg.angular.z = float(speed_scroll.get())/10.0
            vehicle_control_publisher.publish(msg)

            vehicle_control_publisher.publish(msg)

            if recording_active:
                file_lock.acquire()
                if file_isopen:
                    image_mutex.acquire()

                    cv2.imwrite("VehicleData/" + str(last_image_number) + ".png", vehicle_image)
                    data_file.write("" + str(last_image_number) + " " + str(angle_scroll.get())
                                    + " " + str(msg.linear.x) + " "
                                    + str(speed_scroll.get()) + " " + str(msg.angular.z)+"\n")
                    last_image_number += 1
                    image_mutex.release()
                # get_image use mutex...
                # speed msg.angular.z
                # angle msg.linear.x
                file_lock.release()
                pass

        threading._sleep(0.5)

# Structures used to communicate with ROS

bridge = cv_bridge.CvBridge()
msg = Twist()
msg.linear.x = 0
msg.linear.y = 0
msg.linear.z = 0
msg.angular.x = 0
msg.angular.y = 0
msg.angular.z = 0

# Initialization for ROS...
rospy.init_node("VehicleDataCollector", anonymous=False)

vehicle_image_subscriber = rospy.Subscriber("/vehicle_cam/image_raw", Image, vehicle_image_callback)
vehicle_control_publisher = rospy.Publisher("/vehicle_model/cmd_vel", Twist, queue_size=1)


# Starting control and recording threads...
print ("Connected to ROS")

threading.Thread(target=control_gui).start()

threading.Thread(target=collect_thread).start()

rospy.spin()
