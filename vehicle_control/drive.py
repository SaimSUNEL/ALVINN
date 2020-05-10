#!/usr/bin/env python
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import rospy

import math
import threading







class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()







last_x = 0
last_y = 0

angle = 0.0 
magnitude = 0.0 

last_stayed_angle = 0.0
total = 0 

relative = 0.0

bridge = None



if __name__ == "__main__":
	try: 
		 
		


		publisher = rospy.Publisher (  "/vehicle_model/cmd_vel" , Twist , queue_size = 1000  )
		rospy.init_node ( "3PI_drive" , anonymous = False )
		
		
		
		msg = Twist ( )
		msg.linear.x = 0
		msg.linear.y = 0 
		msg.linear.z = 0
		msg.angular.x = 0 
		msg.angular.y = 0 
		msg.angular.z = 0
		
		print "Please press a key to drive the robot..."
		
		data = ""
		
		while data != 27:
			data = getch ( )
			if data == 'a':
				magnitude = 1.0
							
				msg.linear.x = -1.5
				msg.linear.y = 0 
				msg.angular.z = 0.0
			elif data == 'd':
				magnitude = -1
					
				msg.linear.x = 1.5
				msg.linear.y = 0 
				msg.angular.z = 0.0
			elif data == 'w':
							
				msg.linear.x = 0.0
				msg.linear.y = 0.0 
				msg.angular.z = 2.5 
				
				
			elif data ==  's':
							
				
				msg.linear.x = 0.0
				msg.linear.y =0.0 
				msg.angular.z = -2.5
				
				
				
			elif data == 'b':
				msg.linear.x = 0.0
				msg.linear.y =0.0 
				msg.angular.z =  0
				
				
			elif ord (data) == 27:
				print "Program has been closed" 
				exit  ( )
				
			else:
				continue			
			
			publisher.publish ( msg )
			threading._sleep ( 0.25 )
			
			msg.linear.x = 0
			msg.linear.y = 0 
			msg.angular.z = 0
			publisher.publish ( msg )
			
			
		
		
		
		
		 
		
	except rospy.ROSInterruptException as err:
		print err	

