#!/usr/bin/env python
"""
Created on Wed Feb  4 00:58:20 2026

@author: hquij
"""

import sys
import rospy
import rospkg
import time
from devices_interface import robot_ctl as robot
from intera_interface import (
    Navigator,
    HeadDisplay,
    Head
)

if __name__ == '__main__':
    
    try:
        #Initialize Example node
        rospy.init_node('sawyer_move_to_home', anonymous=False)

        #Create sawyer robot instance
        sawyer = robot.SawyerRobot()

        #Create Navigator instance
        navigator = Navigator()

        #Create Head Display instance
        headdisplay = HeadDisplay()
        
        #Create Head movements instance
        head = Head()

        #Get current package path
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('sawyer_sequences')

        #Display Starting home image
        image_path = package_path + "/share/home_01.png"
        headdisplay.display_image(image_path)

        sawyer.move_to_cartesian_relative(position=[0.0,0.0,0.2], orientation=[0.0,0.0,0.0], move_confirm=True, verbose=True)

        #Wait 1 seconds
        time.sleep(1.0)

        #Display moving arm to home image
        image_path = package_path + "/share/home_02.png"
        headdisplay.display_image(image_path)

        #Move robot to home position
        sawyer.move_to_home()
	
    	#Wait 2 seconds
        time.sleep(2.0)

        #Display moving head pan to home image
        image_path = package_path + "/share/home_03.png"
        headdisplay.display_image(image_path)
        head.set_pan(-0.5, speed=0.3)

        #Green signalizes idle state
        #if sawyer._is_clicksmart == True:
        #        sawyer.set_green_light()
        
        #Display Homing finished image
        image_path = package_path + "/share/home_04.png"
        headdisplay.display_image(image_path)

        #Wait 2 seconds
        time.sleep(2.0)

        #Display Intera SDK home image
        #image_path = package_path + "/share/sawyer_sdk_research.png"
        #headdisplay.display_image(image_path)

        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing testing node")
        sys.exit()
    
    except rospy.ROSInterruptException:
        
        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing testing node")
