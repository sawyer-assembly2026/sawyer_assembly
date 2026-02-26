#!/usr/bin/env python
"""
Created on Wed Feb  4 00:58:20 2026

@author: hquij
"""

import rospy
import time
from devices_interface import robot_ctl as robot


if __name__ == '__main__':
    
    try:
        #Initialize Example node
        rospy.init_node('sawyer_move_to_home', anonymous=False)

        #Create sawyer robot instance
        sawyer = robot.SawyerRobot()

        sawyer.move_to_cartesian_relative(position=[0.0,0.0,0.2], orientation=[0.0,0.0,0.0], move_confirm=True, verbose=True)

        #Wait 1 seconds
        time.sleep(1.0)

        #Move robot to home position
        sawyer.move_to_home()
	
    	#Wait 2 seconds
        time.sleep(2.0)

        #Green signalizes idle state
        if sawyer._is_clicksmart == True:
                sawyer.set_green_light()
        
        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing testing node")
    
    except rospy.ROSInterruptException:
        
        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing testing node")
