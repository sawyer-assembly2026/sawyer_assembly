#!/usr/bin/env python

"""
Created on Wed Nov 19 17:23:10 2025

@author: Hector Quijada

Example moves sawyer robot to predetermined home position and starts adquiring FT sensor data
End effector can be moved by hand

"""

import rospy
import rospkg
import time
import numpy as np
import os
from datetime import datetime
from devices_interface import robot_ctl as robot
from devices_interface import ATI_Net as ati
from geometry_msgs.msg import WrenchStamped

#Global variables for force and torque
#Need to be global to be updated by call back function in subscriber
ati_ft_data_ext = np.zeros((1,12))
ft_samples = np.zeros((1,12))

def obtain_ftdata(data,args):
	
    global ft_samples

    #Split wrench data
    ati_ft_data_ext[0][0] = data.wrench.force.x
    ati_ft_data_ext[0][1] = data.wrench.force.y
    ati_ft_data_ext[0][2] = data.wrench.force.z
    ati_ft_data_ext[0][3] = data.wrench.torque.x
    ati_ft_data_ext[0][4] = data.wrench.torque.y
    ati_ft_data_ext[0][5] = data.wrench.torque.z
    ati_ft_data_ext[0][6] = args[0]
    ati_ft_data_ext[0][7] = args[1]
    ati_ft_data_ext[0][8] = args[2]
    ati_ft_data_ext[0][9] = args[3]
    ati_ft_data_ext[0][10] = args[4]
    ati_ft_data_ext[0][11] = args[5]

    #Stack measurement at call back
    ft_samples = np.vstack((ft_samples,ati_ft_data_ext))
    
def adquisition_cycle(test_time=1.0,direction="",category=[]):

    global ft_samples
    ft_samples = np.zeros((1,12))
    arg = category
    
    try:
        sub = rospy.Subscriber("/robot/ati_ft_sensor_topic/", WrenchStamped, obtain_ftdata, callback_args=(arg[0][0],arg[0][1],arg[0][2],arg[0][3],arg[0][4],arg[0][5]))

        #Define the duration you want the subscriber to run (e.g., 10 seconds)
        duration = rospy.Duration(test_time)
        start_time = rospy.Time.now()
        rospy.loginfo("Ft data adquisition started. It will run for %s seconds.", duration/1e9)

        # rospy.spin() would block until shutdown, so we use a loop
        while rospy.Time.now() - start_time < duration and not rospy.is_shutdown():
            # Sleep to prevent the loop from consuming all CPU resources
            rospy.Rate(100).sleep() # Adjust sleep time as needed

        final_time = rospy.Time.now() - start_time

        rospy.loginfo("Subscriber finished running for %s seconds. Saving data and shutting down.", final_time/1e9)
        
        #Delete first auxiliary value
        ft_samples = np.delete(ft_samples, 0, axis=0)

        #Save all data to a csv file
        current_datetime = datetime.now().strftime("%d%m%Y_%H_%M_%S")
        ATI_net_path = "samples_" + current_datetime + "_" + direction + ".csv"
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('tactile')
        csv_file_path = os.path.join(package_path, 'samples', ATI_net_path)
        np.savetxt(csv_file_path, ft_samples, delimiter=',', fmt='%f')

        rospy.loginfo("Data saved succesfully")

        #Unsubscribe to topic to keep node running
        sub.unregister()
    
    except rospy.ROSInterruptException:

        # Explicitly shutdown the node after the duration
        rospy.loginfo("Test unsuccesful, finishing testing node")
        rospy.signal_shutdown("Finishing testing node")


if __name__ == '__main__':

    test_movements = [
                    ("None", np.array([[1., 0., 0., 0., 0., 0.]]))] 
                    #("X+", np.array([[1., 0., 0., 0., 0., 1.]])), 
                    #("X-", np.array([[1., 0., 0., 0., 1., 0.]])),
                    #("Y+", np.array([[1., 0., 0., 0., 1., 1.]])),
                    #("Y-", np.array([[1., 0., 0., 1., 0., 0.]])),
                    #("Z+", np.array([[1., 0., 0., 1., 0., 1.]])),
                    #("Z-", np.array([[1., 0., 0., 1., 1., 0.]])),
                    #("RotX+", np.array([[1., 0., 0., 1., 1., 1.]])),
                    #("RotX-", np.array([[1., 0., 1., 0., 0., 0.]])),
                    #("RotY+", np.array([[1., 0., 1., 0., 0., 1.]])),
                    #("RotY-", np.array([[1., 0., 1., 0., 1., 0.]])),
                    #("RotZ+", np.array([[1., 0., 1., 0., 1., 1.]])),
                    #("RotZ-", np.array([[1., 0., 1., 1., 0., 0.]])),
                    #("X+Y+", np.array([[1., 0., 1., 1., 0., 1.]])),
                    #("X+Y-", np.array([[1., 0., 1., 1., 1., 0.]])),
                    #("X-Y+", np.array([[1., 0., 1., 1., 1., 1.]])),
                    #("X-Y-", np.array([[1., 1., 0., 0., 0., 0.]]))
                    #]
    
    try:
        #Initialize Example node
        rospy.init_node('sawyer_ftdata_home', anonymous=False)

        #Create sawyer robot instance
        sawyer = robot.SawyerRobot()

        #Wait 2 seconds
        time.sleep(2)

        if sawyer._is_clicksmart == True:
            sawyer.set_red_light()
        
        #Move robot to home
        sawyer.move_to_home()

        #Wait 2 seconds
        time.sleep(2)

        for movement in test_movements:
            
            if sawyer._is_clicksmart == True:
                sawyer.set_green_light()
            
            #Start adquiring data
            adquisition_cycle(test_time=10.0,direction=movement[0],category=movement[1])
            
            if sawyer._is_clicksmart == True:
                sawyer.set_blue_light()
            
            #Wait 2 seconds
            time.sleep(2)
            
        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing testing node")
    
    except rospy.ROSInterruptException:
        rospy.loginfo("Test unsuccesful, finishing testing node")
