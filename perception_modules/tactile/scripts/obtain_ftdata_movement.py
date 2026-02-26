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
from geometry_msgs.msg import WrenchStamped
import multiprocessing

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

def move_to_pos(robot,pos_command,step, rot_step):

        rospy.loginfo("Moving to %s, with a step of %s", pos_command,step)

        if pos_command == "None":
            pass
        elif pos_command == "X+":
            robot.cartesian_approach(move_position=[0.0,-step,0.0], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
        elif pos_command == "X-":
            robot.cartesian_approach(move_position=[0.0,step,0.0], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
        elif pos_command == "Y+":
            robot.cartesian_approach(move_position=[step,0.0,0.0], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
        elif pos_command == "Y-":
            robot.cartesian_approach(move_position=[-step,0.0,0.0], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
        elif pos_command == "Z+":
            robot.cartesian_approach(move_position=[0.0,0.0,step], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
        elif pos_command == "Z-":
            robot.cartesian_approach(move_position=[0.0,0.0,-step], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
        elif pos_command == "RotX+":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[0.0,rot_step,0.0],move_confirm=True)
        elif pos_command == "RotX-":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[0.0,-rot_step,0.0],move_confirm=True)
        elif pos_command == "RotY+":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[rot_step,0.0,0.0],move_confirm=True)
        elif pos_command == "RotY-":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[-rot_step,0.0,0.0],move_confirm=True)
        elif pos_command == "RotZ+":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[0.0,0.0,0.0174533*13],move_confirm=True)
        elif pos_command == "RotZ-":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[0.0,0.0,-0.0174533*13],move_confirm=True)
        elif pos_command == "X+Y+":
            robot.cartesian_approach(move_position=[step,-step,0.0], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
        elif pos_command == "X+Y-":
            robot.cartesian_approach(move_position=[-step,-step,0.0], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
        elif pos_command == "X-Y+":
            robot.cartesian_approach(move_position=[step,step,0.0], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
        elif pos_command == "X-Y-":
            robot.cartesian_approach(move_position=[-step,step,0.0], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)

def adquisition_cycle(robot,pos_command,step, rot_step, test_time=1.0, category=[]):

    global ft_samples
    ft_samples = np.zeros((1,12))
    arg = category

    try:
        
        #Move robot to desired position
        move_to_pos(robot,pos_command,step,rot_step)
	
        #Sleep 0.2 seconds to wait for transient time when moving
        if pos_command == "RotX+" or pos_command == "RotX-" or pos_command == "RotY+" or pos_command == "RotY-":
            time.sleep(2.0)
        else:
            time.sleep(2.0)

        #Subscribte to ATI ft data topic after moving
        sub = rospy.Subscriber("/robot/ati_ft_sensor_topic/", WrenchStamped, obtain_ftdata, callback_args=(arg[0][0],arg[0][1],arg[0][2],arg[0][3],arg[0][4],arg[0][5]))

        #Define the duration you want the subscriber to run (e.g., 10 seconds)
        duration = rospy.Duration(test_time)
        rospy.loginfo("Ft data adquisition started. It will run for %s seconds.", duration/1e9)

        #Start to count time after movement finished
        start_time = rospy.Time.now()
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
        ATI_net_path = "samples_" + current_datetime + "_" + pos_command + ".csv"
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
        rospy.signal_shutdown()
        

if __name__ == '__main__':
	
    #Order changed to do 45 deegree movements in free space
    test_movements = [
                    ("None", np.array([[1., 0., 0., 0., 0., 0.]])), 
                    ("X+", np.array([[1., 0., 0., 0., 0., 1.]])), 
                    ("X-", np.array([[1., 0., 0., 0., 1., 0.]])),
                    ("Y+", np.array([[1., 0., 0., 0., 1., 1.]])),
                    ("Y-", np.array([[1., 0., 0., 1., 0., 0.]])),
                    ("Z+", np.array([[1., 0., 0., 1., 0., 1.]])),
                    ("Z-", np.array([[1., 0., 0., 1., 1., 0.]])),
                    ("X+Y+", np.array([[1., 0., 1., 1., 0., 1.]])),
                    ("X+Y-", np.array([[1., 0., 1., 1., 1., 0.]])),
                    ("X-Y+", np.array([[1., 0., 1., 1., 1., 1.]])),
                    ("X-Y-", np.array([[1., 1., 0., 0., 0., 0.]])),
                    ("RotZ+", np.array([[1., 0., 1., 0., 1., 1.]])),
                    ("RotZ-", np.array([[1., 0., 1., 1., 0., 0.]])),
                    ("RotX+", np.array([[1., 0., 0., 1., 1., 1.]])),
                    ("RotX-", np.array([[1., 0., 1., 0., 0., 0.]])),
                    ("RotY+", np.array([[1., 0., 1., 0., 0., 1.]])),
                    ("RotY-", np.array([[1., 0., 1., 0., 1., 0.]]))
                    ]

    try:

        #Initialize Example node
        rospy.init_node('sawyer_ftdata_movement', anonymous=False)

        #Create sawyer robot instance
        sawyer = robot.SawyerRobot()
	
        #Open Sawyer Gripper
        sawyer.open_gripper()	
	
        #Wait 2 seconds
        time.sleep(2)

        #Move robot to assembly base position
        sawyer.move_to_cartesian_absolute(pos_no = 1)
	
        #Wait 1 second
        time.sleep(1)

        #Open Sawyer Gripper
        sawyer.close_gripper()	
	
        #Wait 2 seconds
        time.sleep(2)

        #Set speed (linear speed m/s, rotational speed rad/s)
        sawyer.set_speed(max_linear_speed = 0.01, max_linear_accel = 0.01, max_rotational_speed = 0.05,max_rotational_accel = 0.05)

        #Wait 2 seconds
        time.sleep(2)

        #Desired rotation step 1deg = 0.0175 approx.
        des_rot_step = 0.0174533*45
        #Desired translation step 0.001 m = 1mm
        des_trans_step = 0.005

        #Move to position adquire data for 5 seconds
        for movement in test_movements:
            
            direction = movement[0]
            category = movement[1]
	    
            #If rotating move up before moving to 45 deegres
            if direction == "RotX+":
                #Set speed (linear speed m/s, rotational speed rad/s)
                sawyer.set_speed(max_linear_speed = 0.2, max_linear_accel = 0.2, max_rotational_speed = 0.3,max_rotational_accel = 0.3)
		
                #Move above assembly
                sawyer.move_to_cartesian_relative(position=[0.0,0.0,0.35],orientation=[0.0,0.0,0.0],move_confirm=True)
            
                #Move robot to rotation movements base position
                sawyer.move_to_cartesian_absolute(pos_no = 2)
  		
            #Creating parallel processes
            adquisition_cycle(robot=sawyer, pos_command=direction, step=des_trans_step, rot_step=des_rot_step, test_time=3.0, category=category)
    
            # both processes finished
            rospy.loginfo("Movement Done to %s, returning to home for new movement", direction)

            #Wait 1 seconds
            time.sleep(1)
	    
            if direction == "RotX+" or direction == "RotX-" or direction == "RotY+" or direction == "RotY-":
            	#Move robot to rotation movements base position
                sawyer.move_to_cartesian_absolute(pos_no = 2)
            else:
                #Move robot to assembly base position
                sawyer.move_to_cartesian_absolute(pos_no = 1)

            #Wait 1 seconds
            time.sleep(1)

        rospy.loginfo("Testing sequence finished")
        
        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing testing node")

    except rospy.ROSInterruptException:
        rospy.loginfo("Test unsuccesful, finishing testing node")
        rospy.signal_shutdown()
