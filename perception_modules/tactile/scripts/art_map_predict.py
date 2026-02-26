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
from neural_networks import ART
from devices_interface import ATI_Net as ati
from geometry_msgs.msg import WrenchStamped

#Global variables for force and torque
#Need to be global to be updated by call back function in subscriber
ati_ft_data = np.zeros((1,6))
ati_ft_data[0][0] = 0.5
ati_ft_data[0][1] = 0.5
ati_ft_data[0][2] = 0.5
ati_ft_data[0][3] = 0.5
ati_ft_data[0][4] = 0.5
ati_ft_data[0][5] = 0.5

def obtain_ftdata(data):

    #Split wrench data
    ati_ft_data[0][0] = data.wrench.force.x
    ati_ft_data[0][1] = data.wrench.force.y
    ati_ft_data[0][2] = data.wrench.force.z
    ati_ft_data[0][3] = data.wrench.torque.x
    ati_ft_data[0][4] = data.wrench.torque.y
    ati_ft_data[0][5] = data.wrench.torque.z
    
def prediction_adquisition_cycle(artmap, robot):
    
    global ati_ft_data

    #Desired rotation step 1deg = 0.0175 approx.
    rot_step = 0.0175*3
    #Desired translation step 0.001 m = 1mm
    tran_step = 0.01

    pred = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-tran_step, 0.0, 0.0, 0.0, 0.0, 0.0],
            [tran_step, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -tran_step, 0.0, 0.0, 0.0, 0.0],
            [0.0, tran_step, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, tran_step, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -rot_step, 0.0, 0.0],
            [0.0, 0.0, 0.0, rot_step, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -rot_step, 0.0],
            [0.0, 0.0, 0.0, 0.0, rot_step, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -rot_step],
            [0.0, 0.0, 0.0, 0.0, 0.0, rot_step],
            [-tran_step, -tran_step, 0.0, 0.0, 0.0, 0.0],
            [-tran_step, tran_step, 0.0, 0.0, 0.0, 0.0],
            [tran_step, -tran_step, 0.0, 0.0, 0.0, 0.0],
            [tran_step, tran_step, 0.0, 0.0, 0.0, 0.0]
            ]
	    
            #[0.0, 0.0, -tran_step, 0.0, 0.0, 0.0],

    try:
        rospy.Subscriber("/robot/ati_ft_sensor_topic/", WrenchStamped, obtain_ftdata, queue_size=1)

        global_start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            
            complement_encode_ati_ft_data = np.concatenate((ati_ft_data[0], 1-ati_ft_data[0]))
            
            j = artmap.predict(complement_encode_ati_ft_data,rho_a=0.85)
                
            k = j[0]
            
            if j[0] is not None:
                robot.move_to_cartesian_relative(position=[pred[k][0],pred[k][1],pred[k][2]],orientation=[pred[k][3],pred[k][4],pred[k][5]], move_confirm=False, verbose=False)
                time.sleep(1.0)
            else:
                rospy.loginfo("Current input not recognized by Fuzzy ARTMAP: %s", j[1])
                rospy.loginfo("Category choices:\n None (0)\n X+ (1)\n X- (2)\n Y+ (3)\n Y- (4)\n Z+ (5)\n Z- (6)\n RotX+ (7)\n RotX- (8)\n RotY+ (9)\n RotY- (10)\n RotZ+ (11)\n RotZ- (12)\n X+Y+ (13)\n X+Y- (14)\n X-Y+ (15)\n X-Y- (16)\n")
                category = raw_input('Input Category for retraining or type "cancel" to avoid retraining: ')
                
                if category is not "cancel":
                    
                    categories = [
                                    ("None", np.array([[1., 0., 0., 0., 0., 0.]])), 
                                    ("X+", np.array([[1., 0., 0., 0., 0., 1.]])), 
                                    ("X-", np.array([[1., 0., 0., 0., 1., 0.]])),
                                    ("Y+", np.array([[1., 0., 0., 0., 1., 1.]])),
                                    ("Y-", np.array([[1., 0., 0., 1., 0., 0.]])),
                                    ("Z+", np.array([[1., 0., 0., 1., 0., 1.]])),
                                    ("Z-", np.array([[1., 0., 0., 1., 1., 0.]])),
                                    ("RotX+", np.array([[1., 0., 0., 1., 1., 1.]])),
                                    ("RotX-", np.array([[1., 0., 1., 0., 0., 0.]])),
                                    ("RotY+", np.array([[1., 0., 1., 0., 0., 1.]])),
                                    ("RotY-", np.array([[1., 0., 1., 0., 1., 0.]])),
                                    ("RotZ+", np.array([[1., 0., 1., 0., 1., 1.]])),
                                    ("RotZ-", np.array([[1., 0., 1., 1., 0., 0.]])),
                                    ("X+Y+", np.array([[1., 0., 1., 1., 0., 1.]])),
                                    ("X+Y-", np.array([[1., 0., 1., 1., 1., 0.]])),
                                    ("X-Y+", np.array([[1., 0., 1., 1., 1., 1.]])),
                                    ("X-Y-", np.array([[1., 1., 0., 0., 0., 0.]]))
                                    ]
                    
                    for movement in categories:
                        direction = movement[0]
                        if direction == category:
                            Ia = j[1]
                            Ib = movement[1]
                            
                            #Retrain Fuzzy ARTMAP (Give dimensions before adding complement)
                            artmap.train(Ia_dim=6, Ib_dim=6, train_path="/home/kid/ros_ws/src/sawyer_simulator/sawyer_sim_examples/train_02.csv", save_weights=True, load_csv=False,  Ia_retrain = Ia, Ib_retrain = Ib)

                            #Load trained weights
                            artmap.load_weights()
                    
                else: 
                        rospy.loginfo("Retrain canceled, resuming prediction sequence...")
        
    except rospy.ROSInterruptException:
	    rospy.signal_shutdown("Finishing testing node")


if __name__ == '__main__':
    
    try:
        #Initialize Example node
        rospy.init_node('sawyer_ftdata_predict', anonymous=False)

        #Create sawyer robot instance
        sawyer = robot.SawyerRobot()
        
        #Create FuzzyARTMAP Neural Network Instance
        artmap = ART.FuzzyArtMap()

        #Train Fuzzy ARTMAP (Give dimensions before adding complement)
        #artmap.train(Ia_dim=6, Ib_dim=6, train_path="/home/hector/ros_ws/ros_ws/src/sawyer_assembly/neural_networks/src/neural_networks/train_files/train_art_02.csv", save_weights=False)

        #Load trained weights
        artmap.load_weights()

        #Wait 2 seconds
        time.sleep(2)
        
        #Move robot to home Red Signaling movement
        sawyer.move_to_home()
        
        if sawyer._is_clicksmart == True:
            sawyer.set_red_light()
	
	    #Set speed (linear speed m/s, rotational speed rad/s)
        sawyer.set_speed(max_linear_speed = 0.3, max_linear_accel = 0.3, max_rotational_speed = 0.05, max_rotational_accel = 0.05)
	
        #Wait 2 seconds
        time.sleep(2)
            
        #Start adquiring data, blue signaling force prediction and movements
        
        if sawyer._is_clicksmart == True:
            sawyer.set_blue_light()
            
        prediction_adquisition_cycle(artmap=artmap, robot=sawyer)
            
        #Wait 2 seconds
        time.sleep(2)
        
        #Green signalizes idle state
        if sawyer._is_clicksmart == True:
                sawyer.set_green_light()
        
        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing testing node")
    
    except rospy.ROSInterruptException:
        
        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing testing node")
