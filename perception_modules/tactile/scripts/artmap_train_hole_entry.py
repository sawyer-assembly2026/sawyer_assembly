#!/usr/bin/env python

"""
Created on Tue Jan 27 12:35:06 2026

@author: Hector Quijada

Example sawyer moves above assembly, and starts moving downward in z by 0.001mm every second and reads variable J equal to force and scaled moment


"""

import rospy
import time
import math
import numpy as np
from devices_interface import robot_ctl as robot
from neural_networks import ART
from geometry_msgs.msg import WrenchStamped

#Global variables for force and torque
#Need to be global to be updated by call back function in subscriber
ati_ft_data = np.full((1, 6), 0.5)

#Callback function for subscriber
def obtain_ftdata(data):
    
    #Split wrench data and normalize
    ati_ft_data[0][0] = data.wrench.force.x
    ati_ft_data[0][1] = data.wrench.force.y
    ati_ft_data[0][2] = data.wrench.force.z
    ati_ft_data[0][3] = data.wrench.torque.x
    ati_ft_data[0][4] = data.wrench.torque.y
    ati_ft_data[0][5] = data.wrench.torque.z


def retrain_artmap(artmap,Ia,train_path, new_pattern=False, ideal_pattern=False):
    
    if new_pattern == True and ideal_pattern == False:
        rospy.loginfo("Current input not recognized by Fuzzy ARTMAP: %s", Ia)
    elif new_pattern == False and ideal_pattern ==False:
        rospy.loginfo("Current input generated a z loop, you can train this pattern  %s", Ia)
    elif new_pattern == False and ideal_pattern ==True:
        rospy.loginfo("Ideal pattern has been found: %s\n If desired it can be saved in knowledge base", Ia)
    
    #Prompt user to classify current force pattern
    rospy.loginfo("Category choices for movement in prediction:\n None (0)\n X- (1)\n X+ (2)\n Y- (3)\n Y+ (4)\n Z+ (5)\n RotX- (6)\n RotX+ (7)\n RotY- (8)\n RotY+ (9)\n RotZ- (10)\n RotZ+ (11)\n X-Y- (12)\n X-Y+ (13)\n X+Y- (14)\n X+Y+ (15)\n")
    category_prompt = raw_input('Input Category for retraining or type "cancel" to avoid retraining: ')
    
    if category_prompt != "cancel":
                
        j = int(category_prompt)
        Ib = obtain_train_categories(j)
        
        #Retrain Fuzzy ARTMAP
        artmap.train(Ia_dim=6, Ib_dim=6, train_path=train_path, save_weights=True, load_csv=False,  Ia_retrain = Ia, Ib_retrain = Ib)

        #Load trained weights
        artmap.load_weights()
        
    else:
        rospy.loginfo("Retrain canceled, resuming prediction sequence...")


def obtain_train_categories(n):
    
    categories = [
                    ("None", np.array([[1., 0., 0., 0., 0., 0.]])), 
                    ("X+", np.array([[1., 0., 0., 0., 0., 1.]])), 
                    ("X-", np.array([[1., 0., 0., 0., 1., 0.]])),
                    ("Y+", np.array([[1., 0., 0., 0., 1., 1.]])),
                    ("Y-", np.array([[1., 0., 0., 1., 0., 0.]])),
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
    
    return categories[n][1]

if __name__ == '__main__':
    
    trans_step = 0.006

    mov_pred = [
            [0.0, -0.001, 0.0, 0.0, 0.0, 0.0],
            [trans_step, -0.001, 0.0, 0.0, 0.0, 0.0],
            [-trans_step, -0.001, 0.0, 0.0, 0.0, 0.0],
            [trans_step, 0.005, 0.0, 0.0, 0.0, 0.0],
            [-trans_step, 0.005, 0.0, 0.0, 0.0, 0.0],
            [0.0, trans_step, 0.0, 0.0, 0.0, 0.0],
            [-trans_step, trans_step+0.004, 0.0, 0.0, 0.0, 0.0],
            [trans_step, trans_step+0.004, 0.0, 0.0, 0.0, 0.0],
            [0.0, trans_step+0.005, 0.0, 0.0, 0.0, 0.0],
            [-trans_step-0.002, trans_step, 0.0, 0.0, 0.0, 0.0],
            [trans_step+0.002, trans_step, 0.0, 0.0, 0.0, 0.0]
            ]

    try:
        #Initialize Example node
        rospy.init_node('sawyer_artmap_assembly', anonymous=False)

        #Create sawyer robot instance
        sawyer = robot.SawyerRobot()
        
        #Create FuzzyARTMAP Neural Network Instance
        artmap = ART.FuzzyArtMap()
	
        if sawyer._is_clicksmart == True:
            sawyer.set_green_light()

        #Train Fuzzy ARTMAP (Give dimensions before adding complement)
        artmap.train(Ia_dim=6, Ib_dim=6, train_path="/home/hector/ros_ws/src/sawyer_assembly/neural_networks/src/neural_networks/train_files/train_art_02.csv", save_weights=True, load_csv=True)

        #Load trained weights
        artmap.load_weights()

        #Wait 2 seconds
        time.sleep(2)
	
        if sawyer._is_clicksmart == True:
            sawyer.set_red_light()

        #Move above object
        sawyer.move_to_cartesian_absolute(pos_no=11)

        #Move near object
        sawyer.move_to_cartesian_absolute(pos_no=12)

        #Close Gripper
        sawyer.close_gripper()

        #Move up
        sawyer.move_to_cartesian_relative(position=[0.0,0.0,0.2], orientation=[0.0,0.0,0.0])

        #Move above assembly (safe position)
        sawyer.move_to_cartesian_absolute(pos_no=13)
	
    	#Wait 1 seconds
        time.sleep(1)
	
    	#Move above assembly (close position)
        sawyer.move_to_cartesian_absolute(pos_no=14)
        
        #Wait 2 seconds
        time.sleep(2)
            
        #Start adquiring data, blue signaling force prediction and movements
        
        if sawyer._is_clicksmart == True:
            sawyer.set_blue_light()
           
        #Set speed (linear speed m/s, rotational speed rad/s)	
        sawyer.set_speed(max_linear_speed=0.003, max_linear_accel = 0.003, max_rotational_speed = 0.01, max_rotational_accel = 0.01)	

        #Subscribe to force torque sensor topic
        rospy.Subscriber("/robot/ati_ft_sensor_topic/", WrenchStamped, obtain_ftdata, queue_size=1)

        for move in mov_pred:

            #Move to sampling position then down 
            sawyer.cartesian_approach(move_position=[move[0],move[1],0.0], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
            print(move[0],move[1],move[2])

            #Wait for transient
            time.sleep(1.0)
            
            sawyer.cartesian_approach(move_position=[0.0,0.0,-0.01], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
            
            #Wait for transient
            time.sleep(1.0)

            #Complement encode input
            complement_encode_ati_ft_data = np.concatenate((ati_ft_data[0], 1-ati_ft_data[0]))

            #Use Fuzzy ARTMAP to predict next movement
            category_predicted = artmap.predict(complement_encode_ati_ft_data,rho_a=0.85)
                        
            #Split output list
            prediction = category_predicted[0]
            print(prediction)
            Ia = category_predicted[1][:6].reshape(1,6)
            retrain_artmap(artmap, Ia, train_path="/home/hector/ros_ws/src/sawyer_assembly/neural_networks/src/neural_networks/train_files/train_art_02.csv", new_pattern=False)

            sawyer.cartesian_approach(move_position=[0.00,0.0,0.01], joint_speed=0.01 ,linear_speed=0.01, frecuency=100, verbose=False)
            
            #Move above assembly (close position)
            sawyer.move_to_cartesian_absolute(pos_no=14)

            #Wait 2 seconds
            time.sleep(2)
        
        #Green signalizes idle state
        if sawyer._is_clicksmart == True:
                sawyer.set_green_light()
        
        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing assembly node")
    
    except rospy.ROSInterruptException:
        
        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing assembly node")
