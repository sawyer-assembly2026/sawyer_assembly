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

def force_moment_factor(S=0.6):
    
    #Must declare as global within function
    global ati_ft_data
    
    #Get Force error
    Fx = ati_ft_data[0][0] - 0.5
    Fy = ati_ft_data[0][1] - 0.5
    Fz = ati_ft_data[0][2] - 0.5
    mx = ati_ft_data[0][3] - 0.5
    my = ati_ft_data[0][4] - 0.5
    mz = ati_ft_data[0][5] - 0.5
    
    #J factor for force measurement
    J = math.sqrt((Fx*Fx + Fy*Fy + Fz*Fz) + S*(mx*mx + my*my + mz*mz))
    
    return J

def obtain_move_prediction(prediction):
    
    #Prediction movement step in rotation and translation
    rot_step = 0.0174533*5.0 #0.5 deg
    tran_step = 0.002 #1mm
    
    #Z- movement not used
    
    # mov_pred = [
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [-tran_step, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [tran_step, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, -tran_step, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, tran_step, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.002, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, -rot_step, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, rot_step, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, -rot_step, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, rot_step, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, -rot_step],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, rot_step],
    #         [-tran_step, -tran_step, 0.0, 0.0, 0.0, 0.0],
    #         [-tran_step, tran_step, 0.0, 0.0, 0.0, 0.0],
    #         [tran_step, -tran_step, 0.0, 0.0, 0.0, 0.0],
    #         [tran_step, tran_step, 0.0, 0.0, 0.0, 0.0]
    #         ]

    mov_pred = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, tran_step, 0.0, 0.0, 0.0, 0.0],
            [0.0, -tran_step, 0.0, 0.0, 0.0, 0.0],
            [-tran_step, 0.0, 0.0, 0.0, 0.0, 0.0],
            [tran_step,0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.002, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, rot_step, 0.0],
            [0.0, 0.0, 0.0, 0.0, -rot_step, 0.0],
            [0.0, 0.0, 0.0, rot_step, 0.0, 0.0],
            [0.0, 0.0, 0.0, -rot_step, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -rot_step],
            [0.0, 0.0, 0.0, 0.0, 0.0, rot_step],
            [-tran_step, tran_step, 0.0, 0.0, 0.0, 0.0],
            [tran_step, tran_step, 0.0, 0.0, 0.0, 0.0],
            [-tran_step, -tran_step, 0.0, 0.0, 0.0, 0.0],
            [tran_step, -tran_step, 0.0, 0.0, 0.0, 0.0]
            ]
    
    return mov_pred[prediction]

def obtain_move_categories(n):
    
    move_categories = [
                    "None", 
                    "X-", 
                    "X+",
                    "Y-",
                    "Y+",
                    "Z+",
                    "RotX-",
                    "RotX+", 
                    "RotY-",
                    "RotY+",
                    "RotZ-",
                    "RotZ+",
                    "X-Y-",
                    "X-Y+",
                    "X+Y-",
                    "X+Y+"
                    ]
    
    return move_categories[n]

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
        
def movement_result_classifier(artmap, J_prev, J_current, J_limit, pattern):
    
    #If current forces are lower than 10% of the forces before the prediction movement, retrain artmap with force pattern
    if J_current < 0.1*J_prev:
        retrain_artmap(artmap=artmap, Ia=pattern, train_path="/home/kid/ros_ws/src/sawyer_simulator/sawyer_sim_examples/train_02.csv", ideal_pattern=True)
    elif J_current > 0.1*J_prev and J_current < J_limit:
        pass

def assembly_cycle(artmap, robot):
    
    #Desired end effector value when going down
    zd = 0.04
    
    #Safety z offset
    z_offset = 0.005

    #Value for the hole entrance
    z_hole = 0.0928 - z_offset
	
    #Prediction Threshold
    epsilon = 0.18

    #Maximum force limit with J factor
    J_max = 0.50
    
    #Counter Flags to detect z up and down loop
    z_down = False
    z_up = False
    z_counter = 0
    z_threshold = 2

    #Step Counter
    steps = 1

    try:
        
        #Obtain assembly start time
        start_time = time.clock()
        
        #Subscribe to force torque sensor topic
        rospy.Subscriber("/robot/ati_ft_sensor_topic/", WrenchStamped, obtain_ftdata, queue_size=1)

    	#First approach to hole
        move_confirm = robot.cartesian_approach(move_position=[0.0,0.0,-0.008], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)
        
        #Start assembly cycle after going down and being near the hole
        while not rospy.is_shutdown():
            
            #Obtain Force moment factor
            J = force_moment_factor(S=0.6)
            
            #Define J_prev as the current J factor before any prediction
            J_prev = J
            
            #Extract current z value
            z_current = sawyer.current_endpoint_pose()[0][2]
            
            #First check the max force limit has not been reached
            if J < J_max:
                
                #Check current assembly status
                
                #J below the prediction threshold and within target z value, assembly has finished
                if J < epsilon and z_current <= zd:
                    end_time = time.clock()
                    execution_time = end_time - start_time
                    rospy.loginfo("Assembly finished!")
                    rospy.loginfo("Final Z value = %s", z_current)
                    rospy.loginfo("Current J threshold = %s", epsilon)
                    rospy.loginfo("Final assembly time = %s seconds", execution_time)

                    rospy.signal_shutdown("Finishing assembly node")
                
                #If J below the prediction threshold and not yet at target, move down in z axis
                elif J < epsilon and z_current > zd:

                    #This movement function utilizes sawyer inverse kinematics to interpolate between to cartesian points at a certain speed
                    if z_current >= z_hole:
                        move_confirm = robot.cartesian_approach(move_position=[0.0,0.0,-0.003], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=False)
                    elif z_current < z_hole:
                        move_confirm = robot.cartesian_approach(move_position=[0.0,0.0,-0.01], joint_speed=0.01 ,linear_speed=0.01, frecuency=100, verbose=False)


                    rospy.loginfo("Moving down in z axis")
                    
                    z_down = True

                    #Obtain Force moment factor
                    J = force_moment_factor(S=0.6)

                    #Current cases for assembly
                    rospy.loginfo("--------------------------")
                    rospy.loginfo("J = %s", J)
                    rospy.loginfo("Current Z = %s", z_current)
                    rospy.loginfo("Current step = %s", steps)
            
                    # #Complement encode input
                    # complement_encode_ati_ft_data = np.concatenate((ati_ft_data[0], 1-ati_ft_data[0]))
                    
                    # #Use Fuzzy ARTMAP to predict next movement
                    # category_predicted = artmap.predict(complement_encode_ati_ft_data,rho_a=0.85)
                    
                    # #Split output list
                    # prediction = category_predicted[0]
                    # Ia = category_predicted[1][:6].reshape(1,6)

                    # retrain_artmap(artmap, Ia, train_path="/home/hector/ros_ws/src/sawyer_assembly/neural_networks/src/neural_networks/train_files/train_art_02.csv", new_pattern=False)

                #If J above the prediction threshold and not yet at target, use Fuzzy ARTMAP to predict next movement
                elif J > epsilon and z_current > zd:
                    
                    #Complement encode input
                    complement_encode_ati_ft_data = np.concatenate((ati_ft_data[0], 1-ati_ft_data[0]))
                    
                    #Use Fuzzy ARTMAP to predict next movement
                    category_predicted = artmap.predict(complement_encode_ati_ft_data,rho_a=0.85)
                    
                    #Split output list
                    prediction = category_predicted[0]
                    Ia = category_predicted[1][:6].reshape(1,6)
                 
                    
                    #If value found within current trained categories proceed to move in predicted direction
                    if prediction != None:
                        
                        #Obtain movement vector
                        move_pred = obtain_move_prediction(prediction)   

                        #Check if a up down loop in the z axis exists and retrain fuzzy artmap if desired
                        if prediction == 5:
                            z_up = True
                        else:
                            z_down = False
                            z_counter = 0
                            
                        if z_down and z_up:
                            z_counter += 1
                            z_down = False
                            z_up = False
                            
                        if z_counter >= z_threshold:
                            retrain_artmap(artmap, Ia, train_path="/home/hector/ros_ws/src/sawyer_assembly/neural_networks/src/neural_networks/train_files/train_art_02.csv", new_pattern=False)
                            z_counter = 0
                        
                        #No movement category(0) indicates we are at optimal contact during assembly
                        #However a loop can be induced in which we stay at No movement condition instead of going down
                        if prediction > 0:
                            #Translation movement
                            if move_pred[3] == 0.0 and move_pred[4] == 0.0 and move_pred[5] == 0.0:
                                if z_current < z_hole + 0.003:
                                    move_confirm = robot.cartesian_approach(move_position=[move_pred[0],move_pred[1],move_pred[2]], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=False)
                                    rospy.loginfo("In hole, executing small movements")
                                else:
                                    move_confirm = robot.cartesian_approach(move_position=[move_pred[0]*4.0,move_pred[1]*4.0,move_pred[2]], joint_speed=0.01 ,linear_speed=0.01, frecuency=100, verbose=False)						
                                    rospy.loginfo("Above hole, executing regular movements")
                            #Rotation movement
                            elif move_pred[0] == 0.0 and move_pred[1] == 0.0 and move_pred[2] == 0.0:
                                robot.rotation_interpolation(move_orientation=[move_pred[3],move_pred[4],move_pred[5]], joint_speed=0.001, final_time=2.0, frecuency=100, verbose=False)
                        

                            #Obtain Force moment factor
                            J = force_moment_factor(S=0.6)

                            #Current cases for assembly
                            rospy.loginfo("--------------------------")
                            rospy.loginfo("J = %s", J)
                            rospy.loginfo("Current Z = %s", z_current)
                            rospy.loginfo("Current step = %s", steps)

                            str_mov = obtain_move_categories(prediction)
                            rospy.loginfo("Fuzzy ARTMAP movement done in %s direction", str_mov)
                            rospy.loginfo("Move prediction: %s", move_pred)
                            

                            #Wait for transient to pass after movement
                            time.sleep(1.0)
                            
                            #Obtain current force moment factor
                            J_current = force_moment_factor(S=0.6)
                            
                            #Classify result as good or excellent only if it is not a no movement or a z movement upwards
                            #That is because when searching the hole at the beginning of the assembly a z movement upwards will reduce force significantly
                            #But does not mean it is a pattern we want to save to the knowledge base
                            if prediction != 5 and prediction >0:
                                #Classify how good the movement was and act accordingly
                                movement_result_classifier(artmap, J_prev, J_current, J_max, Ia)
                            
                        #If inside a loop with no movement increase epsilon to continue moving
                        else:
                            rospy.loginfo("No movement predicted, increasing J threshold")
                            epsilon += 0.01
                        
                    #Now if category was not found within one of the force patterns prompt user to retrain weights
                    else:
                        retrain_artmap(artmap, Ia, train_path="/home/hector/ros_ws/src/sawyer_assembly/neural_networks/src/neural_networks/train_files/train_art_02.csv", new_pattern=True)
                
                #If J above epsilon and we are at the desired z value, 
                elif J > epsilon and z_current <= zd:
                    end_time = time.clock()
                    execution_time = end_time - start_time
                    rospy.loginfo("Assembly finished!")
                    rospy.loginfo("Final Z value = %s", z_current)
                    rospy.loginfo("Current J threshold = %s", epsilon)
                    rospy.loginfo("Final assembly time = %s", execution_time)
                    
                    rospy.signal_shutdown("Finishing assembly node")
                
                steps +=1        

            else:
                rospy.loginfo("Assembly stopped, maximum force exceeded")
                rospy.signal_shutdown("Finishing assembly node")
            
    
    except rospy.ROSInterruptException:

        rospy.signal_shutdown("Finishing testing node")


if __name__ == '__main__':
    
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
        artmap.train(Ia_dim=6, Ib_dim=6, train_path="/home/hector/ros_ws/src/sawyer_assembly/neural_networks/src/neural_networks/train_files/train_art_02.csv", save_weights=False, load_csv=True)

        #Load trained weights
        artmap.load_weights()

        #Wait 2 seconds
        time.sleep(2)
	
        if sawyer._is_clicksmart == True:
            sawyer.set_red_light()

        #Close Gripper
        sawyer.close_gripper()

        #Move above assembly (safe position)
        sawyer.move_to_cartesian_absolute(pos_no=13)
	
    	#Wait 1 seconds
        time.sleep(1)
	
    	#Move above assembly (close position)
        #sawyer.move_to_cartesian_absolute(pos_no=14)
        sawyer.move_to_cartesian_absolute(pos_no=16)
        
        #Wait 2 seconds
        time.sleep(2)
            
        #Start adquiring data, blue signaling force prediction and movements
        
        if sawyer._is_clicksmart == True:
            sawyer.set_blue_light()
           
        #Set speed (linear speed m/s, rotational speed rad/s)	
        sawyer.set_speed(max_linear_speed=0.001, max_linear_accel = 0.001, max_rotational_speed = 0.01, max_rotational_accel = 0.01)	
	
        assembly_cycle(artmap=artmap, robot=sawyer)
            
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
