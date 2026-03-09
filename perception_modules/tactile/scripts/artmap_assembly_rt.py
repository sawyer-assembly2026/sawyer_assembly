#!/usr/bin/env python

"""
Created on Tue Jan 27 12:35:06 2026

@author: Hector Quijada

Example sawyer moves above assembly, and starts moving downward in z by 0.001mm every second and reads variable J equal to force and scaled moment


"""

import sys
import os
import subprocess
import rospy
import rospkg
import cv2
import cv_bridge
from sensor_msgs.msg import Image
import time
import time
import math
import numpy as np
from intera_interface import (
    Navigator,
    HeadDisplay,
    Head
)
from devices_interface import robot_ctl as robot
from neural_networks import ART
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    WrenchStamped
)
from tf.transformations import (
    quaternion_from_euler
)

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
    rot_step = 0.0174533*3.0 #0.5 deg
    tran_step = 0.01 #1mm
    
    #Z- movement not used
    
    mov_pred = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-tran_step, 0.0, 0.0, 0.0, 0.0, 0.0],
            [tran_step, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -tran_step, 0.0, 0.0, 0.0, 0.0],
            [0.0, tran_step, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.002, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, rot_step, 0.0, 0.0],
            [0.0, 0.0, 0.0, -rot_step, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, rot_step, 0.0],
            [0.0, 0.0, 0.0, 0.0, -rot_step, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -rot_step],
            [0.0, 0.0, 0.0, 0.0, 0.0, rot_step],
            [-tran_step, -tran_step, 0.0, 0.0, 0.0, 0.0],
            [-tran_step, tran_step, 0.0, 0.0, 0.0, 0.0],
            [tran_step, -tran_step, 0.0, 0.0, 0.0, 0.0],
            [tran_step, tran_step, 0.0, 0.0, 0.0, 0.0]
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

def retrain_artmap(artmap,robot,base_image,Ia,J,eps,train_path, new_pattern=False, ideal_pattern=False):
    
    #Get current package path
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('tactile')

    #Create image publisher for head display
    image_pub = rospy.Publisher('/robot/head_display', Image, latch=True, queue_size=10)

    if new_pattern == True and ideal_pattern == False:
        rospy.loginfo("Current input not recognized by Fuzzy ARTMAP: %s", Ia)
    elif new_pattern == False and ideal_pattern ==False:
        rospy.loginfo("Current input generated a z loop, you can train this pattern  %s", Ia)
    elif new_pattern == False and ideal_pattern ==True:
        rospy.loginfo("Ideal pattern has been found: %s\n If desired it can be saved in knowledge base", Ia)
    
    #Prompt user to classify current force pattern
    if robot._is_clicksmart == True:
        robot.set_red_light()

    # Add force input text to the image
    force_input = str(Ia)
    img_force = add_image_text(text_input=force_input,base_image=base_image, position=(75, 100), font_scale=0.72, color=(0, 0, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
    # Add J input text to the image
    J_input = str(J)
    img_force_J = add_image_text(text_input=J_input,base_image=img_force, position=(335, 138), font_scale=0.72, color=(0, 0, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
        # Add J input text to the image
    eps_input = str(eps)
    img_force_J_eps = add_image_text(text_input=eps_input,base_image=img_force_J, position=(790, 138), font_scale=0.72, color=(0, 0, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)

    #Initializa Navigator 
    inital_state = navigator.get_wheel_state("right_wheel")
    square_button_state = navigator.get_button_state("right_button_square")
    ok_button_state = navigator.get_button_state("right_button_ok")
    back_button_state = navigator.get_button_state("right_button_back")
    
    #Release button if pressed
    while square_button_state != 0 or ok_button_state != 0:
        #Display Release button image
        image_path = package_path + "/share/assembly_images/releasebuttons.png"
        headdisplay.display_image(image_path)

    #Button signal debounce
    time.sleep(0.25)

    while square_button_state == 0 and ok_button_state == 0:

        #Update button states
        current_state = navigator.get_wheel_state("right_wheel")
        square_button_state = navigator.get_button_state("right_button_square")
        ok_button_state = navigator.get_button_state("right_button_ok")
        back_button_state = navigator.get_button_state("right_button_back")

        category_input = abs(inital_state - current_state)

        #Update inital state in case outside limits
        if category_input > 15:
            inital_state = navigator.get_wheel_state("right_wheel")
                    
        category_input_str = str(category_input)

        # Add category input text to the image
        img_force_class = add_image_text(text_input=category_input_str,base_image=img_force_J_eps, position=(675, 420), font_scale=8.0, color=(250, 225, 100), thickness=5, font=cv2.FONT_HERSHEY_SIMPLEX)

        #Convert to ros image topic
        img_msg = cv_bridge.CvBridge().cv2_to_imgmsg(img_force_class, encoding="bgr8")

        #Publish in head image display
        image_pub.publish(img_msg)
                    
        if back_button_state != 0:
            if new_pattern == True:
                #Remove weights destined to retrain
                artmap.remove_prediction_weight()
            return True

        if square_button_state != 0:
            category_input = "cancel"

    if category_input is not "cancel":
                
        j = int(category_input)
        Ib = obtain_train_categories(j)
        
        if robot._is_clicksmart == True:
            robot.set_blue_light()

        #Display Fuzzy ARTMAP retrain image
        onoing_image_path = package_path + "/share/assembly_images/artmap_retrain.png"
        ongoing_img = cv2.imread(onoing_image_path)

        #Convert to ros image topic
        img_msg = cv_bridge.CvBridge().cv2_to_imgmsg(ongoing_img, encoding="bgr8")

        #Publish in head image display
        image_pub.publish(img_msg)

        #Retrain Fuzzy ARTMAP
        artmap.train(Ia_dim=6, Ib_dim=6, train_path=train_path, save_weights=True, load_csv=False,  Ia_retrain = Ia, Ib_retrain = Ib)

        #Load trained weights
        artmap.load_weights()

        return False
        
    else:
        rospy.loginfo("Retrain canceled, resuming prediction sequence...")

        if robot._is_clicksmart == True:
            robot.set_blue_light()

        #Display ongoing base image to add text
        onoing_image_path = package_path + "/share/assembly_images/ongoing.png"
        ongoing_img = cv2.imread(onoing_image_path)

        #Convert to ros image topic
        img_msg = cv_bridge.CvBridge().cv2_to_imgmsg(ongoing_img, encoding="bgr8")

        #Publish in head image display
        image_pub.publish(img_msg)

        if new_pattern == True:
            #Remove weights destined to retrain
            artmap.remove_prediction_weight()

        return False
        
def movement_result_classifier(artmap, J_prev, J_current, J_limit, pattern):
    
    #If current forces are lower than 10% of the forces before the prediction movement, retrain artmap with force pattern
    if J_current < 0.1*J_prev:
        retrain_artmap(artmap=artmap, Ia=pattern, train_path="/home/kid/ros_ws/src/sawyer_simulator/sawyer_sim_examples/train_02.csv", ideal_pattern=True)
    elif J_current > 0.1*J_prev and J_current < J_limit:
        pass

def add_image_text(text_input,base_image, position, font_scale, color, thickness, font=cv2.FONT_HERSHEY_SIMPLEX):

    # Add text image to topic
    img_iterate = base_image.copy()
    return cv2.putText(img_iterate, text_input, position, font, font_scale, color, thickness, cv2.LINE_AA)

def send_ik_movement(move_pred,robot,ik_step,move_euler,orientation=False):

    if orientation:
        #Get current endpoint pose
        current_position, current_orientation = robot.current_endpoint_pose(quaternion=False)
    else:
        #Get current endpoint pose
        current_position, current_orientation = robot.current_endpoint_pose(quaternion=True)

    current_position[0] = current_position[0] + move_pred[1]
    current_position[1] = current_position[1] - move_pred[0]
    current_position[2] = current_position[2] + move_pred[2]

    if orientation:
        move_euler[0] = current_orientation[0] + move_pred[3]
        move_euler[1] = current_orientation[1] + move_pred[4]
        move_euler[2] = current_orientation[2] + move_pred[5]

        roll = move_euler[0]
        pitch = move_euler[1]
        yaw = move_euler[2]

        move_quaternion = quaternion_from_euler(roll, pitch, yaw)
        
        current_orientation = [0.0,0.0,0.0,0.0]

        current_orientation[0] = float(move_quaternion[0])
        current_orientation[1] = float(move_quaternion[1])
        current_orientation[2] = float(move_quaternion[2])
        current_orientation[3] = float(move_quaternion[3])
    
    else:
        pass

    #Request inverse kinematics at position ik_step with tip desired depending on EAOT
    joint_angles = robot.ik_srv(current_position, current_orientation)

    #Send position
    if joint_angles:
            robot._limb.set_joint_positions(joint_angles)
    else:
        rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

#Assembly function
def assembly_cycle(artmap, train_file, robot, head_display, image_pub, train_entry=False, rho=0.9, eps=0.18):
    
    #Assembly finished flag
    assembly_success = False

    #Desired end effector value when going down
    zd = 0.04
    
    #Safety z offset
    z_offset = 0.005

    #Value for the hole entrance
    z_hole = 0.0928 - z_offset

    #Prediction Threshold
    epsilon = eps

    #Maximum force limit with J factor
    J_max = 0.50
    
    #Counter Flags to detect z up and down loop
    z_down = False
    z_up = False
    z_counter = 0
    z_threshold = 2

    #Step Counter
    iterations = 0

    try:

        #Get current package path
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('tactile')

        #Get relative train path to package
        rospack = rospkg.RosPack()
        nn_path = rospack.get_path('neural_networks')
        train_file_no = str(train_file)
        art_train_path = nn_path + "/src/neural_networks/train_files/train_art_" + train_file_no + ".csv"

        #Read retrain base image to add text
        retrain_image_path = package_path + "/share/retrain_home.png"
        retrain_img = cv2.imread(retrain_image_path)

        #Read assembly base image to add text
        onoing_image_path = package_path + "/share/assembly_images/ongoing.png"
        ongoing_img = cv2.imread(onoing_image_path)

        #Obtain assembly start time
        start_time = time.clock()
        
        #Subscribe to force torque sensor topic
        rospy.Subscriber("/robot/ati_ft_sensor_topic/", WrenchStamped, obtain_ftdata, queue_size=1)

    	#First approach to hole
        move_confirm = robot.cartesian_approach(move_position=[0.0,0.0,-0.005], joint_speed=0.001 ,linear_speed=0.001, frecuency=100, verbose=True)

        #Go back flag
        back_flag = False

        #Set slow speed flag
        set_slow_speed = False

        #Set desired speed (m/s)
        robot._limb.set_joint_position_speed(0.003)

        ik_step = Pose()
        move_euler = [0.0,0.0,0.0]

        r = rospy.Rate(100)

        #Start assembly cycle after going down and being near the hole
        while not rospy.is_shutdown() and back_flag == False:
            
            #Obtain Force moment factor
            J = force_moment_factor(S=0.6)
            
            #Define J_prev as the current J factor before any prediction
            #J_prev = J

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

                    assembly_success = True
                    break
                
                #If J below the prediction threshold and not yet at target, move down in z axis
                elif J < epsilon and z_current > zd:

                    #Convert values to string
                    #z_current_str = str(z_current)
                    #J_str = str(J)
                    #pred_str = "Z-"
                    #steps_str = str(steps)

                    # Add assembly text info
                    #img_z_value = add_image_text(text_input=z_current_str,base_image=ongoing_img, position=(520, 258), font_scale=0.85, color=(0, 0, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
                    #img_J_value = add_image_text(text_input=J_str,base_image=img_z_value, position=(520, 313), font_scale=0.85, color=(0, 0, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
                    #img_next_prediction = add_image_text(text_input=pred_str,base_image=ongoing_img, position=(520, 368), font_scale=0.85, color=(0, 0, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
                    #img_steps = add_image_text(text_input=steps_str,base_image=img_next_prediction, position=(520, 424), font_scale=0.85, color=(0, 0, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)

                    #Convert to ros image topic
                    #img_msg = cv_bridge.CvBridge().cv2_to_imgmsg(img_next_prediction, encoding="bgr8")

                    #Publish in head image display
                    #image_pub.publish(img_msg)

                    send_ik_movement(move_pred=[0.0, 0.0, -0.01, 0.0, 0.0, 0.0],robot=robot, ik_step = ik_step,move_euler = move_euler)

                    #rospy.loginfo("Moving down in z axis")
                
                    #z_down = True

                    if train_entry:
                        #Obtain Force moment factor
                        J = force_moment_factor(S=0.6)

                        if J > epsilon:

                            if robot._is_clicksmart == True:
                                robot.set_red_light()

                            #Complement encode input
                            complement_encode_ati_ft_data = np.concatenate((ati_ft_data[0], 1-ati_ft_data[0]))
                            
                            #Use Fuzzy ARTMAP to predict next movement
                            category_predicted = artmap.predict(complement_encode_ati_ft_data,rho_a=rho)
                            
                            #Split output list
                            prediction = category_predicted[0]
                            Ia = category_predicted[1][:6].reshape(1,6)

                            send_ik_movement(move_pred=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],robot=robot, ik_step = ik_step,move_euler = move_euler)
                            back_flag = retrain_artmap(artmap, robot, retrain_img, Ia, J,eps,train_path=art_train_path, new_pattern=False)

                #If J above the prediction threshold and not yet at target, use Fuzzy ARTMAP to predict next movement
                elif J > epsilon and z_current > zd:
                    
                    #Complement encode input
                    complement_encode_ati_ft_data = np.concatenate((ati_ft_data[0], 1-ati_ft_data[0]))
                    
                    #Use Fuzzy ARTMAP to predict next movement
                    category_predicted = artmap.predict(complement_encode_ati_ft_data,rho_a=rho)
                    
                    #Split output list
                    prediction = category_predicted[0]
                    Ia = category_predicted[1][:6].reshape(1,6)
                 
                    #If value found within current trained categories proceed to move in predicted direction
                    if prediction != None:

                        #Obtain movement vector
                        move_pred = obtain_move_prediction(prediction)   

                        #Convert values to string
                        #z_current_str = str(z_current)
                        #J_str = str(J)
                        #pred_str = obtain_move_categories(prediction)
                        #steps_str = str(steps)
                        #print(pred_str)

                        # Add assembly text info
                        #img_z_value = add_image_text(text_input=z_current_str,base_image=ongoing_img, position=(520, 258), font_scale=0.85, color=(0, 0, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
                        #img_J_value = add_image_text(text_input=J_str,base_image=img_z_value, position=(520, 313), font_scale=0.85, color=(0, 0, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
                        #img_next_prediction = add_image_text(text_input=pred_str,base_image=ongoing_img, position=(520, 368), font_scale=0.85, color=(0, 0, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
                        #img_steps = add_image_text(text_input=steps_str,base_image=img_next_prediction, position=(520, 424), font_scale=0.85, color=(0, 0, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)

                        #Convert to ros image topic
                        #img_msg = cv_bridge.CvBridge().cv2_to_imgmsg(img_next_prediction, encoding="bgr8")

                        #Publish in head image display
                        #image_pub.publish(img_msg)

                        #Check if a up down loop in the z axis exists and retrain fuzzy artmap if desired
                        #if prediction == 5:
                        #    z_up = True
                        #else:
                        #    z_down = False
                        #    z_counter = 0
                            
                        #if z_down and z_up:
                        #    z_counter += 1
                        #    z_down = False
                        #    z_up = False
                            
                        #if z_counter >= z_threshold:
                        #    back_flag = retrain_artmap(artmap, robot, retrain_img, Ia, J,eps,train_path=art_train_path, new_pattern=False)
                        #    z_counter = 0
                        
                        #No movement category(0) indicates we are at optimal contact during assembly
                        #However a loop can be induced in which we stay at No movement condition instead of going down
                        if prediction > 0:
                            
                            if move_pred[3] != 0.0 or move_pred[4] != 0.0 or move_pred[5] != 0.0:
                                send_ik_movement(move_pred=move_pred,robot=robot, ik_step = ik_step,move_euler = move_euler,orientation=True)
                            else:
                                send_ik_movement(move_pred=move_pred,robot=robot, ik_step = ik_step,move_euler = move_euler,orientation=False)
                            
                            #Obtain current force moment factor
                            #J_current = force_moment_factor(S=0.6)
                            
                            #Classify result as good or excellent only if it is not a no movement or a z movement upwards
                            #That is because when searching the hole at the beginning of the assembly a z movement upwards will reduce force significantly
                            #But does not mean it is a pattern we want to save to the knowledge base
                            #if prediction != 5 and prediction >0:
                                #Classify how good the movement was and act accordingly
                            #    movement_result_classifier(artmap, J_prev, J_current, J_max, Ia)
                            
                        #If inside a loop with no movement increase epsilon to continue moving
                        else:
                            if train_entry:
                                send_ik_movement(move_pred=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],robot=robot, ik_step = ik_step,move_euler = move_euler)
                                back_flag = retrain_artmap(artmap, robot, retrain_img, Ia, J,eps,train_path=art_train_path, new_pattern=False)
                            else:
                                rospy.loginfo("No movement predicted, increasing J threshold")
                                epsilon += 0.01
                        
                    #Now if category was not found within one of the force patterns prompt user to retrain weights
                    else:
                        send_ik_movement(move_pred=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],robot=robot, ik_step = ik_step,move_euler = move_euler)
                        back_flag = retrain_artmap(artmap, robot, retrain_img, Ia, J,eps,train_path=art_train_path, new_pattern=True)
                
                #If J above epsilon and we are at the desired z value, 
                elif J > epsilon and z_current <= zd:
                    end_time = time.clock()
                    execution_time = end_time - start_time
                    rospy.loginfo("Assembly finished!")
                    rospy.loginfo("Final Z value = %s", z_current)
                    rospy.loginfo("Current J threshold = %s", epsilon)
                    rospy.loginfo("Final assembly time = %s", execution_time)

                    assembly_success = True
                    break
                
                iterations +=1        

            else:
                rospy.loginfo("Assembly stopped, maximum force exceeded")
                assembly_success = False
                break
            

            r.sleep()

    except rospy.ROSInterruptException:

        rospy.signal_shutdown("Finishing testing node")

    end_time = time.clock()
    execution_time = end_time - start_time
    frecuency = iterations/execution_time
    rospy.loginfo("Prediction Frecuency, %s", frecuency)
    return assembly_success, execution_time

if __name__ == '__main__':
    
    try:

        """
        Initialize Assembly ROS node
        """
        rospy.init_node('sawyer_artmap_assembly', anonymous=False)
        
        #Create image publisher for head display
        image_pub = rospy.Publisher('/robot/head_display', Image, latch=True, queue_size=10)
        
        finish_process = False
        restart_assembly = False
        ati_open = False

        """
        Class instances for robot control and Fuzzy ARTMAP Training
        """

        #Create Head Display instance
        headdisplay = HeadDisplay()
        #Create navigator instance
        navigator = Navigator()

        #Select knowledge base
        train_file = 3

        """
        Tactile Package Path
        """
        #Get current package path
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('tactile')

        while not finish_process:
            
            if not restart_assembly:
                """
                START PAGE
                Select Mode
                """
                #Create sawyer robot instance
                sawyer = robot.SawyerRobot()

                #Create FuzzyARTMAP Neural Network Instance
                artmap = ART.FuzzyArtMap()

                #Display Start page select mode image
                image_path = package_path + "/share/assembly_images/mode.png"
                headdisplay.display_image(image_path)

                #Set Idle State
                if sawyer._is_clicksmart == True:
                    sawyer.set_green_light()

                #Set train entry flag, if true train entry sequence starts whenever going down
                train_entry = False

                #Initialize button state
                #Square for Training mode
                #Show for Normal mode
                square_button_state = navigator.get_button_state("right_button_square")
                show_button_state = navigator.get_button_state("right_button_show")
                back_button_state = navigator.get_button_state("right_button_back")

                #Release button if pressed
                while square_button_state != 0 or show_button_state != 0 or back_button_state != 0:

                    #Initialize button state
                    #Square for Training mode
                    #Show for Normal mode
                    square_button_state = navigator.get_button_state("right_button_square")
                    show_button_state = navigator.get_button_state("right_button_show")
                    back_button_state = navigator.get_button_state("right_button_back")

                    #Display Release button image
                    image_path = package_path + "/share/assembly_images/releasebuttons.png"
                    headdisplay.display_image(image_path)

                #Button signal debounce
                time.sleep(0.1)
                
                #Display database selection image
                image_path = package_path + "/share/assembly_images/mode.png"
                headdisplay.display_image(image_path)

                while square_button_state == 0 and show_button_state == 0 and back_button_state == 0:

                    square_button_state = navigator.get_button_state("right_button_square")
                    show_button_state = navigator.get_button_state("right_button_show")
                    back_button_state = navigator.get_button_state("right_button_back")

                    if square_button_state != 0:
                        train_entry = True
                    elif show_button_state != 0:
                        train_entry = False
                    elif back_button_state != 0:

                        #Set Exit, Error State
                        if sawyer._is_clicksmart == True:
                            sawyer.set_green_light()

                        #Display database selection image
                        image_path = package_path + "/share/sawyer_sdk_research.png"
                        headdisplay.display_image(image_path)

                        sys.exit()

                """
                START PAGE
                Database selection
                """
                #Display database selection image
                image_path = package_path + "/share/assembly_images/select_database.png"
                headdisplay.display_image(image_path)

                #Set save weights flag, if true knowledge base goes back to primitive
                save_weights = False

                #Initialize button state
                #Square for New database
                #Ok for current database
                square_button_state = navigator.get_button_state("right_button_square")
                ok_button_state = navigator.get_button_state("right_button_ok")

                #Release button if pressed
                while square_button_state != 0 or ok_button_state != 0:
                    
                    #Square for New database
                    #Ok for current database
                    square_button_state = navigator.get_button_state("right_button_square")
                    ok_button_state = navigator.get_button_state("right_button_ok")

                    #Display Release button image
                    image_path = package_path + "/share/assembly_images/releasebuttons.png"
                    headdisplay.display_image(image_path)

                #Button signal debounce
                time.sleep(0.1)

                #Display database selection image
                image_path = package_path + "/share/assembly_images/select_database.png"
                headdisplay.display_image(image_path)
                
                #Select database to be used
                #Square restarts knowledge base to primitive
                #Wheel center(Ok) uses current knowledge base
                while square_button_state == 0 and ok_button_state == 0:

                    square_button_state = navigator.get_button_state("right_button_square")
                    ok_button_state = navigator.get_button_state("right_button_ok")

                    if square_button_state != 0:
                        save_weights = True
                    elif ok_button_state != 0:
                        save_weights = False

            """
            Fuzzy ARTMAP training ongoing
            #ToDo feature to select train file with screen
            """
            #Get relative train path to package
            rospack = rospkg.RosPack()
            nn_path = rospack.get_path('neural_networks')
            train_file_no = str(train_file)
            art_train_path = nn_path + "/src/neural_networks/train_files/train_art_" + train_file_no + ".csv"

            #Blue light indicates process ongoing
            if sawyer._is_clicksmart == True:
                sawyer.set_blue_light()

            #Display Fuzzy ARTMAP train image
            image_path = package_path + "/share/assembly_images/artmap_train.png"
            headdisplay.display_image(image_path)

            #Train Fuzzy ARTMAP (Give dimensions before adding complement)
            artmap.train(Ia_dim=6, Ib_dim=6, train_path=art_train_path, save_weights=save_weights, load_csv=True)

            #Load trained weights
            artmap.load_weights()

            """
            WARNING
            Move outside robot range
            """
            if sawyer._is_clicksmart == True:
                sawyer.set_green_light()

            #Display warning image
            image_path = package_path + "/share/assembly_images/range.png"
            headdisplay.display_image(image_path)

            sawyer.set_speed(max_linear_speed=0.1, max_linear_accel = 0.1, max_rotational_speed = 0.77, max_rotational_accel = 0.77)

            time.sleep(3.0)

            """
            Home robot
            """
            #Blue light indicates process ongoing
            if sawyer._is_clicksmart == True:
                sawyer.set_blue_light()

            #Move robot to home position
            home_return_code = os.system("rosrun sawyer_sequences sawyer_home.py")

            time.sleep(2.0)

            #Start streaming F/T sensor data
            if not ati_open:
                subprocess.Popen(["rosrun", "devices_interface", "ati_data_publisher.py"])
                ati_open = True

            """
            MESSAGE
            Robot moving above assembly
            """
            #Display approach image
            image_path = package_path + "/share/assembly_images/approach01.png"
            headdisplay.display_image(image_path)

            #Close Gripper
            sawyer.close_gripper()

            #Move above assembly (safe position)
            sawyer.move_to_cartesian_absolute(pos_no=13)
        
            #Wait 1 seconds
            time.sleep(1)

            """
            MESSAGE
            Robot moving near assembly
            """
            #Display approach image
            image_path = package_path + "/share/assembly_images/approach02.png"
            headdisplay.display_image(image_path)

            #Move above assembly (close position)
            #sawyer.move_to_cartesian_absolute(pos_no=14)
            sawyer.move_to_cartesian_absolute(pos_no=16)
            
            #Green light indicates process ongoing
            if sawyer._is_clicksmart == True:
                sawyer.set_green_light()

            #Wait 1 second
            time.sleep(1)

            """
            MESSAGE
            Initial positioning
            Move by hand with zero gravity if desired
            Manual initial positioning must be near a 4 mm error range from hole
            #ToDo add message indicating tolerance for manual position including translation and rotations
            """
            #Display approach image
            image_path = package_path + "/share/assembly_images/startassembly.png"
            headdisplay.display_image(image_path)

            #Initialize button state
            #Ok for current database
            ok_button_state = navigator.get_button_state("right_button_ok")

            #Release button if pressed
            while ok_button_state != 0:
                #Ok for current database
                ok_button_state = navigator.get_button_state("right_button_ok")

                #Display Release button image
                image_path = package_path + "/share/assembly_images/releasebuttons.png"
                headdisplay.display_image(image_path)

            #Button signal debounce
            time.sleep(0.1)

            #Display database selection image
            image_path = package_path + "/share/assembly_images/startassembly.png"
            headdisplay.display_image(image_path)
            
            #Select database to be used
            #Square restarts knowledge base to primitive
            #Wheel center(Ok) uses current knowledge base
            while ok_button_state == 0:
                ok_button_state = navigator.get_button_state("right_button_ok")


            """
            Fuzzy ARTMAP assembly ongoing
            """
            #Display database selection image
            image_path = package_path + "/share/assembly_images/ongoing.png"
            headdisplay.display_image(image_path)
                
            #Start adquiring data, blue signaling force prediction and movements
            
            if sawyer._is_clicksmart == True:
                sawyer.set_blue_light()
            
            #Set speed (linear speed m/s, rotational speed rad/s)	
            sawyer.set_speed(max_linear_speed=0.001, max_linear_accel = 0.001, max_rotational_speed = 0.01, max_rotational_accel = 0.01)

            #Start assembly cycle
            assembly_final_state, execution_time = assembly_cycle(artmap=artmap, train_file=train_file,robot=sawyer, head_display=headdisplay, train_entry=train_entry, image_pub=image_pub, rho=0.95, eps=0.12)
            
            time.sleep(1.0)
            
            """
            Final state Assembly options
            """
            #Green signalizes idle state
            if sawyer._is_clicksmart == True:
                    sawyer.set_green_light()

            #Display assembly final state image
            if assembly_final_state:
                #Display finish success image
                image_path = package_path + "/share/assembly_images/finish_success.png"
                headdisplay.display_image(image_path)
            else:
                #Display finish unsuccess image
                image_path = package_path + "/share/assembly_images/finish_unsuccess.png"
                headdisplay.display_image(image_path)

            #Initialize button state
            #Square to restart assembly with new database
            #Ok to restart with current database
            #Back to go to start page
            square_button_state = navigator.get_button_state("right_button_square")
            ok_button_state = navigator.get_button_state("right_button_ok")
            back_button_state = navigator.get_button_state("right_button_back")

            #Release button if pressed
            while square_button_state != 0 or ok_button_state != 0 or back_button_state != 0:

                square_button_state = navigator.get_button_state("right_button_square")
                ok_button_state = navigator.get_button_state("right_button_ok")
                back_button_state = navigator.get_button_state("right_button_back")

                #Display Release button image
                image_path = package_path + "/share/assembly_images/releasebuttons.png"
                headdisplay.display_image(image_path)

            #Button signal debounce
            time.sleep(0.1)
            
            #Display assembly final state image
            if assembly_final_state:
                #Display finish success image
                image_path = package_path + "/share/assembly_images/finish_success.png"
                headdisplay.display_image(image_path)
            else:
                #Display finish unsuccess image
                image_path = package_path + "/share/assembly_images/finish_unsuccess.png"
                headdisplay.display_image(image_path)

            #Convert values to string
            execution_time_str = str(execution_time) + " s"
            img_success = cv2.imread(image_path)

            # Add assembly text info
            img_success = add_image_text(text_input=execution_time_str,base_image=img_success, position=(620, 520), font_scale=0.85, color=(255, 255, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)

            #Convert to ros image topic
            img_msg = cv_bridge.CvBridge().cv2_to_imgmsg(img_success, encoding="bgr8")

            #Publish in head image display
            image_pub.publish(img_msg)

            #Finish assembly process flag
            finish_process = False
            restart_assembly = False

            while square_button_state == 0 and ok_button_state == 0 and back_button_state == 0:

                square_button_state = navigator.get_button_state("right_button_square")
                ok_button_state = navigator.get_button_state("right_button_ok")
                back_button_state = navigator.get_button_state("right_button_back")

                if square_button_state != 0:
                    save_weights = True
                    restart_assembly = True
                    artmap = ART.FuzzyArtMap()
                elif ok_button_state != 0:
                    save_weights = False
                    restart_assembly = True
                    artmap = ART.FuzzyArtMap()
                elif back_button_state != 0:
                    finish_process = False
                    restart_assembly = False


        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing assembly node")
    
    except rospy.ROSInterruptException:
        
        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing assembly node")
