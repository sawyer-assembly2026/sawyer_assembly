#!/usr/bin/env python

"""
Created on Wed Nov 19 17:23:10 2025

@author: Hector Quijada

Example moves sawyer robot to predetermined home position and starts adquiring FT sensor data
End effector can be moved by hand

"""

import rospy
import rospkg
import cv2
import cv_bridge
import math
from sensor_msgs.msg import Image
import time
import numpy as np
import os
import sys
from datetime import datetime
from devices_interface import robot_ctl as robot
from intera_interface import (
    Navigator,
    HeadDisplay,
    Head
)
from neural_networks import ART
from devices_interface import ATI_Net as ati
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    WrenchStamped
)
from tf.transformations import (
    euler_from_quaternion, 
    quaternion_from_euler,
    quaternion_slerp,
)

from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)

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
    
def prediction_adquisition_cycle(artmap, robot, navigator):
    
    global ati_ft_data

    #Desired rotation step 1deg = 0.0175 approx.
    rot_step = 0.0175*0.03
    #Desired translation step 0.001 m = 1mm
    tran_step = 0.001

    pred = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-tran_step, 0.0, 0.0, 0.0, 0.0, 0.0],
            [tran_step, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -tran_step, 0.0, 0.0, 0.0, 0.0],
            [0.0, tran_step, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, tran_step, 0.0, 0.0, 0.0],
            [0.0, 0.0, -tran_step, 0.0, 0.0, 0.0],
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
    
    move_categories = [
                    ("None", np.array([[1., 0., 0., 0., 0., 0.]])), 
                    ("X-", np.array([[1., 0., 0., 0., 0., 1.]])), 
                    ("X+", np.array([[1., 0., 0., 0., 1., 0.]])),
                    ("Y-", np.array([[1., 0., 0., 0., 1., 1.]])),
                    ("Y+", np.array([[1., 0., 0., 1., 0., 0.]])),
                    ("Z-", np.array([[1., 0., 0., 1., 0., 1.]])),
                    ("Z+", np.array([[1., 0., 0., 1., 1., 0.]])),
                    ("RotX-", np.array([[1., 0., 0., 1., 1., 1.]])),
                    ("RotX+", np.array([[1., 0., 1., 0., 0., 0.]])),
                    ("RotY-", np.array([[1., 0., 1., 0., 0., 1.]])),
                    ("RotY+", np.array([[1., 0., 1., 0., 1., 0.]])),
                    ("RotZ-", np.array([[1., 0., 1., 0., 1., 1.]])),
                    ("RotZ+", np.array([[1., 0., 1., 1., 0., 0.]])),
                    ("X-Y-", np.array([[1., 0., 1., 1., 0., 1.]])),
                    ("X-Y+", np.array([[1., 0., 1., 1., 1., 0.]])),
                    ("X+Y-", np.array([[1., 0., 1., 1., 1., 1.]])),
                    ("X+Y+", np.array([[1., 1., 0., 0., 0., 0.]]))
                    ]

    try:
        #Create Head Display instance
        headdisplay = HeadDisplay()

        #Get current package path
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('tactile')

        #force input in image font
        position1 = (75, 100)
        font1 = cv2.FONT_HERSHEY_SIMPLEX
        font_scale1 = 0.72
        color1 = (0, 0, 255)
        thickness1 = 2

        #Current class selected in image font
        position2 = (675, 420)
        font2 = cv2.FONT_HERSHEY_SIMPLEX
        font_scale2 = 8.0
        color2 = (250, 225, 100)
        thickness2 = 5

        #Read image to add text
        image_path = package_path + "/share/retrain_home.png"
        img = cv2.imread(image_path)

        #Current prediction in image font
        position3 = (280, 460)
        font3 = cv2.FONT_HERSHEY_SIMPLEX
        font_scale3 = 6.0
        color3 = (250, 225, 100)
        thickness3 = 5

        #Read prediction image to add text
        image_pred_path = package_path + "/share/prediction.png"
        img_pred = cv2.imread(image_pred_path)

        rospy.Subscriber("/robot/ati_ft_sensor_topic/", WrenchStamped, obtain_ftdata, queue_size=1)
        image_pub = rospy.Publisher('/robot/head_display', Image, latch=True, queue_size=10)

        #Set desired speed (m/s)
        robot._limb.set_joint_position_speed(0.01)
        ik_step = Pose()
        start_time = time.clock()
        move_euler = [0,0,0]
        i = 0
        r = rospy.Rate(100)

        #while not rospy.is_shutdown():

        for i in range(5000):

            if rospy.is_shutdown():
                return

            complement_encode_ati_ft_data = np.concatenate((ati_ft_data[0], 1-ati_ft_data[0]))
           
            j = artmap.predict(complement_encode_ati_ft_data,rho_a=0.9)

            k = j[0]
            i+=1

            if k != None:
                
                # #Get current endpoint pose
                current_position, current_orientation = robot.current_endpoint_pose(quaternion=True)

                #ik_step.position.x = current_position[0] + pred[k][1]
                #ik_step.position.x = current_position[0]
                #ik_step.position.y = current_position[1] - pred[k][0]
                #ik_step.position.y = current_position[1]
                current_position[2] = current_position[2] - 0.005

                #move_euler[0] = current_orientation[0] + pred[k][3]
                #move_euler[1] = current_orientation[1] + pred[k][4]
                #move_euler[2] = current_orientation[2] + pred[k][5]

                #roll = move_euler[0]
                #pitch = move_euler[1]
                #yaw = move_euler[2]

                #move_quaternion = quaternion_from_euler(roll, pitch, yaw)

                #ik_step.orientation.x = float(move_quaternion[0])
                #ik_step.orientation.y = float(move_quaternion[1])
                #ik_step.orientation.z = float(move_quaternion[2])
                #ik_step.orientation.w = float(move_quaternion[3])
                
                #ik_step.orientation.x = current_orientation[0]
                #ik_step.orientation.y = current_orientation[1]
                #ik_step.orientation.z = current_orientation[2]
                #ik_step.orientation.w = current_orientation[3]

                #Request inverse kinematics at position ik_step with tip desired depending on EAOT
                joint_angles = robot.ik_srv(current_position, current_orientation)

                #Send position
                if joint_angles:
                    robot._limb.set_joint_positions(joint_angles)
                else:
                    rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

                #img_pred_iterate = img_pred.copy()
                #img_pred_class = cv2.putText(img_pred_iterate, move_categories[k][0], position3, font3, font_scale3, color3, thickness3, cv2.LINE_AA)
                #Convert to ros image topic
                #img_msg = cv_bridge.CvBridge().cv2_to_imgmsg(img_pred_class, encoding="bgr8")

                #Publish in head image display
                #image_pub.publish(img_msg)
                #robot.move_to_cartesian_relative(position=[pred[k][0],pred[k][1],pred[k][2]],orientation=[pred[k][3],pred[k][4],pred[k][5]], move_confirm=True, verbose=True)
                #time.sleep(1.0)
            else:
                #Remove weights destined to retrain
                artmap.remove_prediction_weight()

                # if robot._is_clicksmart == True:
                #     robot.set_red_light()

                # # Add force input text to the image
                # force_input = str(j[1][:6])
                # img_iterate = img.copy()
                # img_force = cv2.putText(img_iterate, force_input, position1, font1, font_scale1, color1, thickness1, cv2.LINE_AA)

                # #Initializa Navigator 
                # inital_state = navigator.get_wheel_state("right_wheel")
                # square_button_state = navigator.get_button_state("right_button_square")
                # ok_button_state = navigator.get_button_state("right_button_ok")

                # while square_button_state == 0 and ok_button_state == 0:

                #     #Update button states
                #     current_state = navigator.get_wheel_state("right_wheel")
                #     square_button_state = navigator.get_button_state("right_button_square")
                #     ok_button_state = navigator.get_button_state("right_button_ok")

                #     #rospy.loginfo("Current input not recognized by Fuzzy ARTMAP: %s", j[1])
                #     #rospy.loginfo("Category choices:\n None (0)\n X+ (1)\n X- (2)\n Y+ (3)\n Y- (4)\n Z+ (5)\n Z- (6)\n RotX+ (7)\n RotX- (8)\n RotY+ (9)\n RotY- (10)\n RotZ+ (11)\n RotZ- (12)\n X+Y+ (13)\n X+Y- (14)\n X-Y+ (15)\n X-Y- (16)\n")
                #     category_input = abs(inital_state - current_state)

                #     #Update inital state in case outside limits
                #     if category_input > 16:
                #          inital_state = navigator.get_wheel_state("right_wheel")
                    
                #     category_input_str = str(category_input)

                #     # Add force input text to the image
                #     img_force_iterate = img_force.copy()
                #     img_force_class = cv2.putText(img_force_iterate, category_input_str, position2, font2, font_scale2, color2, thickness2, cv2.LINE_AA)

                #     #Convert to ros image topic
                #     img_msg = cv_bridge.CvBridge().cv2_to_imgmsg(img_force_class, encoding="bgr8")

                #     #Publish in head image display
                #     image_pub.publish(img_msg)

                #     #category_input = raw_input('Input Category for retraining or type "cancel" to avoid retraining: ')
                    
                #     if ok_button_state != 0:
                #          category_input = "cancel"

                # if category_input is not "cancel":
                        
                #     #Train Choice
                #     tc = int(category_input)

                #     category = categories[tc][0]
                        
                #     for movement in categories:
                #         direction = movement[0]
                #         if direction == category:
                #             Ia = j[1][:6].reshape(1,6)
                #             Ib = movement[1]
                                
                #             #Retrain Fuzzy ARTMAP (Give dimensions before adding complement)
                #             artmap.train(Ia_dim=6, Ib_dim=6, train_path="/home/kid/ros_ws/src/sawyer_simulator/sawyer_sim_examples/train_3.csv", save_weights=True, load_csv=False,  Ia_retrain = Ia, Ib_retrain = Ib)

                #             #Load trained weights
                #             artmap.load_weights()

                #             if robot._is_clicksmart == True:
                #                 robot.set_blue_light()

                #         #Display Intera SDK home image
                #         image_path = package_path + "/share/sawyer_sdk_research.png"
                #         headdisplay.display_image(image_path)
                    
                # else: 
                #         rospy.loginfo("Retrain canceled, resuming prediction sequence...")
                        
                #         #Remove weights destined to retrain
                #         artmap.remove_prediction_weight()

                #         if robot._is_clicksmart == True:
                #             robot.set_blue_light()

                #         #Display Intera SDK home image
                #         image_path = package_path + "/share/sawyer_sdk_research.png"
                #         headdisplay.display_image(image_path)

            r.sleep()
            
        
        else:
        
            end_time = time.clock()
            execution_time = end_time - start_time

            frecuency = (i+1)/execution_time
            print("Iterations", i+1)
            print("execution time", execution_time)
            print("loop frecuency", frecuency)
            sys.exit()

    except rospy.ROSInterruptException:
        rospy.signal_shutdown("Finishing testing node")


if __name__ == '__main__':
    
    try:
        #Initialize Example node
        rospy.init_node('sawyer_ftdata_predict', anonymous=False)

        ####INTIALIZE ROBOT INSTANCES####

        #Create sawyer robot instance
        sawyer = robot.SawyerRobot()

        #Create Navigator instance
        navigator = Navigator()

        #Create Head Display instance
        headdisplay = HeadDisplay()
        
        #Create Head movements instance
        head = Head()

        ####TRAIN FUZZY ARTMAP AND LOAD WEIGHTS####

        #Create FuzzyARTMAP Neural Network Instance
        artmap = ART.FuzzyArtMap()

        #Train Fuzzy ARTMAP (Give dimensions before adding complement)
        artmap.train(Ia_dim=6, Ib_dim=6, train_path="/home/hector/ros_ws/src/sawyer_assembly/neural_networks/src/neural_networks/train_files/train_art_3.csv", save_weights=True, load_csv=True)

        #Load trained weights
        artmap.load_weights()

        #Wait 2 seconds
        time.sleep(2)
        
        if sawyer._is_clicksmart == True:
            sawyer.set_red_light()
	
	    #Set speed (linear speed m/s, rotational speed rad/s)
        sawyer.set_speed(max_linear_speed = 0.1, max_linear_accel = 0.1, max_rotational_speed = 0.77, max_rotational_accel = 0.77)
	
        #Wait 2 seconds
        time.sleep(2)
            
        #Start adquiring data, blue signaling force prediction and movements
        
        if sawyer._is_clicksmart == True:
            sawyer.set_blue_light()
            
        prediction_adquisition_cycle(artmap=artmap, robot=sawyer, navigator=navigator)
            
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
