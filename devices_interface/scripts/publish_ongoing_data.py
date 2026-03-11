#!/usr/bin/env python

import rospy
import numpy as np
import rospkg
import cv2
import cv_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import (
    Float32,
    Int32
)
from geometry_msgs.msg import WrenchStamped


#Global variables for force and torque
#Need to be global to be updated by call back function in subscriber
ongoing_data = [0.0,0.0,0.0,0.0,0]
ati_ft_data_raw = np.full((1, 6), 0.5)

#Callback function for subscriber
def obtain_ftdata_raw(data):
    
    #Split wrench data and normalize
    ati_ft_data_raw[0][0] = data.wrench.force.x
    ati_ft_data_raw[0][1] = data.wrench.force.y
    ati_ft_data_raw[0][2] = data.wrench.force.z
    ati_ft_data_raw[0][3] = data.wrench.torque.x
    ati_ft_data_raw[0][4] = data.wrench.torque.y
    ati_ft_data_raw[0][5] = data.wrench.torque.z
    
def obtain_zd(data):
    ongoing_data[0] = data.data

def obtain_epsilon(data):
    ongoing_data[1] = data.data

def obtain_zcurrent(data):
    ongoing_data[2] = data.data

def obtain_Jcurrent(data):
    ongoing_data[3] = data.data

def obtain_prediction(data):
    ongoing_data[4] = data.data

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

def add_image_text(text_input,base_image, position, font_scale, color, thickness, font=cv2.FONT_HERSHEY_SIMPLEX):

    # Add text image to topic
    img_iterate = base_image.copy()
    return cv2.putText(img_iterate, text_input, position, font, font_scale, color, thickness, cv2.LINE_AA)

if __name__ == '__main__':
    
    try:
        
        #Initialize ROS node
        rospy.init_node('assembly_data_headdisplay', anonymous=False)
        
        #Get tactile package path
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('tactile')

        #Read assembly base image to add text
        onoing_image_path = package_path + "/share/assembly_images/ongoing.png"
        ongoing_img = cv2.imread(onoing_image_path)

        #Set FPS rate
        r = rospy.Rate(20)

        #Subscribe to force torque sensor topic
        rospy.Subscriber('/robot/artmap_assembly_zd/', Float32, obtain_zd, queue_size=1)
        rospy.Subscriber('/robot/artmap_assembly_epsilon/', Float32, obtain_epsilon, queue_size=1)
        rospy.Subscriber('/robot/artmap_assembly_zcurrent/', Float32, obtain_zcurrent, queue_size=1)
        rospy.Subscriber('/robot/artmap_assembly_Jcurrent/', Float32, obtain_Jcurrent, queue_size=1)
        rospy.Subscriber('/robot/artmap_assembly_prediction/', Int32, obtain_prediction, queue_size=1)
        rospy.Subscriber('/robot/ati_ft_sensor_topic/', WrenchStamped, obtain_ftdata_raw, queue_size=1)

        #Create image publisher for head display
        image_pub = rospy.Publisher('/robot/head_display', Image, latch=True, queue_size=10)

        #run while rospy is not shutdown
        while not rospy.is_shutdown():
            
            #Convert values to strings
            f = ongoing_data[0]*1000
            zd_str = '%.6f' % f
            zd_str = zd_str + " mm"

            f = ongoing_data[1]
            epsilon_str = '%.6f' % f
            
            f = ongoing_data[2]*1000
            zcurrent_str = '%.6f' % f
            zcurrent_str = zcurrent_str + " mm"

            f = ongoing_data[3]
            Jcurrent_str = '%.6f' % f

            prediction = ongoing_data[4]

            if prediction == 100:
                prediction_str = "Z-"
            else:
                prediction_str = obtain_move_categories(prediction)

            force_str = str(ati_ft_data_raw)

            #Add assembly text info
            img_z_value = add_image_text(text_input=zcurrent_str,base_image=ongoing_img, position=(430, 326), font_scale=0.85, color=(0, 255, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
            img_J_value = add_image_text(text_input=Jcurrent_str,base_image=img_z_value, position=(430, 379), font_scale=0.85, color=(0, 255, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
            img_next_prediction = add_image_text(text_input=prediction_str,base_image=img_J_value, position=(430, 432), font_scale=0.85, color=(0, 255, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
            img_force = add_image_text(text_input=force_str,base_image=img_next_prediction, position=(75, 275), font_scale=0.80, color=(0, 255, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)

            img_eps = add_image_text(text_input=epsilon_str,base_image=img_force, position=(306, 181), font_scale=0.85, color=(0, 255, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)
            img_z_target = add_image_text(text_input=zd_str,base_image=img_eps, position=(767, 181), font_scale=0.85, color=(0, 255, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)

            #Convert to ros image topic
            img_msg = cv_bridge.CvBridge().cv2_to_imgmsg(img_z_target, encoding="bgr8")

            #Publish in head image display
            image_pub.publish(img_msg)

            r.sleep()


    except rospy.ROSInterruptException:
        pass

