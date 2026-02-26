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
import sys
from datetime import datetime

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from intera_interface import robot_ctl_ik as robot
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

def move_to_pos(robot,pos_command,step,speed):

        rospy.loginfo("Moving to %s, with a step of %s", pos_command,step)

        if pos_command == "None":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[0.0,0.0,0.0],linear_speed=speed)
        elif pos_command == "X+":
            robot.move_to_cartesian_relative(position=[step,0.0,0.0],orientation=[0.0,0.0,0.0],linear_speed=speed)
        elif pos_command == "X-":
            robot.move_to_cartesian_relative(position=[-step,0.0,0.0],orientation=[0.0,0.0,0.0],linear_speed=speed)
        elif pos_command == "Y+":
            robot.move_to_cartesian_relative(position=[0.0,step,0.0],orientation=[0.0,0.0,0.0],linear_speed=speed)
        elif pos_command == "Y-":
            robot.move_to_cartesian_relative(position=[0.0,-step,0.0],orientation=[0.0,0.0,0.0],linear_speed=speed)
        elif pos_command == "Z+":
            robot.move_to_cartesian_relative(position=[0.0,0.0,step],orientation=[0.0,0.0,0.0],linear_speed=speed)
        elif pos_command == "Z-":
            robot.move_to_cartesian_relative(position=[0.0,0.0,-step],orientation=[0.0,0.0,0.0],linear_speed=speed)
        elif pos_command == "RotX+":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[step+0.1,0.0,0.0],linear_speed=speed)
        elif pos_command == "RotX-":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[-step-0.1,0.0,0.0],linear_speed=speed)
        elif pos_command == "RotY+":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[0.0,step+0.1,0.0],linear_speed=speed)
        elif pos_command == "RotY-":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[0.0,-step-0.1,0.0],linear_speed=speed)
        elif pos_command == "RotZ+":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[0.0,0.0,step+0.1],linear_speed=speed)
        elif pos_command == "RotZ-":
            robot.move_to_cartesian_relative(position=[0.0,0.0,0.0],orientation=[0.0,0.0,-step-0.1],linear_speed=speed)
        elif pos_command == "X+Y+":
            robot.move_to_cartesian_relative(position=[step,step,0.0],orientation=[0.0,0.0,0.0],linear_speed=speed)
        elif pos_command == "X+Y-":
            robot.move_to_cartesian_relative(position=[step,-step,0.0],orientation=[0.0,0.0,0.0],linear_speed=speed)
        elif pos_command == "X-Y+":
            robot.move_to_cartesian_relative(position=[-step,step,0.0],orientation=[0.0,0.0,0.0],linear_speed=speed)
        elif pos_command == "X-Y-":
            robot.move_to_cartesian_relative(position=[-step,-step,0.0],orientation=[0.0,0.0,0.0],linear_speed=speed)

def adquisition_cycle(robot,pos_command,step, test_time=1.0, category=[]):

    global ft_samples
    ft_samples = np.zeros((1,12))
    arg = category

    try:
        sub = rospy.Subscriber("/robot/ft_sensor_topic/", WrenchStamped, obtain_ftdata ,callback_args=(arg[0][0],arg[0][1],arg[0][2],arg[0][3],arg[0][4],arg[0][5]))

        #Define the duration you want the subscriber to run (e.g., 10 seconds)
        duration = rospy.Duration(test_time)
        rospy.loginfo("Ft data adquisition started. It will run for %s seconds after movement finished", duration/1e9)
        
        #Move robot to desired position
        move_to_pos(robot,pos_command,step,speed=0.01)

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
        package_path = rospack.get_path('sawyer_sim_examples')
        csv_file_path = os.path.join(package_path, ATI_net_path)
        np.savetxt(csv_file_path, ft_samples, delimiter=',', fmt='%f')

        rospy.loginfo("Data saved succesfully")

        #Unsubscribe to topic to keep node running
        sub.unregister()
    
    except rospy.ROSInterruptException:

        # Explicitly shutdown the node after the duration
        rospy.loginfo("Test unsuccesful, finishing testing node")
        rospy.signal_shutdown()

def load_gazebo_models(table_pose=Pose(position=Point(x=0.90, y=0.0, z=0.0)),
                       table_reference_frame="world",
                       peg_pose=Pose(position=Point(x=0.6, y=0.0, z=0.9425)),
                       hole_pose=Pose(position=Point(x=0.6, y=0.0, z=0.7725)),
                       object_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('sawyer_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    
    # Load Peg URDF
    peg_xml = ''
    with open (model_path + "peg_hole/peg_0001.urdf", "r") as peg_file:
        peg_xml=peg_file.read().replace('\n', '')
    # Load Hole URDF
    hole_xml = ''
    with open (model_path + "peg_hole/hole_0001.urdf", "r") as hole_file:
        hole_xml=hole_file.read().replace('\n', '')
    
    
    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException as e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("peg_0001", peg_xml, "/",
                               peg_pose, object_reference_frame)
        resp_urdf = spawn_urdf("hole_0001", hole_xml, "/",
                               hole_pose, object_reference_frame)

    except rospy.ServiceException as e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("cafe_table")
        resp_delete = delete_model("hole_0001")
        resp_delete = delete_model("peg_0001")
    except rospy.ServiceException as e:
        print("Delete Model service call failed: {0}".format(e))



if __name__ == '__main__':

    test_movements = [
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

    try:

        #Initialize Example node
        rospy.init_node('sawyer_ftdata_movement', anonymous=False)

        #Load tables
        load_gazebo_models()
        
        #Create sawyer robot instance
        sawyer = robot.SawyerRobot()

        #Wait 2 seconds
        time.sleep(2)

        #Move robot to home
        sawyer.move_to_home()

        #Wait 2 seconds
        time.sleep(1.5)

        #Move to overhead position
        sawyer.move_to_cartesian_absolute(position=[0.6,0.0,0.15],orientation=[-3.13,0.0,-3.12], linear_speed=0.1)
        rospy.sleep(1.0)
        #Move down to position for peg gripping
        sawyer.move_to_cartesian_relative(position=[0.0,0.0,-0.19],orientation=[0.0,0.0,0.0])
        rospy.sleep(1.0)
        #Close gripper
        sawyer.close_gripper()
        
        #Move to position adquire data for movement duration plus test_time
        for movement in test_movements:
            
            direction = movement[0]
            category = movement[1]            
            
            #Creating parallel processes
            adquisition_cycle(robot=sawyer, pos_command=direction, step=0.010, test_time=1.0, category=category)
    
            # both processes finished
            rospy.loginfo("Movement Done to %s, returning to home for new movement", direction)

            #Wait 1 seconds
            time.sleep(1)

            #Back to main position
            sawyer.move_to_cartesian_absolute(position=[0.6,0.0,-0.04],orientation=[-3.13,0.0,-3.12], linear_speed=0.1)

        
        rospy.sleep(1.0)
        #Open gripper
        sawyer.open_gripper()
        #Back to main position
        sawyer.move_to_cartesian_absolute(position=[0.6,0.0,0.15],orientation=[-3.13,0.0,-3.12], linear_speed=0.1)
        rospy.loginfo("Testing sequence finished")
        
        # Explicitly shutdown the node after the duration
        rospy.signal_shutdown("Finishing testing node")

    except rospy.ROSInterruptException:
        rospy.loginfo("Test unsuccesful, finishing testing node")
        rospy.signal_shutdown()