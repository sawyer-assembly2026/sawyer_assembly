#!/usr/bin/env python

# Copyright (c) 2015-2018, Rethink Robotics Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sawyer SDK Inverse Kinematics Pick and Place Demo
"""
import argparse
import struct
import sys
import copy

import rospy
import rospkg

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
from tf.transformations import quaternion_slerp
import intera_interface
from intera_interface import robot_ctl_ik as robot

def load_gazebo_models(table_pose=Pose(position=Point(x=0.90, y=0.0, z=0.0)),
                       table_reference_frame="world",
                       peg_pose=Pose(position=Point(x=0.6, y=0.1265, z=0.9425)),
                       hole_pose=Pose(position=Point(x=0.6, y=0.00, z=0.7725)),
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

def main():
    """SDK Inverse Kinematics Pick and Place Example

    A Pick and Place example using the Rethink Inverse Kinematics
    Service which returns the joint angles a requested Cartesian Pose.
    This ROS Service client is used to request both pick and place
    poses in the /base frame of the robot.

    Note: This is a highly scripted and tuned demo. The object location
    is "known" and movement is done completely open loop. It is expected
    behavior that Sawyer will eventually mis-pick or drop the block. You
    can improve on this demo by adding perception and feedback to close
    the loop.
    """
    rospy.init_node("ik_pick_and_place_demo")
    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    load_gazebo_models()
    # Remove models from the scene on shutdown
    #rospy.on_shutdown(delete_gazebo_models)

    limb = 'right'
    hover_distance = 0.15 # meters
    # Starting Joint angles for right arm
    starting_joint_angles = {'right_j0': -0.041662954890248294,
                             'right_j1': -1.0258291091425074,
                             'right_j2': 0.0293680414401436,
                             'right_j3': 2.17518162913313,
                             'right_j4':  -0.06703022873354225,
                             'right_j5': 0.3668371433926965,
                             'right_j6': 1.7659649178699421}
    
    # An orientation for gripper fingers to be overhead and parallel to the obj
    overhead_orientation = Quaternion(
                             x=0.0,
                             y=-1.0,
                             z=0.0,
                             w=0.0)
    block_poses = list()
    # The Pose of the block in its initial location.
    # You may wish to replace these poses with estimates
    # from a perception node.
    block_poses.append(Pose(
        position=Point(x=0.6, y=0.1265, z=-0.029),
        orientation=overhead_orientation))
    # Feel free to add additional desired poses for the object.
    # Each additional pose will get its own pick and place.
    block_poses.append(Pose(
        position=Point(x=0.6, y=0.0, z=-0.060),
        orientation=overhead_orientation))
    rospy.sleep(1.0)
    # Move to the desired starting angles
    sawyer = robot.SawyerRobot()

    sawyer.move_to_home()
    rospy.sleep(2.0)
    sawyer.move_to_cartesian_absolute(position=[0.6,0.1265,0.15],orientation=[-3.13,0.0,-3.13], linear_speed=0.1)
    rospy.sleep(1.0)
    sawyer.move_to_cartesian_relative(position=[0.0,0.0,-0.17],orientation=[0.0,0.0,0.0])
    rospy.sleep(1.0)
    sawyer.close_gripper()
    rospy.sleep(1.0)
    sawyer.move_to_cartesian_relative(position=[0.0,0.0,0.17],orientation=[0.0,0.0,0.0])
    rospy.sleep(1.0)
    sawyer.move_to_cartesian_relative(position=[0.0,-0.1265,0.0],orientation=[0.0,0.0,0.0])
    rospy.sleep(1.0)
    sawyer.move_to_cartesian_relative(position=[0.0,0.0,-0.17],orientation=[0.0,0.0,0.0], linear_speed=0.01)
    rospy.sleep(1.0)
    sawyer.open_gripper()
    sawyer.move_to_cartesian_relative(position=[0.0,0.0,0.17],orientation=[0.0,0.0,0.0])


    return 0

if __name__ == '__main__':
    sys.exit(main())
