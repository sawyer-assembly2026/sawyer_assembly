
"""
Created on Wed Nov 19 17:23:10 2025

@author: Hector Quijada

Class Sawyer robot that integrates functionality for gripper handling and linear movements in space given a 
cartesian position or angle position


"""

import rospy
import rospkg
import numpy as np
import os
import math
from intera_interface import (
    Gripper,
    SimpleClickSmartGripper,
    get_current_gripper_interface,
    Cuff,
    Limb,
    Lights,
    RobotParams,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from tf.transformations import (
    euler_from_quaternion, 
    quaternion_from_euler,
    quaternion_slerp,
)
from tf import TransformListener

class SawyerRobot():
    
    #Initialize gripper and limb
    """
    #ToDo
    #Check tip_name = "stp_021709TP00448_tip" as we need to move in the current EOAT TCP
    """
    def __init__(self, timeout=None):
        
        try:
            #Obtain limb reference
            self._limb = Limb()
	    self.tf = TransformListener()
            
        except:
            rospy.logerr("Limb reference not found, check connection to robot")
        
        #Max time in seconds to complete motion goal before returning. None is interpreted as an infinite timeout
        self.timeout = timeout
        
        #Get joint names
        self.joint_names = self._limb.joint_names()
        
        #Positions Path
        positions_path="sawyer_pos.csv"
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('sawyer_sim_examples')
        self.positions_path = os.path.join(package_path, positions_path)

        #Positions saved for later movement
        self.positions = self._read_positions()

        #EOAT configuration
        
        #Important Notes:
        #Gripper must be first configured in Intera Studio with robot in manufacturing mode, then pass to Intera SDK mode
        #Mass, CoM, and Endpoint position are very important
        #To connecto to Intera Studio use Robot IP address in a browser while connected to the robot
        #While in Intera SDK mode, check for topic /robot/io_end_effector/config/"here goes gripper ID config file"
        
        try:
            rp = RobotParams()
            valid_limbs = rp.get_limb_names()
            
            #Get current EOAT reference
            self._cuff = Cuff(limb=valid_limbs[0])
            self._gripper = get_current_gripper_interface()
            
            #Flag for ClickSmartGripper 
            self._is_clicksmart = isinstance(self._gripper, SimpleClickSmartGripper)
            
            #If ClickSmartGripper initialize if needed
            if self._is_clicksmart:
                self._lights = Lights()
                if self._gripper.needs_init():
                    self._gripper.initialize()
                msg = "Smart Click Connection found and initialized, ready for use"
                
                #Set tip name to smart adapter connection
                self.tip_name = "stp_021709TP00448_tip"
                rospy.logwarn(msg)
            else:
                #If electric gripper not calibrated, raise exception, and calibrate, then create instance of SawyerRobot
                if not (self._gripper.is_calibrated() or self._gripper.calibrate() == True):
                    raise
        except:
            
            try:
                self._gripper = Gripper()
                self._is_clicksmart = False
                self.tip_name = "right_gripper_tip"
                msg = "Smart Click Connection not found, connecting to class Gripper(), for electric gripper"
                rospy.logwarn(msg)
            except:
                self._gripper = None
                self._is_clicksmart = False
                self.tip_name = "right_hand"
                msg = "No gripper connection found, check EOAT if gripper connected but not detected"
                rospy.logwarn(msg)
            
    #Positions for movement convenience (CARTESIAN):
    def _read_positions(self):
        
        if os.path.exists(self.positions_path):
        
            pos = np.zeros(6)
            pos_aux = np.zeros(6)
            
            #Extract all Inputs from train file as a single numpy array for ARTa and ARTb
            with open(self.positions_path, 'r') as file:
                
                # Read all lines and process each one
                lines = file.readlines()
                for line in lines:
                    pos_str = line.strip().split(',') #Extract EOL and split in a list with separator ","
                    
                    #Split into two vectors and convert to floar
                    for i in range(len(pos_str)):
                        pos_aux[i] = float(pos_str[i])
                    
                    #Stack current line vector into a matrix
                    pos = np.vstack((pos, pos_aux))
                    #Delete first row as it was auxiliary
                
                pos = np.delete(pos, 0, axis=0)
                
        else:
            msg = "No file named sawyer_pos.csv found, create file with new positions"
            rospy.logwarn(msg)
            pos = None
                
        return pos
    
    #Save Position in a current
    def save_position(self):
        
        position, euler_orientation = self.current_endpoint_pose()
        
        save_pose = np.zeros((1,6))
        
        save_pose[0][0] = position[0]
        save_pose[0][1] = position[1]
        save_pose[0][2] = position[2]
        save_pose[0][3] = euler_orientation[0]
        save_pose[0][4] = euler_orientation[1]
        save_pose[0][5] = euler_orientation[2]
        
        if len(self.positions) > 0:
            self.positions = np.vstack((self.positions, save_pose))
        else:
            self.positions = save_pose
        
        np.savetxt(self.positions_path, self.positions, delimiter=',', fmt='%f')
        rospy.loginfo('New position saved')
        

    #Give current endpoint pose with orientation in euler angles
    def current_endpoint_pose(self, quaternion=False):
        
        """
        #To Do
        Test endpoint pose corresponds to current gripper setup done in Intera Studio
        else use endpoint_state = limb.tip_state(self.tip_name) which returns PoseStamped msg
        """
        # #Returns a dictionary
        # #pose = {'position': (x, y, z), 'orientation': (x, y, z, w)}
        # endpoint_pose = self._limb.endpoint_pose()
        
        # #(x,y,z) Cartesian Position 
        # endpoint_position = endpoint_pose['position']

        # endpoint_position = list(endpoint_position)
        
        # #(x,y,z,w) Cuaternion Orientation 
        # endpoint_orientation_quaternion = endpoint_pose['orientation']
        
        #Obtain pose through transforms
        if self.tf.frameExists(self.tip_name) and self.tf.frameExists("base"):
            t = self.tf.getLatestCommonTime("base",self.tip_name)
            endpoint_position, endpoint_orientation_quaternion = self.tf.lookupTransform("base",self.tip_name, t)

        #Transform to roll pitch yaw for movement
        roll, pitch, yaw = euler_from_quaternion(endpoint_orientation_quaternion)
        
        endpoint_orientation_euler = [roll, pitch, yaw]
        
        if not quaternion:
            #Return as list so values can be edited with euler angles
            return endpoint_position, endpoint_orientation_euler
        else:
            #Return as list so values can be edited with quaternions orientation
            return endpoint_position, endpoint_orientation_quaternion

    #Set green color light
    def set_green_light(self):
        self._set_light("green",True)
        self._set_light("red",False)
        self._set_light("blue",False)
    
    #Set red color light
    def set_red_light(self):
        self._set_light("green",False)
        self._set_light("red",True)
        self._set_light("blue",False)
    
    #Set blue color light
    def set_blue_light(self):
        self._set_light("green",False)
        self._set_light("red",False)
        self._set_light("blue",True)
    
    #Set Light
    def _set_light(self,color,value):
	rp = RobotParams()
	valid_limbs = rp.get_limb_names()
       	self._lights.set_light_state('head_{0}_light'.format(color), on=bool(value))
        self._lights.set_light_state('{0}_hand_{1}_light'.format(valid_limbs[0], color), on=bool(value))
            
    #Open EOAT Gripper
    def open_gripper(self):
        if self._gripper.is_ready() != None:
            if self._gripper.is_ready():
                rospy.loginfo("Gripper open triggered")
                if self._is_clicksmart:
                	self._gripper.set_ee_signal_value('grip', False)
                else:
                	self._gripper.open()
        else:
            rospy.loginfo("Gripper not detected")
    
    #Close EOAT Gripper
    def close_gripper(self):
    if self._gripper.is_ready() != None:
		if self._gripper.is_ready():
		    rospy.loginfo("Gripper close triggered")
		    if self._is_clicksmart:
		        self._gripper.set_ee_signal_value('grip', True)
		    else:
		        self._gripper.close()
	else:
		rospy.loginfo("Gripper not detected")
    
    #Move to desired joint positions in radians, with a timeout (not a linear interpolation)
    #Speed ratio must be within [0.0,1.0]
    def move_to_joint_position(self, joint_angles, speed_ratio, timeout=15.0):
        
        self.timeout = timeout
        
        if rospy.is_shutdown():
            return
        if joint_angles:
            self._limb.set_joint_position_speed(speed=speed_ratio)
            self._limb.move_to_joint_positions(joint_angles,timeout=self.timeout)
            rospy.loginfo("Joint movement finished")  
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def move_to_home(self, speed = 0.1):
        
        #Set home joint angles
        home_joint_angles = {'right_j0': -0.041662954890248294,
                                 'right_j1': -1.3758291091425074,
                                 'right_j2': 0.0293680414401436,
                                 'right_j3': 1.825181629,
                                 'right_j4':  -0.06703022873354225,
                                 'right_j5': 1.096837143,
                                 'right_j6': 1.7659649178699421}
        
        rospy.loginfo("Moving arm to home position...")

        self.move_to_joint_position(home_joint_angles,speed)
        
        rospy.loginfo("Moved to home angles: %s", home_joint_angles)  
    
    def move_to_home_cartesian(self):
        
        rospy.loginfo("Moving arm to home position with cartesian movement...")

        self.move_to_cartesian_absolute(position=[0.529, 0.130, 0.191], orientation=[-3.085, 0.021, 3.139],linear_speed=0.05)
        
        rospy.loginfo("Moved to home angles: %s", self._limb.joint_angles)  

    #Final pose as PoseStamped(), linear_speed in m/s
    #Perform linear movements only, not considering changes in rotation
    def cartesian_approach(self, move_position, joint_speed=0.1 ,linear_speed=0.1, frecuency=100, verbose=True):
       
	final_pose = PoseStamped()
	
        #Set desired speed (m/s)
	self._limb.set_joint_position_speed(joint_speed)
	
        #Get current endpoint pose
        current_position, current_orientation = self.current_endpoint_pose(quaternion=True)
        
        #Obtain final position from relative movement
        final_pose.pose.position.x = current_position[0] + move_position[0]
        final_pose.pose.position.y = current_position[1] + move_position[1]
        final_pose.pose.position.z = current_position[2] + move_position[2]

        #Distance to be traveled by end effector in a linear fashion
        delta = Point()
        
        #Compute steps and final time based on desired linear speed
        delta.x = abs(move_position[0])
        delta.y = abs(move_position[1])
        delta.z = abs(move_position[2])
        
        #Obtain total traveled distance by end effector
        c = math.sqrt((delta.x * delta.x) + (delta.y * delta.y))
        distance_EF = math.sqrt((c*c) + (delta.z * delta.z))
    
        #Compute final time(s) based on total distance(m) to be traveled and desired speed (m/s)
        final_time = distance_EF/linear_speed
        
        #Movement frecuency (Hz)
        #Corresponds to times the command set_joint_positions is going to be send in a second
        frec_hz = frecuency

        #Ik solver frecuency steps 
        steps = final_time * frec_hz
    
        r = rospy.Rate(frec_hz) # Defaults to 100Hz command rate
        
        #Linear interpolation
        
        #Step size for interpolation based on distance to be traveled and desired steps
        ik_delta = Point()
        ik_delta.x = (current_position[0] - final_pose.pose.position.x) / steps
        ik_delta.y = (current_position[1] - final_pose.pose.position.y) / steps
        ik_delta.z = (current_position[2] - final_pose.pose.position.z) / steps

        for d in range(int(steps), -1, -1):
            if rospy.is_shutdown():
                return
            ik_step = Pose()
            ik_step.position.x = d*ik_delta.x + final_pose.pose.position.x 
            ik_step.position.y = d*ik_delta.y + final_pose.pose.position.y
            ik_step.position.z = d*ik_delta.z + final_pose.pose.position.z
            # Only linear movement considered
            ik_step.orientation.x = current_orientation[0]
            ik_step.orientation.y = current_orientation[1]
            ik_step.orientation.z = current_orientation[2]
            ik_step.orientation.w = current_orientation[3]

            #Request inverse kinematics at position ik_step with tip desired depending on EAOT
            joint_angles = self._limb.ik_request(ik_step, self.tip_name)

            #Send position
            if joint_angles:
                    self._limb.set_joint_positions(joint_angles)
            else:
                rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")
            r.sleep()
        
        if verbose:
            #Movement finished confirmation
            rospy.loginfo("Cartestian movement finished")
            rospy.loginfo("Final Pose: %s", final_pose)  
      
    #Final pose as PoseStamped()
    def rotation_interpolation(self, move_orientation, final_pose, joint_speed=0.1, final_time=5, frecuency=100, verbose=True):

	final_pose = PoseStamped()        

	self._limb.set_joint_position_speed(joint_speed)
	
        #Get current endpoint pose
        current_position, current_orientation = self.current_endpoint_pose(quaternion=False)

        #Add orientation movement to current orientation in euler angles and transform to quaternion      
	
        move_euler = [0,0,0]
        move_euler[0] = current_orientation[0] + move_orientation[0]
        move_euler[1] = current_orientation[1] + move_orientation[1] 
        move_euler[2] = current_orientation[2] + move_orientation[2] 
        
        roll = move_euler[0]
        pitch = move_euler[1]
        yaw = move_euler[2]

        move_quaternion = quaternion_from_euler(roll, pitch, yaw)
        
	final_pose.pose.orientation.x = float(move_quaternion[0])
        final_pose.pose.orientation.y = float(move_quaternion[1])
        final_pose.pose.orientation.z = float(move_quaternion[2])
        final_pose.pose.orientation.w = float(move_quaternion[3])
       
        #Obtain frecuency base on final time and steps 
        steps = final_time * frecuency
        r = rospy.Rate(frecuency)
           
	#Get current endpoint pose quaternion
        current_position, current_orientation_qtn = self.current_endpoint_pose(quaternion=True)	
	
        q_current = [current_orientation_qtn[0], 
                     current_orientation_qtn[1],
                     current_orientation_qtn[2],
                     current_orientation_qtn[3]]
        q_pose = [final_pose.pose.orientation.x,
                  final_pose.pose.orientation.y,
                  final_pose.pose.orientation.z,
                  final_pose.pose.orientation.w]
        
        for d in range(int(steps), -1, -1):
            if rospy.is_shutdown():
                return
            ik_step = Pose()
            ik_step.position.x = current_position[0] 
            ik_step.position.y = current_position[1]
            ik_step.position.z = current_position[2]
            # Perform a proper quaternion interpolation
            q_slerp = quaternion_slerp(q_current, q_pose, ((steps-d)/steps))
            ik_step.orientation.x = q_slerp[0]
            ik_step.orientation.y = q_slerp[1]
            ik_step.orientation.z = q_slerp[2]
            ik_step.orientation.w = q_slerp[3]
            joint_angles = self._limb.ik_request(ik_step, self.tip_name)
            if joint_angles:
                    self._limb.set_joint_positions(joint_angles)
            else:
                rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")
            r.sleep()
        
        if verbose:
            #Movement finished confirmation
            rospy.loginfo("Cartestian movement finished")
            rospy.loginfo("Final Pose: %s", final_pose)  
       
