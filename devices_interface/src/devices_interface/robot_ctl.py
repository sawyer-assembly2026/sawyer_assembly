
"""
Created on Wed Nov 19 17:23:10 2025

@author: Hector Quijada

Class Sawyer robot that integrates functionality for gripper handling and linear movements in space given a 
cartesian position or angle position


"""

import rospy
import numpy as np
import os
import errno
import math
import time
from intera_interface import (
    Gripper,
    SimpleClickSmartGripper,
    get_current_gripper_interface,
    Cuff,
    Limb,
    Lights,
    RobotParams,
)
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions,
)
from intera_motion_msgs.msg import TrajectoryOptions
from intera_motion_msgs.msg import TrackingOptions
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from sensor_msgs.msg import ( 
    JointState 
 ) 

from tf.transformations import (
    euler_from_quaternion, 
    quaternion_from_euler,
    quaternion_slerp,
)
from tf import TransformListener

from std_msgs.msg import Header

from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
    SolvePositionIK,
    SolvePositionIKRequest,
)


class SawyerRobot():
    
    #Initialize gripper and limb
    def __init__(self, max_linear_speed = 0.1, max_linear_accel = 0.1, max_rotational_speed = 0.77, max_rotational_accel = 0.77, timeout=None):
        
        try:
            #Obtain limb reference
            self._limb = Limb()
            self.tf = TransformListener()

            self.joint_states = rospy.Subscriber('robot/joint_states', JointState, self._on_joint_states, queue_size=1, tcp_nodelay=True)
            self.joint_names = []
            self.joint_angles = []

        except:
            rospy.logerr("Limb reference not found, check connection to robot")

        #Set movement parameters
        self.max_linear_speed = max_linear_speed
        self.max_linear_accel = max_linear_accel
        self.max_rotational_speed = max_rotational_speed
        self.max_rotational_accel = max_rotational_accel
        
        #Obtain positions path relative to current document location to save all positions
        self._positions_path = self._pos_path()
        
        #Positions saved for later movement
        self.positions = self._read_positions()
        
	    #Tracking Options
        self._track_options = TrackingOptions()
        self._track_options.use_min_time_rate = True
        self._track_options.min_time_rate = 0.1
        self._track_options.use_max_time_rate = True
        self._track_options.max_time_rate = 1.0
        self._track_options.goal_joint_tolerance = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]
		
        #Trajectory configuration
        self._traj_options = TrajectoryOptions()
        self._traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
        self._traj_options.nso_start_offset_allowed = False
        self._traj_options.nso_check_end_offset = False
        self._traj_options.tracking_options = self._track_options
        self._traj_options.path_interpolation_step = 0.0        
        self.traj = MotionTrajectory(trajectory_options = self._traj_options, limb = self._limb)
        
        #Max time in seconds to complete motion goal before returning. None is interpreted as an infinite timeout
        self.timeout = timeout

        self.wpt_opts = MotionWaypointOptions(joint_tolerances = 0.01, max_linear_speed = self.max_linear_speed,
                                         max_linear_accel = self.max_linear_accel,
                                         max_rotational_speed = self.max_rotational_speed,
                                         max_rotational_accel = self.max_rotational_accel,
                                         max_joint_speed_ratio=0.7, corner_distance = 0.0)
        
        self.waypoint = MotionWaypoint(options = self.wpt_opts.to_msg(), limb = self._limb)
        
        #Get joint names
        self.joint_names = self._limb.joint_names()
        
        #Get lights reference
        self._lights = Lights()

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
                if self._gripper.needs_init():
                    self._gripper.initialize()
                msg = "Smart Click Connection found and initialized, ready for use"
                rospy.logwarn(msg)
                
                #Set tip name to smart adapter connection
                self.tip_name = "stp_021709TP00448_tip"

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
        
        #Fk service info
        self._ns = "ExternalTools/" + "right" + "/PositionKinematicsNode/FKService"
        self._fksvc = rospy.ServiceProxy(self._ns, SolvePositionFK)
        self._joints = JointState()
        self._fkreq = SolvePositionFKRequest()
        # Add desired pose for forward kinematics
        self._fkreq.configuration.append(self._joints)
        # Request forward kinematics from base to "right_hand" link
        self._fkreq.tip_names.append(self.tip_name)

        #Ik service client
        self._ik_ns = "ExternalTools/" + "right" + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(self._ik_ns, SolvePositionIK)
        self._ikreq = SolvePositionIKRequest()
        ik_hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ik_poses = {
            'right': PoseStamped(
                header=ik_hdr,
                pose=Pose(
                    position=Point(
                        x=0.450628752997,
                        y=0.161615832271,
                        z=0.217447307078,
                    ),
                    orientation=Quaternion(
                        x=0.704020578925,
                        y=0.710172716916,
                        z=0.00244101361829,
                        w=0.00194372088834,
                    ),
                ),
            ),
        }
        # Add desired pose for inverse kinematics
        self._ikreq.pose_stamp.append(ik_poses['right'])
        # Request inverse kinematics from base to "right_hand" link
        self._ikreq.tip_names.append(self.tip_name)


    def ik_srv(self,position,orientation):

        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        poses = {
            'right': PoseStamped(
                header=hdr,
                pose=Pose(
                    position=Point(
                        x=position[0],
                        y=position[1],
                        z=position[2],
                    ),
                    orientation=Quaternion(
                        x=orientation[0],
                        y=orientation[1],
                        z=orientation[2],
                        w=orientation[3],
                    ),
                ),
            ),
        }

        # Add desired pose for inverse kinematics
        self._ikreq.pose_stamp[0] = poses['right']

        try:
            rospy.wait_for_service(self._ik_ns, 5.0)
            resp = self._iksvc(self._ikreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        
        positions = list(resp.joints[0].position)
        positions[5] = positions[5] + 0.002
        positions = tuple(positions)

        return dict(zip(resp.joints[0].name, positions))

    def fk_srv(self):

        self._joints.name = self.joint_names
        self._joints.position = self.joint_angles

        # Add desired pose for forward kinematics
        self._fkreq.configuration[0] = self._joints

        try:
            rospy.wait_for_service(self._ns, 5.0)
            resp = self._fksvc(self._fkreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        
        return resp


    def _on_joint_states(self, msg):
        self.joint_names=msg.name
        self.joint_angles=msg.position

    def _pos_path(self):
        
        # Get the directory of the current script
        dirname = os.path.dirname(os.path.abspath(__file__))
        
        # Define the relative path components
        relative_path_components = ['positions']

        # Join the directory path and the relative path components
        folder_name = os.path.join(dirname, *relative_path_components)
        
        #Create folder if does not exist
        try:
            os.mkdir(folder_name)
            print("Directory %s created.", folder_name)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory already exists; not created.')

        # Define the relative path components  
        relative_path_components = ['sawyer_positions.csv']

        # Join the directory path and the relative path components
        positions_path = os.path.join(folder_name, *relative_path_components)
        
        return positions_path
        
    #Positions for movement convenience (CARTESIAN):
    def _read_positions(self):
        
        if os.path.exists(self._positions_path):
        
            pos = np.zeros((1,6))
            pos_aux = np.zeros((1,6))
            
            #Extract all Inputs from train file as a single numpy array for ARTa and ARTb
            with open(self._positions_path, 'r') as file:
                
                # Read all lines and process each one
                lines = file.readlines()

                for line in lines:
                    pos_str = line.strip().split(',') #Extract EOL and split in a list with separator ","
                    
                    #Convert to float
                    for i in range(len(pos_str)):
                        pos_aux[0][i] = float(pos_str[i])
                    
                    #Stack current line vector into a matrix
                    pos = np.vstack((pos, pos_aux))
                    
            if len(lines) > 0:
               	#Delete first row as it was auxiliary
                pos = np.delete(pos, 0, axis=0)
            else:
                pos = []
                
        else:
            msg = "No file named sawyer_positions.csv found, creating empty file to start saving positions"
            
            #Create empty file if it does not exist
            with open(self._positions_path, 'w') as f:
                pass        
            
            rospy.loginfo(msg)
            pos = []
                
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
        
        np.savetxt(self._positions_path, self.positions, delimiter=',', fmt='%f')
        rospy.loginfo('New position saved')

    #Give current endpoint pose with orientation in euler angles
    def current_endpoint_pose(self, quaternion=False):
        
        endpoint = self.fk_srv()

        x = endpoint.pose_stamp[0].pose.position.x
        y = endpoint.pose_stamp[0].pose.position.y
        z = endpoint.pose_stamp[0].pose.position.z

        endpoint_position = [x,y,z]

        x = endpoint.pose_stamp[0].pose.orientation.x
        y = endpoint.pose_stamp[0].pose.orientation.y
        z = endpoint.pose_stamp[0].pose.orientation.z
        w = endpoint.pose_stamp[0].pose.orientation.w

        endpoint_orientation_quaternion = [x,y,z,w]

        #Obtain pose through transforms
        #if self.tf.frameExists(self.tip_name) and self.tf.frameExists("base"):
            #t = self.tf.getLatestCommonTime("base",self.tip_name)
            #endpoint_position, endpoint_orientation_quaternion = self.tf.lookupTransform("base",self.tip_name, rospy.Time(0))

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
                    self._gripper.set_ee_signal_value('grip', True)
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
                    self._gripper.set_ee_signal_value('grip', False)
                else:
                    self._gripper.close()
        else:
            rospy.loginfo("Gripper not detected")
    
    #Manual movement of the joint positions in radians as a list
    def move_to_joint_positions(self, positions=[]):
        
        #Reinitialize trajectory so that only one waypoint is executed
        self.traj = MotionTrajectory(trajectory_options = self._traj_options, limb = self._limb)

        #Set waypoint
        self.waypoint.set_joint_angles(positions, self.tip_name, self.joint_names)
        
        #Send movement
        self.traj.append_waypoint(self.waypoint.to_msg())

        #Check trajectory result
        result = self.traj.send_trajectory(timeout=self.timeout)
        
        if result is None:
            rospy.logerr('Trajectory FAILED to send')
            return

        if result.result:
            rospy.loginfo('Motion controller successfully finished the trajectory!')
        else:
            rospy.logerr('Motion controller failed to complete the trajectory with error %s',
                         result.errorId)
    
    def move_to_home(self):
        
        #Set home joint angles
        home_joint_angles = [-1.078287109375, -1.2647822265625, -1.0171767578125, 1.60878125, 0.281830078125, 1.3792490234375, 1.16204296875]
        
        rospy.loginfo("Moving arm to home position...")

        self.move_to_joint_positions(home_joint_angles)
        
        rospy.loginfo("Moved to home angles: %s", home_joint_angles)  

    #Moves to desired absolute position using linear interpolation (Orientation in Euler angles ori = [roll, pitch, yaw])
    #Pos no = 0, moves to the position at position and orientation arguments, else to the saved position
    def move_to_cartesian_absolute(self, position = [0.0, -0.4, 0.34], orientation = [-3.14, 0.008, 1.5708], pos_no=0, move_confirm=True, verbose=True):
        
        #Reinitialize trajectory so that only one waypoint is executed
        self.traj = MotionTrajectory(trajectory_options = self._traj_options, limb = self._limb)

        #Position number 0 means to move to a desired position and orientation else move to position saved
        if pos_no == 0:
            move_pose = PoseStamped()
            #Get position from argument
            move_pose.pose.position.x = position[0]
            move_pose.pose.position.y = position[1]
            move_pose.pose.position.z = position[2]
            roll = orientation[0]
            pitch = orientation[1]
            yaw = orientation[2]
            rospy.loginfo("Moving to predetermined position: %s, %s",position,orientation)
        else:
            try:
                
                move_pose = PoseStamped()
                
                move_pose.pose.position.x = self.positions[pos_no-1][0]
                move_pose.pose.position.y = self.positions[pos_no-1][1]
                move_pose.pose.position.z = self.positions[pos_no-1][2]
                roll = self.positions[pos_no-1][3]
                pitch = self.positions[pos_no-1][4]
                yaw = self.positions[pos_no-1][5]
                rospy.loginfo("Moving to position index: %s", pos_no)
                
            except:
                roll = orientation[0]
                pitch = orientation[1]
                yaw = orientation[2]
                rospy.logerr("Position index not found in positions list")
        
        #Transform orientation back to quaternion as movement command need this orientation

        quaternion = quaternion_from_euler(roll, pitch, yaw)
        move_pose.pose.orientation.x = float(quaternion[0])
        move_pose.pose.orientation.y = float(quaternion[1])
        move_pose.pose.orientation.z = float(quaternion[2])
        move_pose.pose.orientation.w = float(quaternion[3])
        
        #Move to desired position
        joint_angles = self._limb.joint_ordered_angles()
        self.waypoint.set_cartesian_pose(move_pose, self.tip_name, joint_angles)
        
        if verbose:
            rospy.loginfo('Sending waypoint: \n%s', self.waypoint.to_string())

        self.traj.append_waypoint(self.waypoint.to_msg())

        #Check trajectory result
        result = self.traj.send_trajectory(timeout=self.timeout)
        
        if result is None:
            rospy.logerr('Trajectory FAILED to send')
            return

        if move_confirm:
            if result.result:
                rospy.loginfo('Motion controller successfully finished the trajectory!')
            else:
                rospy.logerr('Motion controller failed to complete the trajectory with error %s',
                            result.errorId)
            
    #Moves to desired relative position using linear interpolation (Orientation in Euler angles ori = [roll, pitch, yaw])
    def move_to_cartesian_relative(self, position, orientation, move_confirm=True, verbose=True):

        #Reinitialize trajectory so that only one waypoint is executed
        self.traj = MotionTrajectory(trajectory_options = self._traj_options, limb = self._limb)

        #Get current endpoint pose
        current_position, current_orientation = self.current_endpoint_pose()
        
        move_pose = PoseStamped()
        
        #Add movement to current position
        move_pose.pose.position.x = current_position[0] + position[1]
        move_pose.pose.position.y = current_position[1] - position[0]
        move_pose.pose.position.z = current_position[2] + position[2]
        
        #Add orientation movement to current orientation in euler angles and transform to quaternion      
	
        move_euler = [0,0,0]
        move_euler[0] = current_orientation[0] - orientation[0]
        move_euler[1] = current_orientation[1] - orientation[1] 
        move_euler[2] = current_orientation[2] - orientation[2] 
        
        roll = move_euler[0]
        pitch = move_euler[1]
        yaw = move_euler[2]

        move_quaternion = quaternion_from_euler(roll, pitch, yaw)
        
        move_pose.pose.orientation.x = float(move_quaternion[0])
        move_pose.pose.orientation.y = float(move_quaternion[1])
        move_pose.pose.orientation.z = float(move_quaternion[2])
        move_pose.pose.orientation.w = float(move_quaternion[3])
        
        #Move to desired position
        joint_angles = self._limb.joint_ordered_angles()
        self.waypoint.set_cartesian_pose(move_pose, self.tip_name, joint_angles)
        
        if verbose:
            rospy.loginfo('Sending waypoint: \n%s', self.waypoint.to_string())

        self.traj.append_waypoint(self.waypoint.to_msg())

        #Check trajectory result
        #ToDo
        #wait_for_result argument added as False, this avoids having to wait for each movement to finish before sending
        #a new movement command, check if it will work for tasks requiring continuous movement with computations
        #between each movement
        result = self.traj.send_trajectory(wait_for_result=move_confirm, timeout=self.timeout)
        
        if result is None:
            rospy.logerr('Trajectory FAILED to send')
            return
        
        #If move confirm is False controller will always return True after a trajectory has been sent, so feedback is cutoff
        if move_confirm:
            if result.result:
                rospy.loginfo('Motion controller successfully finished the trajectory!')
            else:
                rospy.logerr('Motion controller failed to complete the trajectory with error %s',
                             result.errorId)

    #Final pose as PoseStamped(), linear_speed in m/s
    #Perform linear movements only, not considering changes in rotation
    def cartesian_approach(self, move_position, joint_speed=0.1 ,linear_speed=0.1, frecuency=100, verbose=True):
       
        final_pose = PoseStamped()

        #Set desired speed (m/s)
        self._limb.set_joint_position_speed(joint_speed)
    
        #Get current endpoint pose
        current_position, current_orientation = self.current_endpoint_pose(quaternion=True)
        
        #Obtain final position from relative movement
        final_pose.pose.position.x = current_position[0] + move_position[1]
        final_pose.pose.position.y = current_position[1] - move_position[0]
        final_pose.pose.position.z = current_position[2] + move_position[2]

        #Distance to be traveled by end effector in a linear fashion
        delta = Point()
        
        #Compute steps and final time based on desired linear speed
        delta.x = abs(move_position[1])
        delta.y = abs(-move_position[0])
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
        ik_delta.x = (move_position[1]) / steps
        ik_delta.y = (-move_position[0]) / steps
        ik_delta.z = (move_position[2]) / steps

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
        
	    #Wait for position to stabilize
        time.sleep(1.0)

        if verbose:
            #Movement finished confirmation
            rospy.loginfo("Cartestian movement finished")
            rospy.loginfo("Final Pose: %s", ik_step)
        
        return True

    #Final pose as PoseStamped()
    def rotation_interpolation(self, move_orientation, joint_speed=0.1, final_time=5, frecuency=100, verbose=True):

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
	
	    #Wait for position to stabilize
        time.sleep(1.0)        

        if verbose:
            #Movement finished confirmation
            rospy.loginfo("Cartestian movement finished")
            rospy.loginfo("Final Pose: %s", ik_step)  
       
    #Change desired speed
    def set_speed(self, max_linear_speed, max_linear_accel, max_rotational_speed, max_rotational_accel, speed_ratio=1.0):

        self.wpt_opts = MotionWaypointOptions(max_linear_speed = max_linear_speed,
                                         max_linear_accel = max_linear_accel,
                                         max_rotational_speed = max_rotational_speed,
                                         max_rotational_accel = max_rotational_accel,
                                         max_joint_speed_ratio= speed_ratio)
        
        self.waypoint = MotionWaypoint(options = self.wpt_opts.to_msg(), limb = self._limb)

if __name__ == '__main__':
    
    sawyer = SawyerRobot()
    
    print(sawyer.current_endpoint_pose())
    
 
