#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import WrenchStamped
from devices_interface import ATI_Net as ati

def publish_ftdata(sensor_ref):
    pub = rospy.Publisher('/robot/ati_ft_sensor_topic/', WrenchStamped, queue_size=1)
    pub_raw = rospy.Publisher('/robot/ati_ft_sensor_topic_raw/', WrenchStamped, queue_size=1)
    rospy.init_node('ati_ftsensor_data_pub', anonymous=True)
    rate = rospy.Rate(100) # 100hz

    #Counter to confirm sampling
    i=0

    #Initialize auxiliary variable to publish data
    ft_wrench = WrenchStamped()

    #Start ATI ft data broadcast
    sensor_ref.start_measuring()
    
    #Get sensor raw data Force (N) and Moment (Nm)
    #Initial Measurement to confirm connection, if no error, data has been read succesfully
    raw_data = sensor_ref.get_data()
    rospy.loginfo("Connection established, ATI ft data sampling ongoing")

    #run while rospy is not shutdown
    while not rospy.is_shutdown():
        
        #Get sensor raw data Force (N) and Moment (Nm)
        raw_data = sensor_ref.get_data()
        
        #Put data in auxiliary data to publish
        ft_wrench.wrench.force.x = raw_data[0]
        ft_wrench.wrench.force.y = raw_data[1]
        ft_wrench.wrench.force.z = raw_data[2]
        ft_wrench.wrench.torque.x = raw_data[3]
        ft_wrench.wrench.torque.y = raw_data[4]
        ft_wrench.wrench.torque.z = raw_data[5]

        pub_raw.publish(ft_wrench)

        #Normalize data
        #Check limits of force and moment
        raw_data[0] = (raw_data[0] + 8)/16
        raw_data[1] = (raw_data[1] + 8)/16
        raw_data[2] = (raw_data[2] + 20)/40
        raw_data[3] = (raw_data[3] + 1.5)/3
        raw_data[4] = (raw_data[4] + 1.5)/3
        raw_data[5] = (raw_data[5] + 1.5)/3

        raw_data = np.clip(raw_data,0.000,1.000)
        
        #Put data in auxiliary data to publish
        ft_wrench.wrench.force.x = raw_data[0]
        ft_wrench.wrench.force.y = raw_data[1]
        ft_wrench.wrench.force.z = raw_data[2]
        ft_wrench.wrench.torque.x = raw_data[3]
        ft_wrench.wrench.torque.y = raw_data[4]
        ft_wrench.wrench.torque.z = raw_data[5]

        pub.publish(ft_wrench)
        rate.sleep()

    #Stop ATI ft data broadcast
    sensor_ref.stop_measuring()

if __name__ == '__main__':
    try:
        #Initialize sensor
        delta = ati.DeltaFtSensor()

        #If no error while initializing start publishing data
        publish_ftdata(delta)

    except rospy.ROSInterruptException:
        pass
