from pid import PID
from lowpass import SimpleLowPassFilter
from yaw_controller import YawController
import rospy


GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        self.vehicle_mass = args[0]
        self.fuel_capacity = args[1]
        self.brake_deadband = args[2]
        self.decel_limit = args[3]
        self.accel_limit = args[4] 
        self.wheel_radius = args[5]
        self.wheel_base = args[6]
        self.steer_ratio = args[7]
        self.max_lat_accel = args[8]
        self.max_steer_angle = args[9]
        
        # Needs TUNING
        self.pidvelocity = PID(0.001, 0.01, 0.0, mn=0.0, mx=1.0) # cap throttle between 0 to 1
        
        #For braking Needs TUNING
        self.pidbrake = PID(0.5, 20.0, 0.0, mn=0.0, mx=100.0)

        self.controlsteering = YawController(self.wheel_base, self.steer_ratio, ONE_MPH, self.max_lat_accel, self.max_steer_angle)

        self.lowpasssteer = SimpleLowPassFilter(0.4)
        # last_brake_torque = 0
        self.max_accel = 10 #m/s^2
        self.max_jerk = 10 #m/s^3
        self.last_brake_torque = 0
    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        throttle, brake, steer = 1., 0., 0.
        target_linear_velocity = args[0]
        target_angular_velocity = args[1]
        current_linear_velocity = args[2]
        dbw_enabled = args[3]
        sample_time = args[4]

        error_linear_velocity = target_linear_velocity - current_linear_velocity
        # rospy.loginfo("Controller.control >>> target_linear_velocity %f, current_linear_velocity %f" % (target_linear_velocity, current_linear_velocity))
        steer = self.controlsteering.get_steering(target_linear_velocity, target_angular_velocity, current_linear_velocity)
        steer = self.lowpasssteer.filt(steer)

        if current_linear_velocity < target_linear_velocity:
            throttle = self.pidvelocity.step(error_linear_velocity, sample_time)
            brake = 0.0
        else:
            brake = self.pidbrake.step(-error_linear_velocity, sample_time)
            throttle = 0.0

        #Braking happens when current_v is bigger than target_v "current_v > target_v" and throttle is zero
        #This also includes when the target_v is very low (almost zero)
        # max_brake_torque = self.decel_limit * self.vehicle_mass * self.wheel_radius
#         if current_linear_velocity > target_linear_velocity:
#
#             brake = min(abs(self.last_brake_torque + self.max_jerk * sample_time * self.vehicle_mass * self.wheel_radius)
#                         , abs(max_brake_torque))
#             throttle = 0
#             if brake < self.brake_deadband:
#                 brake = 0
# 	        # rospy.loginfo("braking is %s N.m", brake)
#         else:
#             brake = max(0, self.last_brake_torque - self.max_jerk * sample_time * self.vehicle_mass * self.wheel_radius)
#
#         #I reduced the last_brake_torque to 0.75 of the original braking tourque as this will yield to a smoother drive by
#         #maintaining the speed and reduce the braking error
#         self.last_brake_torque = brake*0.75

        '''We should also include applying brakes constantly when the light is red or there is an obstacle
        This is done by checking if the target_linear_velocity = 0
        if throttle < 0.:
                   brake = -throttle
                   throttle = 0.
        --------------------------------------------------------------------------------------------------
        this code below is tested and works fine too. However i believe we are required to have braking values in N.m
        this is why i don't recommend using it.
        if current_linear_velocity > target_linear_velocity:
                brake = self.pidbrake.step(-error_linear_velocity,sample_time)
        rospy.loginfo("braking is %s N.m",brake)
        throttle = 0
        if brake < self.brake_deadband:
            brake = 0'''

        # rospy.loginfo("Controller.control <<< throttle %f, brake %f, steer %f" % (throttle, brake, steer))
        return throttle, brake, steer
