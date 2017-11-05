from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        vehicle_mass = args[0]
        fuel_capacity = args[1]
        brake_deadband = args[2]
        decel_limit = args[3]
        accel_limit = args[4] 
        wheel_radius = args[5]
        wheel_base = args[6]
        steer_ratio = args[7]
        max_lat_accel = args[8]
        max_steer_angle = args[9]
        
        #Needs TUNING
        self.pidvelocity = PID(0,0,0)
        
        #Here i assigned min_speed to 1
        self.controlsteering = YawController(wheel_base, steer_ratio, ONE_MPH, max_lat_accel, max_steer_angle)

        #last_brake_torque = 0
        #max_accel = 10 #m/s^2
        #max_jerk = 10 #m/s^3
        pass

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        #The arguments in the dbw_node are: twist vel, direction, current vel and the status of the dbw
        #target_v = args[0]
        #target_phi = args[1]
        #current_v = args[2]
        #dbw_statues = args[3]
        #another argument for the time.
        #sample_time = args[4]
        
        #error_v = target_v - current_v
        #throttle = self.pidvelocity.step(error_v,sample_time)
        #We need to make sure throttle is within the range 0,1
        
        #steer = self.controlsteering.get_steering(target_v, target_phi, current_v)
        
        #Braking happens when current_v is bigger than target_v "current_v > target_v" and throttle is zero
        #This also includes when the target_v is very low (almost zero)
        
        #Braking
        #max_brake_torque = decel_limit * vehicle_mass * wheel_radius
        
        #if current_v > target_v:
            
        #    brake_torque = min(last_brake_torque + max_jerk * sample_time * vehicle_mass * wheel_radius,
        #                        max_brake_torque)
            
        #else:
        #    brake_torque = max(0, last_brake_torque - max_jerk * sample_time * vehicle_mass * wheel_radius)

        #last_brake_torque = brake_torque


        return 1., 0., 0.
