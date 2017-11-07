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
        
        # Needs TUNING
        self.pidvelocity = PID(1.0, 1.0, 1.0, mn=0.0, mx=1.0) # cap throttle between 0 to 1
        
        #For braking Needs TUNING
        #self.pidbrake = PID(1.0, 1.0, 1.0, mn=0.0, mx=1.0) 

        self.controlsteering = YawController(wheel_base, steer_ratio, ONE_MPH, max_lat_accel, max_steer_angle)

        # last_brake_torque = 0
        # max_accel = 10 #m/s^2
        # max_jerk = 10 #m/s^3

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
        throttle = self.pidvelocity.step(error_linear_velocity, sample_time)

        steer = self.controlsteering.get_steering(target_linear_velocity, target_angular_velocity, current_linear_velocity)

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
        # We should also include applying brakes constantly when the light is red or there is an obstacle
        # This is done by checking if the target_linear_velocity = 0
        # if throttle < 0.:
        #            brake = -throttle
        #            throttle = 0.

        #Since i am not able to run the simulator yet i would also recommend trying to use the error for braking 
        #such as with the velocity.
        #If current_linear_velocity > target_linear_velocity:
        #    brake = self.pidbrake.step(-error_linear_velocity,sample_time)
        #    throttle = 0
        #    if brake < brake_deadband:
        #         brake = 0
        return throttle, brake, steer
