
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
        self.controlsteering = YawController (wheel_base, steer_ratio, min_speed = 1 , max_lat_accel, max_steer_angle)
        pass

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        #The arguments in the dbw_node are: twist vel, direction, current vel and the status of the dbw
        target_v = args[0]
        target_phi = args[1]
        current_v = args[2]
        dbw_statues = args[3]
        
        error_v = target_v - current_v
        throttle = self.pidvelocity.step(0,0,0):
        
        
        steer = self.controlsteering.get_steering(target_v, target_phi, current_v)
        
        #Braking?
        
        return throttle, brake, steer
