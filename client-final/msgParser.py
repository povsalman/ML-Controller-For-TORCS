import os
import numpy as np
import joblib
import tensorflow as tf
import msgParser as m
import carState
import carControl
from collections import deque
import logging

class MsgParser(object):
    '''
    A parser for received UDP messages and building UDP messages
    '''
    def __init__(self):
        '''Constructor'''
        pass

    def parse(self, str_sensors):
        '''Return a dictionary with tags and values from the UDP message'''
        sensors = {}
        
        b_open = str_sensors.find('(')
        
        while b_open >= 0:
            b_close = str_sensors.find(')', b_open)
            if b_close >= 0:
                substr = str_sensors[b_open + 1: b_close]
                items = substr.split()
                if len(items) < 2:
                    print("Problem parsing substring:", substr)  # Fixed print statement
                else:
                    value = items[1:]  # Simplified list slicing
                    sensors[items[0]] = value
                b_open = str_sensors.find('(', b_close)
            else:
                print("Problem parsing sensor string:", str_sensors)  # Fixed print statement
                return None
        
        return sensors
    
    def stringify(self, dictionary):
        '''Build a UDP message from a dictionary'''
        msg = ''
        
        for key, value in dictionary.items():
            if value and value[0] is not None:
                msg += f'({key} ' + ' '.join(map(str, value)) + ')'
        
        return msg

def fail(model):
    rpm = model.state.getRpm() or 0.0
    angle = model.state.angle or 0.0
    track_pos = model.state.trackPos or 0.0
    speed = model.state.getSpeedX() or 0.0
    damage = getattr(model.state, "damage", 0)
    wheel_spin = model.state.wheelSpinVel if model.state.wheelSpinVel is not None else [0.0] * 4
    max_spin = max([abs(x) for x in wheel_spin])
    track = model.state.track if model.state.track is not None else [200.0] * 19
    invalid_sensors = not model._validate_sensors(track)

    steer = (angle - track_pos * 0.8) / model.steer_lock
    if abs(track_pos) > 0.3 or invalid_sensors:
        steer *= 2.5
    if invalid_sensors:
        steer = -np.sign(track_pos)
    steer = float(np.clip(steer, -1.0, 1.0))

    accel = model.control.getAccel() or 0.0
    brake = 0.0
    gear = model.state.getGear() or 1
    clutch = 0.0

    # gear
    if gear < model.max_gear and rpm > 7000:
        gear += 1
    if gear > 1 and rpm < 3000:
        gear -= 1
    if speed < 5.0:
        gear = 1

    if (abs(track_pos) > 0.3 or abs(speed) < 2.0 or invalid_sensors) and abs(angle) < 1.0 and model.post_recovery_steps == 0 and model.force_forward_steps == 0:
        model.off_track_steps += 1
        model.recovery_steps += 1
        if model.off_track_steps > model.max_off_track_steps or model.recovery_steps > model.max_recovery_steps or abs(track_pos) > 2.0:
            logging.info("Car stuck or off-track. Forcing forward gear.")
            model.off_track_steps = 0
            model.recovery_steps = 0
            model.post_recovery_steps = model.max_post_recovery_steps
            model.force_forward_steps = model.max_force_forward_steps
            gear = 1
            accel = 0.9 if abs(speed) < 30.0 else 0.2
            brake = 0.0 if abs(speed) < 30.0 else 0.5
            steer = (-angle + track_pos * 1.5) * 12 / model.steer_lock
            steer = float(np.clip(steer, -1.0, 1.0))
        else:
            logging.info("Off-track or stuck detected. Reversing.")
            gear = -1
            accel = 0.4 if abs(speed) < 30.0 else 0.2
            brake = 0.2 if abs(speed) < 30.0 else 0.5
            steer = (-angle + track_pos * 1.5) * 12 / model.steer_lock
            steer = float(np.clip(steer, -1.0, 1.0))
    elif abs(track_pos) > 1.5 or damage > 2500 or model.invalid_sensor_steps > model.max_invalid_sensor_steps:
        model.off_track_steps += 1
        # if model.off_track_steps > model.max_off_track_steps or model.invalid_sensor_steps > model.max_invalid_sensor_steps:
        #     logging.info(f"Car stuck, heavily damaged, or invalid sensors for {model.invalid_sensor_steps} steps. Requesting reset.")
        #     model.off_track_steps = 0
        #     model.invalid_sensor_steps = 0
        #     return None, None, None, None, None
        logging.info("Off-track or damaged. Reversing.")
        gear = -1
        accel = 0.4 if abs(speed) < 30.0 else 0.2
        brake = 0.2 if abs(speed) < 30.0 else 0.5
        steer = (-angle + track_pos * 1.5) * 12 / model.steer_lock
        steer = float(np.clip(steer, -1.0, 1.0))
    else:
        model.off_track_steps = max(0, model.off_track_steps - 1)
        model.recovery_steps = 0
        if model.force_forward_steps > 0:
            logging.info("Forcing forward gear.")
            gear = 1
            accel = 0.9 if abs(speed) < 30.0 else 0.2
            brake = 0.0 if abs(speed) < 30.0 else 0.5
            steer = (-angle + track_pos * 1.5) * 12 / model.steer_lock
            steer = float(np.clip(steer, -1.0, 1.0))
            model.force_forward_steps -= 1
            model.post_recovery_steps = model.max_post_recovery_steps
        elif abs(track_pos) < 0.3 and abs(angle) < 0.2 or speed > 15.0:
            logging.info("Recovery complete. Driving forward.")
            if speed < model.max_speed:
                accel += 0.2
                accel = min(accel, 1.0)
                accel *= 0.7 if max_spin > 50 else 1.0
            else:
                accel -= 0.1
                accel = max(accel, 0.0)
            brake = 0.0
            if speed < 10.0:
                gear = 1
                if rpm > 4000 and speed > 5.0:
                    gear = 2
            else:
                up = True if model.prev_rpm is None or (model.prev_rpm - rpm) < 0 else False
                if up and rpm > 7000:
                    gear += 1
                if not up and rpm < 3000 and gear > 1:
                    gear -= 1
            if model.post_recovery_steps > 0:
                model.post_recovery_steps -= 1
        else:
            if speed < model.max_speed:
                accel += 0.2
                accel = min(accel, 1.0)
                accel *= 0.7 if max_spin > 50 else 1.0
            else:
                accel -= 0.1
                accel = max(accel, 0.0)
            brake = 0.0
            if speed < 10.0:
                gear = 1
                if rpm > 4000 and speed > 5.0:
                    gear = 2
            else:
                up = True if model.prev_rpm is None or (model.prev_rpm - rpm) < 0 else False
                if up and rpm > 7000:
                    gear += 1
                if not up and rpm < 3000 and gear > 1:
                    gear -= 1

    gear = max(model.min_gear, min(model.max_gear, gear))
    accel = max(0.0, min(1.0, accel))
    brake = max(0.0, min(1.0, brake))
    clutch = max(0.0, min(1.0, clutch))
    return steer, accel, brake, clutch, gear
