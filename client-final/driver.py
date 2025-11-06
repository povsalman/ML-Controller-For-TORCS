import os
import numpy as np
import joblib
import tensorflow as tf
import msgParser as m
import carState
import carControl
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BC_MODEL_FILE = "bc_model.keras"
BC_SCALER_FILE = "bc_scaler.gz"
BC_OUTPUT_SCALER_FILE = "bc_output_scaler.gz"
FEATURES_FILE = "bc_features.txt"
DATA_LOG_FILE = "e-track.csv"

class Driver:
    def __init__(self, stage: int = 3, train: bool = False, feature_set: str = "extended",
                 max_off_track_steps: int = 20, max_recovery_steps: int = 15, max_invalid_sensor_steps: int = 10):
        self.stage = stage
        self.train_mode = train
        self.parser = m.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.steer_lock = 0.785398
        self.max_speed = 300.0
        self.max_gear = 6
        self.min_gear = -1
        self.seq_len = 5
        self.seq_buffer = deque(maxlen=self.seq_len)
        self.prev_rpm = None
        self.prev_angle = 0.0
        self.prev_track_pos = 0.0
        self.prev_speed_x = 0.0
        self.off_track_steps = 0
        self.max_off_track_steps = max_off_track_steps
        self.recovery_steps = 0
        self.max_recovery_steps = max_recovery_steps
        self.post_recovery_steps = 0
        self.max_post_recovery_steps = 30
        self.force_forward_steps = 0
        self.max_force_forward_steps = 40
        self.invalid_sensor_steps = 0
        self.max_invalid_sensor_steps = max_invalid_sensor_steps
        self.use_bc_controls = True
        self.log_buffer = []
        self.log_buffer_size = 10

        # Define feature sets
        self.FEATURE_SETS = {
            "base": [
                'Angle', 'TrackPosition', 'SpeedX', 'SpeedY', 'SpeedZ', 'RPM', 'Z',
                'FuelLevel', 'RacePosition',
                *[f'WheelSpinVelocity_{i}' for i in range(1, 5)],
                *[f'Track_{i}' for i in range(1, 20)],
                *[f'Opponent_{i}' for i in range(1, 37)],
                'Gear'
            ],
            "extended": [
                'Angle', 'TrackPosition', 'SpeedX', 'SpeedY', 'SpeedZ', 'RPM', 'Z',
                'FuelLevel', 'RacePosition',
                *[f'WheelSpinVelocity_{i}' for i in range(1, 5)],
                *[f'Track_{i}' for i in range(1, 20)],
                *[f'Opponent_{i}' for i in range(1, 37)],
                'Track_Left_Avg', 'Track_Middle_Avg', 'Track_Right_Avg',
                'CurrentLapTime', 'DistanceCovered', 'DistanceFromStart', 'Gear'
            ]
        }
        self.feature_columns = self.FEATURE_SETS.get(feature_set, self.FEATURE_SETS["extended"])
        if not os.path.exists(FEATURES_FILE):
            with open(FEATURES_FILE, 'w') as f:
                for col in self.feature_columns:
                    f.write(f"{col}\n")
            logging.info(f"Created feature columns file '{FEATURES_FILE}' with {len(self.feature_columns)} features.")
        else:
            with open(FEATURES_FILE, 'r') as f:
                loaded_columns = [line.strip() for line in f]
            if loaded_columns != self.feature_columns:
                logging.warning(f"'{FEATURES_FILE}' contents do not match expected features. Overwriting.")
                with open(FEATURES_FILE, 'w') as f:
                    for col in self.feature_columns:
                        f.write(f"{col}\n")

        self.use_bc = False
        self.bc_model = None
        self.bc_scaler = None
        self.bc_output_scaler = None
        self.expected_feature_count = len(self.feature_columns)
        if os.path.exists(BC_MODEL_FILE) and os.path.exists(BC_SCALER_FILE) and os.path.exists(BC_OUTPUT_SCALER_FILE):
            try:
                self.bc_model = tf.keras.models.load_model(BC_MODEL_FILE)
                self.bc_scaler = joblib.load(BC_SCALER_FILE)
                self.bc_output_scaler = joblib.load(BC_OUTPUT_SCALER_FILE)
                self.expected_feature_count = self.bc_scaler.n_features_in_
                if len(self.feature_columns) != self.expected_feature_count:
                    logging.error(f"Feature columns count ({len(self.feature_columns)}) does not match scaler ({self.expected_feature_count}). Disabling BC.")
                else:
                    self.use_bc = True
                    logging.info(f"Loaded BC model and scalers. Expecting {self.expected_feature_count} features.")
            except Exception as e:
                logging.error(f"Failed to load BC model/scalers: {e}", exc_info=True)
                self.use_bc = False
        else:
            logging.warning("BC model or scalers not found.")

    def init(self) -> str:
        self.angles = [0 for _ in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5
        self.angles[9] = 0
        return self.parser.stringify({'init': self.angles})

    def _validate_sensors(self, track):
        if any(np.isnan(x) for x in track) or all(x <= 0 for x in track):
            self.invalid_sensor_steps += 1
            logging.warning(f"Invalid sensors detected for {self.invalid_sensor_steps} steps.")
            return False
        self.invalid_sensor_steps = 0
        return True

    def _get_state_vector(self) -> np.ndarray:
        track = self.state.track if self.state.track is not None else [200.0] * 19
        raw_track = track[:]
        opponents = self.state.opponents if self.state.opponents is not None else [200.0] * 36
        wheel_spin = self.state.wheelSpinVel if self.state.wheelSpinVel is not None else [0.0] * 4
        track_pos = self.state.trackPos or 0.0
        angle = self.state.angle or 0.0
        speed_x = self.state.getSpeedX() or 0.0

        track = [200.0 if x <= 0 else x for x in track]
        self._validate_sensors(raw_track)

        features = {
            'Angle': angle,
            'TrackPosition': track_pos,
            'SpeedX': speed_x,
            'SpeedY': self.state.getSpeedY() or 0.0,
            'SpeedZ': self.state.getSpeedZ() or 0.0,
            'RPM': self.state.getRpm() or 0.0,
            'Z': self.state.z or 0.0,
            'FuelLevel': getattr(self.state, "fuel", 0.0),
            'RacePosition': getattr(self.state, "racePos", 1.0),
            'CurrentLapTime': getattr(self.state, "curLapTime", 0.0),
            'DistanceCovered': getattr(self.state, "distRaced", 0.0),
            'DistanceFromStart': getattr(self.state, "distFromStart", 0.0),
            'Gear': self.state.getGear() or 1
        }

        for i in range(4):
            features[f'WheelSpinVelocity_{i+1}'] = wheel_spin[i]
        for i in range(19):
            features[f'Track_{i+1}'] = track[i]
        for i in range(36):
            features[f'Opponent_{i+1}'] = opponents[i]

        if "Track_Left_Avg" in self.feature_columns:
            left_indices = [i for i in range(6)]
            mid_indices = [i for i in range(6, 13)]
            right_indices = [i for i in range(13, 19)]
            features['Track_Left_Avg'] = float(np.mean([track[i] for i in left_indices]))
            features['Track_Middle_Avg'] = float(np.mean([track[i] for i in mid_indices]))
            features['Track_Right_Avg'] = float(np.mean([track[i] for i in right_indices]))

        logging.debug(f"Raw track sensors: [{', '.join([f'{x:.1f}' for x in raw_track])}]")
        logging.debug(f"Parsed FuelLevel: {features['FuelLevel']:.3f}")
        logging.debug(f"Track Stats: min={min(track):.1f}, max={max(track):.1f}, mean={np.mean(track):.1f}")

        state_vector = []
        for col in self.feature_columns:
            if col in features:
                state_vector.append(features[col])
                logging.debug(f"Feature '{col}': {features[col]:.3f}")
            else:
                logging.warning(f"Feature '{col}' not available in state. Using 0.0.")
                state_vector.append(0.0)

        if len(state_vector) != self.expected_feature_count:
            logging.error(f"Feature vector length mismatch. Got {len(state_vector)}, expected {self.expected_feature_count}.")
            raise ValueError(f"Feature vector length mismatch: {len(state_vector)} vs {self.expected_feature_count}")

        return np.array(state_vector, dtype=np.float32)

    def bc_action(self):
        rpm = self.state.getRpm() or 0.0
        track_pos = self.state.trackPos or 0.0
        angle = self.state.angle or 0.0
        speed = self.state.getSpeedX() or 0.0
        wheel_spin = self.state.wheelSpinVel if self.state.wheelSpinVel is not None else [0.0] * 4
        track = self.state.track if self.state.track is not None else [200.0] * 19
        invalid_sensors = not self._validate_sensors(track)

        # fallback
        steer_rule = (angle - track_pos * 0.8) / self.steer_lock
        if abs(track_pos) > 0.3 or invalid_sensors:
            steer_rule *= 2.5
        if invalid_sensors:
            steer_rule = -np.sign(track_pos)
        steer_rule = float(np.clip(steer_rule, -1.0, 1.0))

        if not self.use_bc:
            self.prev_rpm = rpm
            return m.fail(self)

        try:
            raw_vec = self._get_state_vector().reshape(1, -1)
            scaled_vec = self.bc_scaler.transform(raw_vec)[0]
            if len(self.seq_buffer) == 0:
                for _ in range(self.seq_len):
                    self.seq_buffer.append(scaled_vec.copy())
            else:
                self.seq_buffer.append(scaled_vec)
            seq_input = np.stack(self.seq_buffer, axis=0).reshape(1, self.seq_len, -1)
            y_scaled = self.bc_model.predict(seq_input, verbose=0)
            action = self.bc_output_scaler.inverse_transform(y_scaled)[0]
            steer_bc, accel_bc, brake_bc, clutch_bc, gear_bc = action
            steer_bc = float(np.clip(steer_bc, -1.0, 1.0))
            accel_bc = float(np.clip(accel_bc, 0.0, 1.0))
            brake_bc = float(np.clip(brake_bc, 0.0, 1.0))
            clutch_bc = float(np.clip(clutch_bc, 0.0, 1.0))
            gear_bc = int(np.clip(round(gear_bc), self.min_gear, self.max_gear))
            logging.info(f"BC predictions: steer={steer_bc:.3f}, accel={accel_bc:.3f}, brake={brake_bc:.3f}, clutch={clutch_bc:.3f}, gear={gear_bc}")
        except Exception as e:
            self.prev_rpm = rpm
            return m.fail(self)

        _, accel_rule, brake_rule, clutch_rule, gear_rule = m.fail(self)

        # BC controls
        if self.use_bc_controls and not invalid_sensors and abs(track_pos) < 0.5:
            if abs(track_pos) > abs(self.prev_track_pos) and abs(track_pos) > 0.3:
                logging.info(f"BC controls unstable (trackPos={track_pos:.3f})")
                steer_final = steer_rule
                accel_final = accel_rule
                brake_final = brake_rule
                clutch_final = clutch_rule
                gear_final = gear_rule
            else:
                steer_final = 0.7 * steer_bc + 0.3 * steer_rule
                accel_final = accel_bc
                brake_final = brake_bc
                clutch_final = clutch_bc
                gear_final = gear_bc
                logging.info(f"Using BC controls: steer={steer_final:.3f}, accel={accel_final:.3f}, brake={brake_final:.3f}, clutch={clutch_final:.3f}, gear={gear_final}")
        else:
            logging.info(f"Invalid sensors or off-track (trackPos={track_pos:.3f})")
            steer_final = steer_rule
            accel_final = accel_rule
            brake_final = brake_rule
            clutch_final = clutch_rule
            gear_final = gear_rule

        self.prev_rpm = rpm
        self.prev_track_pos = track_pos
        return steer_final, accel_final, brake_final, clutch_final, gear_final


    def drive(self, msg: str) -> str:
        self.state.setFromMsg(msg)
        logging.info(f"Raw TORCS message: {msg[:500]}...")
        steer, accel, brake, clutch, gear = self.bc_action()
        if steer is None:
            logging.info("Requesting reset via meta command.")
            return '(meta 1)'
        self.control.steer = steer
        self.control.accel = accel
        self.control.brake = brake
        self.control.clutch = clutch
        self.control.gear = gear
        track = self.state.track if self.state.track is not None else [200.0] * 19
        raw_track = track[:]
        track = [200.0 if x <= 0 else x for x in track]
        opponents = self.state.opponents if self.state.opponents is not None else [200.0] * 36
        wheel_spin = self.state.wheelSpinVel if self.state.wheelSpinVel is not None else [0.0] * 4
        logging.info(f"Sensors: angle={self.state.angle or 0.0:.3f}, trackPos={self.state.trackPos or 0.0:.3f}, "
                     f"speedX={self.state.getSpeedX() or 0.0:.3f}, speedY={self.state.getSpeedY() or 0.0:.3f}, "
                     f"rpm={self.state.getRpm() or 0.0:.3f}, damage={getattr(self.state, 'damage', 0):.1f}, "
                     f"track=[{', '.join([f'{x:.1f}' for x in track])}], "
                     f"raw_track=[{', '.join([f'{x:.1f}' for x in raw_track])}], "
                     f"wheelSpinVel=[{', '.join([f'{x:.3f}' for x in wheel_spin])}]")
        logging.info(f"Controls: steer={steer:.3f}, accel={accel:.3f}, brake={brake:.3f}, clutch={clutch:.3f}, gear={gear}")
        logging.info(f"Recovery: off_track_steps={self.off_track_steps}, recovery_steps={self.recovery_steps}, "
                     f"force_forward_steps={self.force_forward_steps}, post_recovery_steps={self.post_recovery_steps}, "
                     f"invalid_sensor_steps={self.invalid_sensor_steps}")

        # Buffer log entry
        wheel_spin_str = ','.join([str(x) for x in wheel_spin])
        track_str = ','.join([str(x) for x in track])
        opponents_str = ','.join([str(x) for x in opponents])
        log_entry = (f"{self.state.angle},{self.state.trackPos},{self.state.getSpeedX()},{self.state.getSpeedY()},"
                     f"{self.state.getSpeedZ()},{self.state.getRpm()},{self.state.z},{getattr(self.state, 'fuel', 0.0)},"
                     f"{getattr(self.state, 'racePos', 1.0)},{wheel_spin_str},{track_str},{opponents_str},"
                     f"{getattr(self.state, 'curLapTime', 0.0)},{getattr(self.state, 'distRaced', 0.0)},"
                     f"{getattr(self.state, 'distFromStart', 0.0)},{self.state.getGear() or 1},"
                     f"{steer},{accel},{brake},{clutch},{gear}\n")
        self.log_buffer.append(log_entry)
        if len(self.log_buffer) >= self.log_buffer_size:
            with open(DATA_LOG_FILE, "a") as f:
                f.writelines(self.log_buffer)
            self.log_buffer.clear()

        return self.control.toMsg()

    def onShutDown(self):
        if self.log_buffer:
            with open(DATA_LOG_FILE, "a") as f:
                f.writelines(self.log_buffer)
            self.log_buffer.clear()
        logging.info("Driver shutting down.")

    def onRestart(self):
        self.prev_rpm = None
        self.prev_angle = 0.0
        self.prev_track_pos = 0.0
        self.prev_speed_x = 0.0
        self.off_track_steps = 0
        self.recovery_steps = 0
        self.post_recovery_steps = 0
        self.force_forward_steps = 0
        self.invalid_sensor_steps = 0
        self.seq_buffer.clear()
        logging.info("Driver restarted.")

    def reward(self):
        speed = self.state.getSpeedX() or 0.0
        track_pos = self.state.trackPos or 0.0
        reward = speed * (1 - abs(track_pos)) - 100 * (abs(track_pos) > 1.0)
        logging.debug(f"Reward: {reward:.3f} (speed={speed:.3f}, track_pos={track_pos:.3f})")
        return reward