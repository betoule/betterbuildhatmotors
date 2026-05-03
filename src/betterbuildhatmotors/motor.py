"""Model-based DC motor controller with autotuning and speed control."""

import os
import time
import threading
import numpy as np
import scipy.optimize

from buildhat import Motor

from .control import model, control
import math

def get_position_at_time(
    start_pos,
    target_pos,
    elapsed_time,
    max_speed=360,
    max_accel=100,
    max_decel=100,
):
    """
    Returns the exact target position at a given elapsed time for a 
    trapezoidal or triangular velocity profile.
    
    Args:
        start_pos: Position at t=0
        target_pos: Final target position
        elapsed_time: Time since motion started (seconds)
        max_speed: Maximum velocity magnitude (units/s)
        max_accel: Maximum acceleration magnitude (units/s²)
        max_decel: Maximum deceleration magnitude (units/s²)
        
    Returns:
        Target position at elapsed_time
    """
    if max_speed <= 0 or max_accel <= 0 or max_decel <= 0:
        raise ValueError("Speed, acceleration, and deceleration must be positive.")
        
    dist = abs(target_pos - start_pos)
    if dist < 1e-12:
        return start_pos
        
    direction = 1.0 if target_pos > start_pos else -1.0
    
    # Check if we can reach max_speed (trapezoidal) or only triangle
    d_acc_max = max_speed**2 / (2 * max_accel)
    d_dec_max = max_speed**2 / (2 * max_decel)
    
    if d_acc_max + d_dec_max <= dist:
        # Trapezoidal profile
        v_max = max_speed
        t_acc = v_max / max_accel
        t_dec = v_max / max_decel
        t_const = (dist - d_acc_max - d_dec_max) / v_max
    else:
        # Triangular profile
        v_max = math.sqrt(2 * max_accel * max_decel * dist / (max_accel + max_decel))
        t_acc = v_max / max_accel
        t_dec = v_max / max_decel
        t_const = 0.0
        
    # Actual distances covered in each phase
    d_acc = 0.5 * max_accel * t_acc**2
    d_dec = 0.5 * max_decel * t_dec**2
    t_total = t_acc + t_const + t_dec
    
    # Clamp time to trajectory bounds (prevents NaN/out-of-bounds)
    t = max(0.0, min(elapsed_time, t_total))
    
    # Position calculation per phase
    if t <= t_acc:
        # Acceleration phase
        s = 0.5 * max_accel * t**2
        speed = max_accel * t
    elif t <= t_acc + t_const:
        # Constant velocity phase
        s = d_acc + v_max * (t - t_acc)
        speed = v_max
    else:
        # Deceleration phase
        t_dec_phase = t - (t_acc + t_const)
        s = d_acc + v_max * t_const + v_max * t_dec_phase - 0.5 * max_decel * t_dec_phase**2
        speed = v_max - max_decel * t_dec_phase
    return start_pos + s * direction, speed * direction

class ModelBasedDCMotorController:
    """Drop-in replacement for buildhat Motor with better speed control.

    Uses a feedforward model + load estimation for faster, more responsive
    speed control than the built-in PID on the Raspberry Pi build HAT.

    Example:
        >>> motor = ModelBasedDCMotorController('A')
        >>> motor.autotune()
        >>> motor.set_speed(360)
        >>> time.sleep(2)
        >>> motor.stop()
    """

    def __init__(self, motor_port="A", ticks_per_rev=360, logging=False):
        self.motor = Motor(motor_port)
        self._log = []
        self._target_speed = 0
        self.logging = logging
        self._target_position = None    
    # ──────────────────────────────────────────────────────────────
    #  CALIBRATION & TUNING
    # ──────────────────────────────────────────────────────────────

    def run_calibration(self, pwm_sequence=None, step_duration=1.5):
        """Excite motor with open-loop PWM steps, record (t, pos, pwm).

        Args:
            pwm_sequence: list of PWM values to step through. Defaults to a
                sequence covering low to high speeds in both directions.
            step_duration: seconds to hold each PWM value.

        Returns:
            numpy record array with columns 'time', 'pos', 'pwm'.
        """
        if pwm_sequence is None:
            pwm_sequence = [
                0.0,
                0.05,
                0.1,
                0.15,
                0.3,
                0.45,
                0.55,
                0.3,
                0.15,
                0.0,
                -0.15,
                -0.3,
                -0.55,
                -0.15,
                0.0,
            ]

        print("Starting calibration...")
        self.motor.pwm(0.0)
        time.sleep(0.5)

        data = []
        t_start = time.perf_counter()
        step_start = t_start

        try:
            for pwm_val in pwm_sequence:
                self.motor.pwm(pwm_val)
                step_end = step_start + step_duration
                while time.perf_counter() < step_end:
                    t_now = time.perf_counter()
                    pos = self.motor.get_position()
                    data.append((t_now - t_start, pos, pwm_val))
                    time.sleep(0.008)
                step_start = time.perf_counter()
        finally:
            self.motor.pwm(0.0)

        print(f"Collected {len(data)} samples")
        return np.rec.fromrecords(data, names=["time", "pos", "pwm"])

    def autotune(self, calibration_data=None, show=False):
        """Identify motor parameters from calibration data.

        Args:
            calibration_data: path to .npy file. If the file exists, loads it;
                otherwise runs calibration and saves. If None, calibrates in
                memory without saving.
            show: if True and matplotlib is available, plots calibration results.

        Returns:
            tuple of (calibration_data, model_prediction, fitted_params).
        """
        if calibration_data is None:
            results = self.run_calibration()
        elif os.path.exists(calibration_data):
            results = np.load(calibration_data)
        else:
            results = self.run_calibration()
            np.save(calibration_data, results)

        def model_pred(data, params):
            pred = [[data["pos"][0], 0]]
            dt = np.diff(data["time"])
            for i in range(len(data) - 1):
                new_state = model(
                    data["pos"][i], pred[-1][1], data["pwm"][i], dt[i], params
                )
                pred.append(new_state.tolist())
            return np.array(pred)

        def residuals(theta):
            params = {"a": theta[0], "I": theta[1], "load": 0}
            pred = model_pred(results, params)
            return results["pos"] - pred[:, 0]

        bestfit, _ = scipy.optimize.leastsq(residuals, [1300, 10.0])
        print(f"Residual std: {np.std(residuals(bestfit)):.2f}")
        self.params = {"a": bestfit[0], "load": 0.0, "I": 10}
        pred = model_pred(results, self.params)

        if show:
            try:
                from .plotting import plot_log

                ax1, ax2 = plot_log(results)
                ax1.plot(results["time"], pred[:, 1], "r")
            except ImportError:
                print(
                    "matplotlib not installed; install with pip install betterbuildhatmotors[plot]"
                )

        return results, pred, bestfit

    # ──────────────────────────────────────────────────────────────
    #  THREADING & CONTROL
    # ──────────────────────────────────────────────────────────────

    def start_control_loop(self):
        """Start the background control thread.

        Must call autotune() first to set self.params.
        """
        self._running = True
        self.state = (
            time.perf_counter(),
            self.motor.get_position(),
            self.motor.get_position(),
            0,
            0,
            0,
            0,
            0,
        )
        if self.logging:
            self._log = [self.state]

        def loop():
            while self._running:
                self.state = control(
                    self.state,
                    self.motor.get_position(),
                    self.params,
                    self._target_speed,
                )
                self.motor.pwm(self.state[4])
                if self.logging:
                    self._log.append(self.state)
                if self._target_position is not None:
                    if abs(self._target_position - self.state[1]) < 1:
                        self._target_speed=0
                        self._target_position = None
                    else:
                        tpos, tspeed = get_position_at_time(
                            self._start_position,
                            self._target_position,
                            self.state[0] - self._start_time + 0.01,
                            self._traj_speed,
                            self._accel,
                            self._accel,
                        )
                        self._target_speed = tspeed - (self.state[2] - tpos) * 1.
                time.sleep(0.005)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the control loop and cut power to the motor."""
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)
        self.motor.pwm(0.0)

    def set_speed(self, speed):
        """Set the target speed in ticks/s.

        Call after start_control_loop() to change speed on the fly.
        """
        self._target_speed = speed
        self._target_position = None

    def goto(self, target, speed=360, accel=300):
        self._start_position = self.state[1]
        self._target_position = self.state[1] + target
        self._start_time = self.state[0]
        self._accel=accel
        self._traj_speed = speed
    def wait(self):
        while self._target_position is not None:
            time.sleep(0.02)

    def run_test(self, target_speeds, duration=2.0):
        """Run a step-response test through a sequence of speeds.

        Convenience method that starts the control loop, steps through each
        target speed for `duration` seconds, then stops.

        Args:
            target_speeds: list of speeds (ticks/s) to test sequentially.
            duration: seconds to hold each speed.
        """
        print("Running control test...")
        self.start_control_loop()
        try:
            for spd in target_speeds:
                self.set_speed(spd)
                time.sleep(duration)
        finally:
            self.stop()

    def get_log(self):
        """Return recorded control loop data as a numpy record array.

        Columns: time, pos, pos_estimate, speed_estimate, pwm, error,
        load_estimate, target_speed.
        """
        return np.rec.fromrecords(
            self._log,
            names=[
                "time",
                "pos",
                "pos_estimate",
                "speed_estimate",
                "pwm",
                "error",
                "load_estimate",
                "target_speed",
            ],
        )
