import time
import math
import numpy as np
from buildhat import Motor
from scipy.signal import savgol_filter
import threading
import matplotlib.pyplot as plt

def model(pos, speed, pwm, dt, params):
    """ Predict motor velocity at next time step

    The supposition is that the pwm command and friction remain the same during time step
    """ 
    accel = params['I'] * (params['a'] * pwm - speed) - params['load']
    new_speed = accel * dt + speed
    pos = pos + new_speed * dt + 0.5 * accel * dt**2
    return np.array([pos, new_speed])

def ff_command(speed, target_speed, dt, params):
    Iadt = params['I'] * params['a'] * dt
    pwm_ff = (target_speed - speed + params['I'] * dt * speed + params['load'] * dt) /Iadt  
    return np.clip(pwm_ff, -1, 1)

def control(state, new_position, params, target_speed):
    timing, pos, pos_estimate, speed_estimate, pwm, error, load_estimate, old_target = state
    now = time.perf_counter() 
    new_error = pos_estimate - new_position
    if target_speed == old_target:
        load_estimate += 50. * new_error
    else:
        load_estimate = 0.
    #Kd = 500.
    # load_estimate = Kd * new_error
    params['load'] = load_estimate
    new_pwm = ff_command(speed_estimate, target_speed, 0.01, params)
    pred = model(new_position, speed_estimate, new_pwm, 0.01, params)
    return now, new_position, pred[0], pred[1], new_pwm, new_error, load_estimate, target_speed

class ModelBasedDCMotorController:
    def __init__(self, motor_port='A', ticks_per_rev=360):
        self.motor = Motor(motor_port)
        self._log = []
        self._target_speed = 0

    # ──────────────────────────────────────────────────────────────
    #  CALIBRATION & TUNING
    # ──────────────────────────────────────────────────────────────
    def run_calibration(self, pwm_sequence=None, step_duration=1.5):
        """Excites motor with open-loop PWM steps, records (t, pos, pwm)"""
        if pwm_sequence is None:
            pwm_sequence = [0.0, 0.05, 0.1, 0.15, 0.3, 0.45, 0.55, 0.3, 0.15, 0.0, 
                           -0.15, -0.3, -0.55, -0.15, 0.0]

        print("🔄 Starting calibration...")
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

        print(f"📊 Collected {len(data)} samples")
        return np.rec.fromrecords(data, names=['time', 'pos', 'pwm'])

    def autotune(self, calibration_data=None, show=False):
        if calibration_data is None or not os.path.exists(calibration_data):
            results = self.run_calibration()
            if calibration_data is not None:
                np.save(calibration_data, results)
        else:
            results = np.load('calibration_data.npy')

        def model_pred(data, params):
            pred = [[data['pos'][0], 0]]
            dt = np.diff(data['time'])
            for i in range(len(data)-1):
                new_state = model(data['pos'][i], pred[-1][1], data['pwm'][i], dt[i], params)
                pred.append(new_state)
            return np.array(pred)
    
        def residuals(theta):
            params = {'a': theta[0],
                    'I': theta[1],
                    'load': 0}
            pred = model_pred(results, params)
            return  results['pos'] - pred[:, 0]
        import scipy.optimize
        bestfit, _ = scipy.optimize.leastsq(residuals, [1300, 10.])
        print(np.std(residuals(bestfit)))
        self.params = {'a': bestfit[0],
                       'load': 0.,
                       'I': 10}#bestfit[1]}
        pred = model_pred(results, self.params)

        if show:
            ax1, ax2 = plot_log(results)
            ax1.plot(results['time'], pred[:, 1], 'r')

        return results, pred, bestfit

    # ──────────────────────────────────────────────────────────────
    #  THREADING & TESTING
    # ──────────────────────────────────────────────────────────────
    def start_control_loop(self):
        self._running = True
        self.state = time.perf_counter(), self.motor.get_position(), self.motor.get_position(), 0, 0, 0, 0, 0
        self._log = [self.state]
 
        def loop():
            while self._running:
                self.state = control(self.state, self.motor.get_position(), self.params, self._target_speed)
                self.motor.pwm(self.state[4])
                self._log.append(self.state)
            
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=1.0)
        self.motor.pwm(0.0)

    def set_speed(self, speed):
        self._target_speed = speed

    def run_test(self, target_speeds, duration=2.0):
        """Runs step response test, records data, and plots diagnostics"""
        print("🧪 Running control test...")
        self.start_control_loop()
        t_start = time.perf_counter()
        
        try:
            for spd in target_speeds:
                self.set_speed(spd)
                time.sleep(duration)
        finally:
            self.stop()
            
    def get_log(self):
        return np.rec.fromrecords(self._log, names=['time', 'pos', 'pos_estimate', 'speed_estimate', 'pwm', 'error', 'load_estimate', 'target_speed'])

def plot_log(data):
    t = data['time'] - data['time'][0]
    speed = np.diff(data['pos'])/np.diff(t)
    omega_truth = savgol_filter(data['pos'], window_length=10, polyorder=2, deriv=1, delta=np.diff(t).mean())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t[1:], speed, 'k,', label='dpos', linewidth=1.5)
    ax1.plot(t, omega_truth, 'b-', label='speed', linewidth=1.5)
    ax1.plot(t, data['target_speed'], 'k--', label='target')
    ax1.set_ylabel('Speed (tick/s)')
    ax1.legend(); ax1.grid(True)
    
    ax2.plot(t, data['pwm'], 'b-', label='PWM Command', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('PWM (-1 to 1)')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.show()
        
    print("📈 Check plots for stability & tracking.")
    return ax1, ax2

if __name__ == '__main__':
    import os
    ma = ModelBasedDCMotorController('A')
    try:
        results, pred, bestfit = ma.autotune('calibration_data.npy')
        ma.run_test([90, 180, 360, 720, 0], 4.0)
    finally:
        ma.stop()
        del ma.motor
    data = ma.get_log()
    plot_log(data)

