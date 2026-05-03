"""Example: calibrate and test motor speed control.

Run on the Raspberry Pi with the build HAT connected:
    python examples/quick_test.py
"""

import time
from betterbuildhatmotors import ModelBasedDCMotorController


if __name__ == "__main__":
    motor = ModelBasedDCMotorController("A", logging=False)
    try:
        # First run: calibrates and saves. Subsequent runs load the file.
        motor.autotune("calibration_data.npy")
        motor.start_control_loop()
        for i in range(1000):
            motor.goto(360, accel=1200)
            motor.wait()
            motor.goto(-10, accel=1200)
            motor.wait()
            motor.set_speed(360)
            time.sleep(1)
            motor.set_speed(720)
            time.sleep(1)
        motor.stop()
    finally:
        motor.stop()
        del motor.motor

