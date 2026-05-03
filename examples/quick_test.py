"""Example: calibrate and test motor speed control.

Run on the Raspberry Pi with the build HAT connected:
    python examples/quick_test.py
"""

import time
from betterbuildhatmotors import ModelBasedDCMotorController


if __name__ == "__main__":
    motor = ModelBasedDCMotorController("A", logging=True)
    try:
        # First run: calibrates and saves. Subsequent runs load the file.
        motor.autotune("calibration_data.npy")
        motor.start_control_loop()
        # Test: step through speeds
        #motor.run_test([90, 180, 360, 720, 0], duration=3.0)
        motor.goto(360, accel=1200)
        motor.wait()
        motor.stop()
        # Plot results (requires matplotlib)
        try:
            from betterbuildhatmotors.plotting import plot_log

            data = motor.get_log()
            plot_log(data)
        except ImportError:
            print(
                "Install matplotlib to see plots: pip install betterbuildhatmotors[plot]"
            )

    finally:
        motor.stop()
        del motor.motor

