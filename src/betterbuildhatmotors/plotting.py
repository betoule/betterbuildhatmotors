"""Plotting utilities for motor control diagnostics.

Matplotlib is imported lazily so it remains an optional dependency.
"""

import numpy as np
from scipy.signal import savgol_filter


def plot_log(data):
    """Plot speed and PWM diagnostics from control log data.

    Args:
        data: numpy record array with columns 'time', 'pos', 'pwm',
              'target_speed' (as returned by ModelBasedDCMotorController.get_log()).

    Returns:
        tuple of (speed_axis, pwm_axis) matplotlib axes for further customization.
    """
    import matplotlib.pyplot as plt

    t = data["time"] - data["time"][0]
    speed = np.diff(data["pos"]) / np.diff(t)
    omega_truth = savgol_filter(
        data["pos"], window_length=10, polyorder=2, deriv=1, delta=np.diff(t).mean()
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t[1:], speed, "k,", label="dpos", linewidth=1.5)
    ax1.plot(t, omega_truth, "b-", label="speed", linewidth=1.5)
    ax1.plot(t, data["target_speed"], "k--", label="target")
    ax1.set_ylabel("Speed (tick/s)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t, data["pwm"], "b-", label="PWM Command", linewidth=1.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("PWM (-1 to 1)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    print("Check plots for stability and tracking.")
    return ax1, ax2
