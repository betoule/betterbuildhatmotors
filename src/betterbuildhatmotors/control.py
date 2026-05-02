"""Model-based motor control for Raspberry Pi build HAT.

Better speed control using feedforward + load estimation instead of
the built-in PID, giving LEGO Spike Hub-like responsiveness.
"""

import time
import math
import numpy as np


def model(pos, speed, pwm, dt, params):
    """Predict motor velocity at next time step.

    Assumes the PWM command and friction remain constant during the time step.
    """
    accel = params["I"] * (params["a"] * pwm - speed) - params["load"]
    new_speed = accel * dt + speed
    pos = pos + new_speed * dt + 0.5 * accel * dt**2
    return np.array([pos, new_speed])


def ff_command(speed, target_speed, dt, params):
    """Compute feedforward PWM to reach target_speed."""
    Iadt = params["I"] * params["a"] * dt
    pwm_ff = (
        target_speed - speed + params["I"] * dt * speed + params["load"] * dt
    ) / Iadt
    return np.clip(pwm_ff, -1, 1)


def control(state, new_position, params, target_speed):
    """Run one iteration of the control loop.

    Args:
        state: tuple of (timing, pos, pos_estimate, speed_estimate, pwm,
               error, load_estimate, old_target)
        new_position: latest encoder reading from the motor
        params: dict with keys 'I', 'a', 'load'
        target_speed: desired speed in ticks/s

    Returns:
        Updated state tuple.
    """
    timing, pos, pos_estimate, speed_estimate, pwm, error, load_estimate, old_target = (
        state
    )
    now = time.perf_counter()
    new_error = pos_estimate - new_position
    #if target_speed == old_target:
    load_estimate += 50.0 * new_error
    #else:
    #    load_estimate = 0.0
    params["load"] = load_estimate
    new_pwm = ff_command(speed_estimate, target_speed, 0.01, params)
    pred = model(new_position, speed_estimate, new_pwm, 0.01, params)
    return (
        now,
        new_position,
        pred[0],
        pred[1],
        new_pwm,
        new_error,
        load_estimate,
        target_speed,
    )
