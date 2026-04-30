"""Tests for the control logic (no hardware required)."""

import numpy as np
from betterbuildhatmotors.control import model, ff_command, control


class TestModel:
    def test_returns_array_of_two(self):
        params = {"a": 1300, "I": 10, "load": 0}
        result = model(pos=0, speed=0, pwm=0.5, dt=0.01, params=params)
        assert result.shape == (2,)

    def test_zero_pwm_decelerates(self):
        params = {"a": 1300, "I": 10, "load": 0}
        result = model(pos=0, speed=100, pwm=0.0, dt=0.01, params=params)
        assert result[1] < 100, "Speed should decrease with no PWM"

    def test_positive_pwm_accelerates(self):
        params = {"a": 1300, "I": 10, "load": 0}
        result = model(pos=0, speed=0, pwm=0.5, dt=0.01, params=params)
        assert result[1] > 0, "Speed should increase with positive PWM"


class TestFFCommand:
    def test_clipped_to_range(self):
        params = {"a": 1300, "I": 10, "load": 0}
        pwm = ff_command(speed=0, target_speed=99999, dt=0.01, params=params)
        assert pwm <= 1.0
        pwm = ff_command(speed=0, target_speed=-99999, dt=0.01, params=params)
        assert pwm >= -1.0

    def test_zero_error_small_pwm(self):
        params = {"a": 1300, "I": 10, "load": 0}
        pwm = ff_command(speed=100, target_speed=100, dt=0.01, params=params)
        assert abs(pwm) < 0.1, "Near-zero PWM when already at target speed"


class TestControl:
    def test_returns_state_of_length_8(self):
        params = {"a": 1300, "I": 10, "load": 0}
        state = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        new_state = control(state, new_position=0.0, params=params, target_speed=100)
        assert len(new_state) == 8

    def test_load_estimate_resets_on_new_target(self):
        params = {"a": 1300, "I": 10, "load": 0}
        state = (0.0, 10.0, 10.0, 50.0, 0.3, 0.0, 5.0, 100)
        new_state = control(state, new_position=9.5, params=params, target_speed=200)
        assert new_state[6] == 0.0, "Load estimate should reset when target changes"

    def test_load_estimate_accumulates_on_same_target(self):
        params = {"a": 1300, "I": 10, "load": 0}
        state = (0.0, 10.0, 10.0, 50.0, 0.3, 0.0, 0.0, 100)
        new_state = control(state, new_position=9.0, params=params, target_speed=100)
        assert new_state[6] != 0.0, "Load estimate should change when target is same"
