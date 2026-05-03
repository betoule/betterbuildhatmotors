# betterbuildhatmotors

Better model-based motor control for the Raspberry Pi build HAT.

The built-in PID speed controller on the Pi build HAT is slower than the LEGO Spike Hub, so code that works great on the Hub can be unstable on the Pi. This package replaces the built-in controller with a feedforward model that gives fast, responsive speed control -- and handles slopes too.

## Installation

```bash
python3 -m venv --system-site-packages ~/bbm
source ~/bbm/bin/activate
pip install git+https://github.com/betoule/betterbuildhatmotors.git
```

For plotting diagnostics, also install matplotlib:

```bash
pip install git+https://github.com/betoule/betterbuildhatmotors.git[plot]
```

## Quick Start

```python
from betterbuildhatmotors import ModelBasedDCMotorController

# Create controller for motor on port A
motor = ModelBasedDCMotorController('A')

# First time: auto-tune identifies your motor parameters
motor.autotune()

# Now control speed (in ticks/s)
motor.start_control_loop()
motor.set_speed(360)   # spin at 360 ticks/s
# ... do stuff ...
motor.set_speed(0)     # stop
motor.goto(360, speed=360, accel=1200) # Perform a full rotation
motor.stop()
```

## Two-Wheeled Robot Example

```python
from betterbuildhatmotors import ModelBasedDCMotorController
import time

left = ModelBasedDCMotorController('A')
right = ModelBasedDCMotorController('B')

# Calibrate both motors once
left.autotune('left_cal.npy')
right.autotune('right_cal.npy')

# Start control loops
left.start_control_loop()
right.start_control_loop()

try:
    # Drive forward
    left.set_speed(360)
    right.set_speed(360)
    time.sleep(2)

    # Turn right
    left.set_speed(180)
    right.set_speed(-180)
    time.sleep(1)

    # Stop
    left.set_speed(0)
    right.set_speed(0)
finally:
    left.stop()
    right.stop()
```

## API

### ModelBasedDCMotorController(port)

Create a controller for a motor on the given port (`'A'`, `'B'`, `'C'`, `'D'`).

### motor.autotune(calibration_file=None, show=False)

Identify motor parameters by running a calibration sequence. Pass a filename to save/load calibration data so you only need to calibrate once.

### motor.start_control_loop()

Start the background control thread. Call `autotune()` first.

### motor.set_speed(speed)

Set target speed in ticks/s. Positive = forward, negative = reverse. Call while the control loop is running to change speed on the fly.

### motor.stop()

Stop the control loop and cut power to the motor.

### motor.get_log()

Return recorded data as a numpy array for analysis.

### plot_log(data)

Plot speed and PWM diagnostics. Requires matplotlib (`pip install betterbuildhatmotors[plot]`).

## How It Works

Instead of a slow PID loop, the controller uses:

1. **Feedforward model** -- predicts the PWM needed to reach the target speed
2. **Load estimation** -- adapts to friction, weight, and slopes by tracking position error

This gives near-instant response like the LEGO Hub, plus the ability to handle inclines.

## License

MIT
