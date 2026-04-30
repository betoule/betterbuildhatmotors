
My daugther is having a hard time controlling her two-wheeled robot with the
raspberry-pi buildhat header card. For some reason the build-in PID speed
controller was having much slower response than the equivalent on the lego
spike Hub, so that her code working on the Hub was unstable on the
rapsberry-pi. She noticed that she was recovering the expected behavior on flat
surfaces by directly using the motor.set_pwm command, but the robot could not
handle slopes. 

I did not have time to go through the buildhat firmware to understand the
issue, so this is my attempt at providing her with a decent host-side
replacement of the hat speed controller by using the two responsive functions
set_pwm and get_position.

add short description of the code, installation and usage instruction.
