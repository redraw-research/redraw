from duckiebots_unreal_sim.tools import XboxController, get_keyboard_turning, get_keyboard_velocity
import numpy as np


class XboxControllerPolicy:

    def __init__(self, on_step_callback=None):
        self.on_step_callback = on_step_callback


        print("Detecting gamepad (you may have to press a button on the controller)...")
        if XboxController.detect_gamepad():
            self.gamepad = XboxController()
        else:
            raise IOError("Gamepad not detected")

    def policy(self, obs, state, **kwargs):
        if self.on_step_callback is not None:
            self.on_step_callback(obs, state)

        velocity = self.gamepad.LeftJoystickY
        turning = self.gamepad.LeftJoystickX

        # action = -np.asarray([[velocity, turning]], np.float32)
        action = np.asarray([[velocity, turning]], np.float32)

        outs = {"action": action}
        return outs, state
