import time

import numpy as np
import pyglet
from pyglet import gl

# Rendering window size

DEFAULT_WINDOW_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600


class ImageRenderer:

    def __init__(self, height: int = DEFAULT_WINDOW_HEIGHT, width: int = DEFAULT_WINDOW_WIDTH):
        self.window = None
        self._height = height
        self._width = width

    def render_cv2_image(self, cv2_image_hwc: np.ndarray, channel_order: str = "BGR"):

        if self.window is None:
            context = pyglet.gl.current_context
            self.window = pyglet.window.Window(width=self._width, height=self._height)

        self.window.switch_to()
        self.window.dispatch_events()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, self._width, self._height)

        self.window.clear()

        # Setup orthogonal projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glOrtho(0, self._width, 0, self._height, 0, 10)

        # Draw the image to the rendering window
        height = cv2_image_hwc.shape[0]
        width = cv2_image_hwc.shape[1]
        imgData = pyglet.image.ImageData(
            width,
            height,
            channel_order,
            cv2_image_hwc[::-1, :, :].tobytes(),
            pitch=width * 3,
        )
        imgData.blit(0, 0, 0, self._width, self._height)

        self.window.flip()

    def close(self):
        if self.window:
            try:
                self.window.close()
            except ImportError:
                pass

if __name__ == '__main__':
    # The main function will load an image and display it right-side up.
    # All image formats/orientation/bgr color ordering used should match how cv2 loads an image in this demo.
    import cv2
    import os.path

    test_image_path = "~/Downloads/test_image.png"
    test_image_path = os.path.expanduser(test_image_path)
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(test_image_path)
    test_image_hwc_bgr = cv2.imread(test_image_path)
    print(f"time image shape is {test_image_hwc_bgr.shape}")
    print(f"time image dtype is {test_image_hwc_bgr.dtype}")
    print(f"time image max is {test_image_hwc_bgr.max()}")
    print(f"time image min is {test_image_hwc_bgr.min()}")

    image_renderer = ImageRenderer(height=test_image_hwc_bgr.shape[0], width=test_image_hwc_bgr.shape[1])
    image_renderer.render_cv2_image(cv2_image_hwc=test_image_hwc_bgr, channel_order="BGR")

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        image_renderer.close()
