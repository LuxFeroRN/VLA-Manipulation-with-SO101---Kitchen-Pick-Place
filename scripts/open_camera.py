import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

config = OpenCVCameraConfig(
    index_or_path='/dev/video2',
    fps=None,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

with OpenCVCamera(config) as camera:
    print("Camara abierta. Pulsa 'q' para salir.")
    while True:
        frame = camera.read()
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camara SO101", bgr_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
