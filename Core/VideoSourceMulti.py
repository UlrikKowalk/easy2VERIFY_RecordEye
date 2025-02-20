
import cv2
import time
import platform


FRAME_WIDTH=4
FRAME_HEIGHT=3
FRAME_RATE=30
BRIGHTNESS=10
CONTRAST=11
SATURATION=12
HUE=13
GAIN=14
EXPOSURE=15


class VideoSourceMulti:

    def __init__(self, cam_id):
        self.duration = None
        self.fps = None
        self.t = time.time()
        self.cap = None
        self.cam_id = cam_id

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
        #cv2.destroyAllWindows()

    def start_camera(self):
        # obtain operating system
        operating_system = platform.system()

        try:
            # obtain image from camera
            if operating_system == 'Darwin':
                self.cap = cv2.VideoCapture(self.cam_id)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                # self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                self.cap = cv2.VideoCapture(self.cam_id)

            if self.cap.isOpened() and self.cap is not None:
                return self.cap.get(cv2.CAP_PROP_FPS)

        except:
            raise ValueError("Unable to open video source")

    def start_file(self, filename):
        #'2023-01-06 17-53-34.mkv'

        # obtain image from video
        self.cap = cv2.VideoCapture(filename)

        if not self.cap.isOpened():
            raise ValueError("Unable to open video source")
        else:
            return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                # in case of rotated GOPRO
                # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                return (ret, frame)
            else:
                return (ret, None)
        else:
            return (None, None)


