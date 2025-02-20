import cv2
from PySide6 import QtCore
from PySide6 import QtWidgets
from PySide6 import QtGui
import time
import yaml
import sys
import datetime
import pandas as pd
import numpy as np
import json

from Core.VideoSourceMulti import VideoSourceMulti
from Core.HeadTracker import HeadTracker
from Core.FaceMapping import FaceMapping

with open('config.yml') as config:
    configuration = yaml.safe_load(config)
parameters_general = configuration['parameters_general']


class VideoInputThread(QtCore.QThread):
    new_frame_obtained = QtCore.Signal(np.ndarray, int)
    no_camera_signal = QtCore.Signal()
    invalid_file_signal = QtCore.Signal()
    fps_detected = QtCore.Signal(int)
    video_file_ended = QtCore.Signal()

    def __init__(self, cam_id):
        super().__init__()

        self.cam_id = cam_id
        self.input_device = 'camera'
        self.filename = None
        self.fps = None
        self.grayscale = False
        self.is_running = True

    def get_fps(self):
        return self.fps

    def stop(self):
        self.is_running = False

    def set_grayscale(self, grayscale=False):
        self.grayscale = grayscale

    def run(self):

        self.is_running = True
        video_source = VideoSourceMulti(self.cam_id)

        if self.input_device == 'file':
            try:
                self.fps = video_source.start_file(filename=self.filename)
            except:
                self.invalid_file_signal.emit()
        else:
            try:
                self.fps = video_source.start_camera()
            except:
                # no camera found
                self.no_camera_signal.emit()

        if self.fps is not None:
            # use fps if plausible, else assume 30
            self.fps = self.fps if self.fps != 0 else 30

            # send fps value to main app
            self.fps_detected.emit(self.fps)

            start_time = time.time()

            while self.is_running:
                ret, frame = video_source.get_frame()

                if self.grayscale:
                    # for grayscale image
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if ret:
                    self.new_frame_obtained.emit(frame, self.cam_id)
                else:
                    self.is_running = False
                    self.video_file_ended.emit()
                time.sleep((1.0 / self.fps) - ((time.time() - start_time) % (1.0 / self.fps)))
            print('Stopped.')

class HeadTrackerThread(QtCore.QThread):
    # headtrack_data_available_signal = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.wait_interval = 0.002
        self.headtracker = HeadTracker()
        self.is_running = True
        self.data = (0, 0, 0)

    def run(self):
        print('Starting head tracking thread.')

        while self.is_running:
            self.headtracker.track()
            self.data = self.headtracker.get_data()

            # if yaw and pitch and roll:
            #     self.headtrack_data_available_signal.emit(np.array([yaw, pitch, roll]))

            time.sleep(self.wait_interval)

        print('Head tracking thread stopped.')

    def get_data(self):
        return self.data

    def reset_values(self):
        self.headtracker.reset_values()

    def stop(self):
        self.is_running = False

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Multicam Head Tracking")
        self.setStyleSheet("background-color: white;")
        self.face_mapping = FaceMapping()
        self.subject_data = dict()
        self.subject_data["name"] = ''

        self.cam_order = {}
        for cam in parameters_general['cameras']:
            #              name,      id,     orientation
            self.cam_order[cam[0]] = (cam[1], cam[2])

        # build main app layout
        layout_main = QtWidgets.QVBoxLayout()
        layout_menu = QtWidgets.QHBoxLayout()
        layout_video = QtWidgets.QHBoxLayout()
        layout_video.setContentsMargins(0, 0, 0, 0)
        layout_main.addLayout(layout_menu)
        layout_main.addLayout(layout_video)
        window_content = QtWidgets.QWidget()
        window_content.setLayout(layout_main)
        self.setCentralWidget(window_content)

        self.text_label_subject_name = QtWidgets.QLineEdit('')
        self.text_label_subject_name.setDisabled(True)
        self.button_save_result = QtWidgets.QPushButton('Save Result')
        self.button_save_result.setDisabled(True)
        # self.button_save_result.clicked.connect(self.on_click_save_result)
        self.button_new_subject = QtWidgets.QPushButton('New Subject')
        # self.button_new_subject.clicked.connect(self.on_click_new_subject)
        self.text_label_subject_name = QtWidgets.QLineEdit('')
        self.text_label_subject_name.setDisabled(True)
        self.button_record_movement = QtWidgets.QPushButton('Record Movement')
        self.button_record_movement.setDisabled(True)
        # self.button_record_movement.clicked.connect(self.on_click_record_movement)

        layout_menu.addWidget(self.button_new_subject)
        layout_menu.addWidget(self.text_label_subject_name)
        # layout_menu.addWidget(self.button_clear)
        layout_menu.addWidget(self.button_record_movement)
        layout_menu.addWidget(self.button_save_result)

        self.video_monitor_labels = []
        # create video monitors
        for _ in self.cam_order:
            tmp = QtWidgets.QLabel(self)
            tmp.resize(parameters_general["video_width"], parameters_general["video_height"])
            self.video_monitor_labels.append(tmp)
            layout_video.addWidget(tmp)

        # create a separate video thread for each camera
        self.video_threads = []
        for cam in self.cam_order:
            self.video_threads.append(VideoInputThread(self.cam_order[cam][0]))
        for video_thread in self.video_threads:
            # video_thread.new_frame_obtained.connect(self.new_frame_available)
            video_thread.start()

        # create a separate video thread for each camera
        self.video_threads = []
        for cam in self.cam_order:
            self.video_threads.append(VideoInputThread(self.cam_order[cam][0]))
        for video_thread in self.video_threads:
            # video_thread.new_frame_obtained.connect(self.new_frame_available)
            video_thread.start()

        try:
            self.headtracker_thread = HeadTrackerThread()
            self.headtracker_thread.start()
        except:
            self.use_head_tracker = False
            print('No dedicated head tracking device found.')

    def __del__(self):
        for video_thread in self.video_threads:
            video_thread.new_frame_obtained.disconnect()
            video_thread.no_camera_signal.disconnect()
            video_thread.invalid_file_signal.disconnect()
            video_thread.fps_detected.disconnect()
            video_thread.video_file_ended.disconnect()
            video_thread.stop()
            video_thread.quit()
        if self.use_head_tracker:
            self.headtracker_thread.stop()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())
