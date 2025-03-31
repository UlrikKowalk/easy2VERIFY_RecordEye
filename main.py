import os
import random
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
import uuid
import serial
import serial.tools.list_ports

from Core.VideoSourceMulti import VideoSourceMulti
from Core.HeadTracker import HeadTracker
from Core.FaceMapping import FaceMapping

with open('config.yml') as config:
    configuration = yaml.safe_load(config)
    parameters_general = configuration['parameters_general']


class VideoInputThread(QtCore.QThread):
    new_frame_obtained = QtCore.Signal(np.ndarray, int, float)
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
                    self.new_frame_obtained.emit(frame, self.cam_id, self.fps)
                else:
                    self.is_running = False
                    self.video_file_ended.emit()
                time.sleep((1.0 / self.fps) - ((time.time() - start_time) % (1.0 / self.fps)))
            print('Video thread stopped.')

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

class ArduinoThread(QtCore.QThread):
    arduino_connect = QtCore.Signal()
    arduino_disconnect = QtCore.Signal()

    def __init__(self, num_leds, port='COM3'):
        super().__init__()
        self.wait_interval = 0.02
        self.port = port
        self.arduino = serial.Serial(port=self.port, baudrate=9600, timeout=.1)
        self.is_running = True
        self.num_leds = num_leds
        self.led_pos = int((self.num_leds+1)/2)
        self.is_ready = False
        time.sleep(2)

    def run(self):
        print('Starting Arduino thread.')

        while self.is_running:

            if self.is_ready and not os.path.exists(self.port):
                self.arduino_disconnect.emit()
                self.is_ready = False
            elif not self.is_ready and os.path.exists(self.port):
                if self.arduino.readline().decode().strip() == "READY":
                    self.arduino_connect.emit()
                    self.is_ready = True

            self.arduino.write(f'{self.led_pos}.\n'.encode('utf-8'))
            time.sleep(self.wait_interval)


        print('Arduino thread stopped.')

    def set_data(self, led_pos):
        self.led_pos = int(led_pos)
        # print(f'received new led: {self.led_pos}')

    def stop(self):
        self.is_running = False
        self.arduino.write(f'{-255}.\n'.encode('utf-8'))
        time.sleep(1)
        self.arduino.close()

    def clear_leds(self):
        self.led_pos = -255

    def get_ready(self):
        return self.is_ready

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Eye recorder")
        self.setStyleSheet("background-color: white;")
        self.face_mapping = FaceMapping()

        self.width_video_full = 320
        self.height_video_full = 240
        self.face_x = 0
        self.face_y = 0
        self.iris_l = [0, 0]
        self.iris_r = [0, 0]
        self.rec_factor = 0.95  # used to smoothen data

        self.frame_idx = 0

        self.cam_order = {}
        for cam in parameters_general['cameras']:
            #              name,      id,     orientation
            self.cam_order[cam[0]] = (cam[1], cam[2])

        self.cam_in_use = 0
        self.is_first = True
        self.is_cam_running = False
        self.is_arduino_ready = False
        self.is_blink = False
        self.is_recording = False

        # self.video_writer = None

        self.head_azimuth_cam = 0
        self.head_azimuth_cam_absolute = 0
        self.head_elevation_cam = 0
        self.head_roll_cam = 0
        self.face_distance = 0

        self.num_leds = 237
        tmp_scaling_factor = 60 / self.num_leds
        self.min_angle = -45
        self.max_angle = 45
        self.freq1 = 0.25 * tmp_scaling_factor
        self.amp1 = 1
        self.phase1 = 2 * np.pi * random.random()
        self.freq2 = 0.0875 * tmp_scaling_factor
        self.amp2 = 1
        self.phase2 = 2 * np.pi * random.random()

        # stores the individual graphic file names in list
        self.list_filename = []
        # stores the individual targets in list
        self.list_target = []
        # stores head rotation in list
        self.list_head_rotation = []
        # stores head elevation in list
        self.list_head_elevation = []
        # stores head roll in list
        self.list_head_roll = []
        # stores face distane in list
        self.list_face_distance = []

        self.new_user_directory_name = ''
        self.name_directory_data = './data'
        if not os.path.exists(self.name_directory_data):
            os.makedirs(self.name_directory_data)
            print(f'New directory created: {self.name_directory_data}')

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

        self.button_save = QtWidgets.QPushButton('Save')
        self.button_save.setEnabled(False)
        self.button_save.clicked.connect(self.on_click_save)
        self.button_record = QtWidgets.QPushButton('Record')
        self.button_record.setEnabled(False)
        self.button_record.clicked.connect(self.on_click_record)

        layout_menu.addWidget(self.button_record)
        layout_menu.addWidget(self.button_save)

        self.video_monitor_labels = []

        # left eye
        tmp = QtWidgets.QLabel(self)
        tmp.resize(parameters_general["video_width"], parameters_general["video_height"])
        self.video_monitor_labels.append(tmp)
        layout_video.addWidget(tmp)
        # right eye
        tmp = QtWidgets.QLabel(self)
        tmp.resize(parameters_general["video_width"], parameters_general["video_height"])
        self.video_monitor_labels.append(tmp)
        layout_video.addWidget(tmp)

        # create Aruidno thread for led display
        try:
            self.arduino_thread = ArduinoThread(self.num_leds, port="COM4")
            self.arduino_thread.arduino_connect.connect(self.arduino_connect)
            self.arduino_thread.arduino_disconnect.connect(self.arduino_disconnect)
            self.arduino_thread.start()
        except:
            print('No Arduino device found.')

        # create a separate video thread for each camera
        self.video_threads = []
        for cam in self.cam_order:
            self.video_threads.append(VideoInputThread(self.cam_order[cam][0]))
        for video_thread in self.video_threads:
            video_thread.new_frame_obtained.connect(self.new_frame_available)
            video_thread.start()
            video_thread.set_grayscale(True)

        try:
            self.headtracker_thread = HeadTrackerThread()
            self.headtracker_thread.start()
        except:
            self.use_head_tracker = False
            print('No dedicated head tracking device found.')

    def __del__(self):
        print('Stopping threads.')
        self.arduino_thread.stop()
        for video_thread in self.video_threads:
            video_thread.stop()
        if self.use_head_tracker:
            self.headtracker_thread.stop()

    def closeEvent(self, event):
        self.__del__()

    @QtCore.Slot()
    def arduino_connect(self):
        print('Arduino connected.')
        self.is_arduino_ready = True
        self.check_all_systems()

    @QtCore.Slot()
    def arduino_disconnect(self):
        print('Arduino was disconnected.')
        self.is_arduino_ready = False
        self.check_all_systems()

    def check_all_systems(self):
        if self.is_arduino_ready and self.is_cam_running:
            self.button_record.setEnabled(True)
        else:
            self.button_record.setEnabled(False)
            self.is_recording = False

    @QtCore.Slot(np.ndarray, int, float)
    def new_frame_available(self, cv_img, cam_id, fps):

        if self.is_cam_running == False:
            self.is_cam_running = True
            self.check_all_systems()

        # if frame originates from currently used camera, perform face calculations
        if self.cam_in_use == cam_id:
            if self.is_first:
                self.start_time = time.time()
            duration = time.time() - self.start_time
            self.start_time = time.time()

            if self.is_first:
                self.face_mapping.first(cv_img)
                # self.head_azimuth_cam_absolute = self.head_azimuth_cam
                # self.eye_azimuth_cam_absolute = self.eye_azimuth_cam
            try:
                self.face_mapping.calculate_face_orientation(cv_img)
                self.head_azimuth_cam, self.head_elevation_cam, self.head_roll_cam, self.face_distance = self.face_mapping.get_position_data()
                _, _, self.is_blink = self.face_mapping.get_gaze()

                # self.calibrate_estimates()
                # self.correct_azimuth()
                # self.determine_cam_in_use()
                self.update_face_coordinates()

                self.is_error = False
            except:
                self.is_error = True

            # convert full image to qt image
            qt_img_face = self.convert_cv_qt(cv_img, cv_img.shape[1], cv_img.shape[0])
            # qt_img_face = self.crop(qt_img_face)
            qt_img_eye_left, rect_left = self.crop_eye(qt_img_face, eye='left')
            qt_img_eye_right, rect_right = self.crop_eye(qt_img_face, eye='right')

            # save images of left and right eye to file in user data directory
            if self.is_recording and not self.is_blink and self.arduino_thread.get_ready():
                # top, left, height, width
                left_top, left_left, left_height, left_width = rect_left.getRect()
                right_top, right_left, right_height, right_width = rect_right.getRect()

                filename = self.save_image(cv_img[left_left:left_left + left_height, left_top:left_top + left_width, :],
                                           cv_img[right_left:right_left + right_height, right_top:right_top + right_width, :])

                self.list_filename.append(filename)
                self.list_head_rotation.append(self.head_azimuth_cam)
                self.list_head_elevation.append(self.head_elevation_cam)
                self.list_head_roll.append(self.head_roll_cam)
                self.list_face_distance.append(self.face_distance)



                led_pos, target = self.get_led_pos_and_target(fps)
                self.arduino_thread.set_data(led_pos=led_pos)
                self.list_target.append(target)

                # increase frame counter by 1
                self.frame_idx += 1

            # try:
            #     # self.update_data()
            #     # self.update_graphs_head()
            #     # self.update_graphs_eye()
            # except:
            #     self.is_error = True


            # draw camera frame onto correct video monitor
            self.update_video_display(qt_img_eye_left, qt_img_eye_right)

        if self.cam_in_use == cam_id and self.is_first:
            self.is_first = False
            if self.use_head_tracker:
                self.headtracker_thread.reset_values()

    def get_led_pos_and_target(self, fps):
        float_sample = (self.amp1 * np.sin(2 * np.pi * self.freq1 * self.frame_idx / fps + self.phase1) +
                        self.amp2 * np.sin(2 * np.pi * self.freq2 * self.frame_idx / fps + self.phase2))
        # normalized to [0,1]
        float_sample = 0.5 * (float_sample / (self.amp1 + self.amp2) + 1.0)
        float_target = (float_sample * (self.max_angle - self.min_angle) + self.min_angle) / 180 * np.pi
        float_pos = float_sample * self.num_leds
        int_pos = np.round(float_pos)
        return int_pos, float_target

    def save_image(self, img_l, img_r):
        # concatenate both eyes horizontally
        img_concat = cv2.hconcat([img_r, img_l])
        # transfer to grayscale
        # img_grayscale = cv2.cvtColor(img_concat, cv2.COLOR_BGR2GRAY)
        # factor by which to rescale
        scaling_factor = 200.0/img_concat.shape[1]
        # save under this name
        filename = (f'{self.name_directory_data}/{self.new_user_directory_name}/'
                      f'{self.new_user_directory_name}_{self.frame_idx}.png')
        # save image
        cv2.imwrite(filename,
                    cv2.resize(img_concat, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA))

        return f'{self.new_user_directory_name}_{self.frame_idx}.png'

    def update_video_display(self, qt_img_eye_left, qt_img_eye_right):
        # project video of left and right eye to screen
        self.video_monitor_labels[0].setPixmap(qt_img_eye_right)
        self.video_monitor_labels[1].setPixmap(qt_img_eye_left)

    @staticmethod
    def convert_cv_qt(rgb_image, desired_width, desired_height):
        """Convert from an opencv image to QPixmap"""
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        tmp = convert_to_Qt_format.scaled(desired_width, desired_height, QtCore.Qt.KeepAspectRatio,
                                          mode=QtCore.Qt.SmoothTransformation)
        return QtGui.QPixmap.fromImage(tmp)

    def crop(self, frame):
        # calculate a face width as double the distance between both irises
        face_width = int((self.iris_l[0] - self.iris_r[0]) * 2)
        # face detail is rectangular, so use same measurement
        face_height = face_width
        # leftmost starting point of face rect
        face_rect_x = int(min(frame.width(), max(0, int(self.face_x - face_width / 2))))
        # topmost starting point of face rect
        face_rect_y = int(min(frame.height(), max(0, int(self.face_y - face_height / 2))))
        # create rectangle around face
        temp_rect = QtCore.QRect(face_rect_x, face_rect_y, face_width, face_height)
        # crop new frame from face rect
        return frame.copy(temp_rect).scaled(self.height_video_full, self.height_video_full,
                                            mode=QtCore.Qt.SmoothTransformation)

    def crop_eye(self, frame, eye='left'):

        if eye == 'right':
            center = self.iris_r
        else:
            center = self.iris_l

        # calculate eye width as the distance between both irises
        eye_width = int((self.iris_l[0] - self.iris_r[0])*0.8)
        # face detail is rectangular, so use same measurement
        eye_height = eye_width
        # leftmost starting point of face rect
        eye_rect_x = int(min(frame.width(), max(0, int(center[0] - eye_width / 2))))
        # topmost starting point of face rect
        eye_rect_y = int(min(frame.height(), max(0, int(center[1] - eye_height / 2))))
        # create rectangle around face
        temp_rect = QtCore.QRect(eye_rect_x, eye_rect_y, eye_width, eye_height)
        # crop new frame from face rect
        return frame.copy(temp_rect).scaled(self.height_video_full, self.height_video_full,
                                            mode=QtCore.Qt.SmoothTransformation), temp_rect

    def update_face_coordinates(self):
        face_x, face_y = self.face_mapping.get_center()
        iris_l, iris_r, self.is_blink = self.face_mapping.get_iris()

        # if not self.is_blink:
        self.face_x = self.rec_factor * float(face_x) + (1.0 - self.rec_factor) * self.face_x
        self.face_y = self.rec_factor * float(face_y) + (1.0 - self.rec_factor) * self.face_y
        self.iris_l = (self.rec_factor * iris_l[0] + (1.0 - self.rec_factor) * self.iris_l[0],
                       self.rec_factor * iris_l[1] + (1.0 - self.rec_factor) * self.iris_l[1])
        self.iris_r = (self.rec_factor * iris_r[0] + (1.0 - self.rec_factor) * self.iris_r[0],
                       self.rec_factor * iris_r[1] + (1.0 - self.rec_factor) * self.iris_r[1])

    def on_click_record(self):
        print('Recording started.')
        self.button_record.setEnabled(False)
        self.button_save.setEnabled(True)

        self.new_user_directory_name = uuid.uuid4()
        tmp_directory = f'{self.name_directory_data}/{self.new_user_directory_name}'
        os.makedirs(tmp_directory)
        print(f'New directory created: {tmp_directory}')

        self.is_recording = True
        self.setStyleSheet("background-color: #ff6666;")

    def on_click_save(self):
        self.arduino_thread.clear_leds()
        self.button_save.setEnabled(False)
        self.check_all_systems()

        self.is_recording = False
        self.setStyleSheet("background-color: white;")
        tmp_directory = f'{self.name_directory_data}/{self.new_user_directory_name}'

        tmp_dataframe = pd.DataFrame({
            'filename':         self.list_filename,
            'target':           self.list_target,
            'head_rotation':    self.list_head_rotation,
            'head_elevation':   self.list_head_elevation,
            'head_roll':        self.list_head_roll,
            'face_distance':    self.list_face_distance
        })
        tmp_dataframe.to_csv(
            path_or_buf=f"{tmp_directory}/dataset.csv",
            index=False)
        print(f'{self.frame_idx} frames successfully stored.')
        self.frame_idx = 0

    def save_data_to_file(self, data):
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())
