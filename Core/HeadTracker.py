import serial
import struct
import numpy as np


class HeadTracker:

    def __init__(self):

        self.header = bytes([48, 49, 32, 255])
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.baseline = (0, 0, 0)

        self.ser = serial.Serial('COM3')
        # self.ser = serial.Serial('/dev/tty.usbmodem79213101') #osx
        self.ser.reset_input_buffer()
        self.is_running = True

    def get_data(self):
        tmp_yaw = self.yaw - self.baseline[0]
        # account for circular jump
        if tmp_yaw > 180:
            tmp_yaw -= 360
        elif tmp_yaw < -180:
            tmp_yaw += 360
        return tmp_yaw, self.pitch - self.baseline[1], self.roll - self.baseline[2]

    def update_data(self, yaw, pitch, roll):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    def reset_values(self):
        self.baseline = [self.yaw, self.pitch, self.roll]

    def track(self):
        buffer = b'1234'
        # read  bytes until we found the header
        while buffer != self.header:
            buffer = buffer[1:] + self.ser.read(1)

        # get data (w/o header)
        data = self.ser.read(40)

        # valid data; get timestamp, yaw, pitch, and roll
        a, yaw, pitch, roll = struct.unpack('=L3f', data[7:23])  # 7:23

        self.update_data(yaw=yaw + 180.0, pitch=pitch, roll=roll)

    def stop(self):
        self.is_running = False
