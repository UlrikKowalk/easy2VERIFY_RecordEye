import serial
import time
import os


print(os.path.exists("COM4"))

arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
time.sleep(2)
arduino.write(f'{119}.\n'.encode('utf-8'))