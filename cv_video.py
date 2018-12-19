from picamera import PiCamera
from picamera.array import PiRGBArray

from subprocess import Popen, PIPE
import threading
from time import sleep
import os, fcntl
import cv2

iframe = 0

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32

#Yolo v3 is a full convolutional model. It does not care the size of input image, as long as h and w are multiplication of 32

#camera.resolution = (160,160)
#camera.resolution = (416, 416)
#camera.resolution = (544, 544)
#camera.resolution = (608, 608)
#camera.resolution = (608, 288)


#camera.capture('frame.jpg')
sleep(0.1)

rawCapture = PiRGBArray(camera, size=(640, 480))

#spawn darknet process
#yolo_proc = Popen(["./darknet",
#                   "detect",
#                   "./cfg/yolov3-tiny.cfg",
#                   "./yolov3-tiny.weights",
#                   "-thresh","0.1"],
#                   stdin = PIPE, stdout = PIPE)

#fcntl.fcntl(yolo_proc.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port = True):
    im = frame.array
    cv2.imwrite("test.jpg", im)
#               cv2.imshow('yolov3-tiny',im)
#               key = cv2.waitKey(5) 
#            except Exception:
#               pass
#            camera.capture('frame.jpg')
#            yolo_proc.stdin.write('frame.jpg\n')
#        if len(stdout.strip())>0:
#            print('get %s' % stdout)
    cv2.imshow("test",im)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)

    if key==ord('q'):
        break
