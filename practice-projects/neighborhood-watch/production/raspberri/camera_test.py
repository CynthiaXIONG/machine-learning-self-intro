from picamera import picamera
from time import sleep

camera = PiCamera()
camera.start_preview()
sleep(2)
camera.capture('home/pi/Desktop/cam_image.jpg')
camera.stop_preview()