#------------------------------------------------------------------------
# LIBRAIRIES
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

import datetime
import time

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect

import numpy as np

import cv2
import argparse

import json
import os
print(os.getpid())
#------------------------------------------------------------------------
# VARIABLES
with open('../config/config.json', 'r') as f:
    data        = json.load(f)

frame_interval  = data["frame"]["frame_interval"]
frame_increment = data["frame"]["frame_increment"]
fps             = 30
count           = 1 # number of inferences to perform (see the loop in make_inference def)
counter         = 0
variance_blurry = 10
api_token       = data["api_token"]
#------------------------------------------------------------------------
# CONFIG CAMERA
picam2          = Picamera2()
picam2.video_configuration.controls.FrameRate = fps

video_config    = picam2.create_video_configuration()
picam2.configure(video_config)
#------------------------------------------------------------------------
# ARGUMENTS
parser          = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--record', type=str, choices=['on', 'off'], default='off', help='Record video output (on/off)')
parser.add_argument('--picture_move', type=str, choices=['on', 'off'], default='off', help='Enregistrer la détection de boule (on/off)')
parser.add_argument('--stop_on_detect', type=str, choices=['on', 'off'], default='off', help='Stopper dès la première détection (on/off)')
parser.add_argument('--nbframes', type=int, default=1000, help='Number of frames to record')
args            = parser.parse_args()
#------------------------------------------------------------------------
# ZONES 
# Frame size 1280x720:
#+------------------------------------------------------+
#|                                                      |
#|                                                      |
#|                                                      |
#|   +-------------------------+    +------------------+
#|   |        pinsL            |    |      pinsR       |
#|   +-------------------------+    +------------------+
#|                                                      |
#|   +-------------------------+    +------------------+
#|   |        lineL            |    |      lineR       |
#|   +-------------------------+    +------------------+
#+------------------------------------------------------+
# Lines zones
x1_lineL        = data['zones']['line_left']["x1"]
x2_lineL        = data['zones']['line_left']["x2"]
x1_lineR        = data['zones']['line_right']["x1"]
x2_lineR        = data['zones']['line_right']["x2"]
y1_line         = data['zones']['line']["y1"]
y2_line         = data['zones']['line']["y2"]

# Pins zones
x1_pinsL        = data['zones']['pins_left']["x1"]
x2_pinsL        = data['zones']['pins_left']["x2"]

x1_pinsR        = data['zones']['pins_right']["x1"]
x2_pinsR        = data['zones']['pins_right']["x2"]
y1_pins         = data['zones']['pins']["y1"]
y2_pins         = data['zones']['pins']["y2"]

bg_lineL        = None
bg_lineR        = None

stopFrame       = False
#------------------------------------------------------------------------
# MOTION DETECTION
threshold       = data["detect_move"]["threshold"]
contourArea     = data["detect_move"]["contour_area"]
output_main     = data["output"]["folder_path"] + data["output"]["video"]["main_name"]
output_pins     = data["output"]["folder_path"] + data["output"]["video"]["pins_name"]
output_lineL    = data["output"]["folder_path"] + data["output"]["video"]["lineL_name"]
output_lineR    = data["output"]["folder_path"] + data["output"]["video"]["lineR_name"]

motion_nbr_frame_to_skip    = data['detect_move']['nbr_frame_to_skip']
nbr_frame_before_inference  = data['detect_move']['nbr_frame_before_inference']
motion_detected = { "lineL": False, 
                    "lineL_frame_detected": 0,
                    "lineL_frame_inference": False,
                    "lineR": False, 
                    "lineR_frame_detected": 0, 
                    "lineR_frame_inference": False}
#------------------------------------------------------------------------
# FUNCTIONS
#------------------------------------------------------------------------
# EXECUTE FRAMES
picam2.start()
#------------------------------------------------------------------------
# INIT OUT IF ASKED
if args.record == "on":
    fourcc      = cv2.VideoWriter_fourcc(*'XVID')
    #out_pins    = cv2.VideoWriter(output_pins, fourcc, fps, (300,300))
    out_pins    = cv2.VideoWriter(output_pins, fourcc, fps, (600,600))
    #out_main    = cv2.VideoWriter(output_main, fourcc, fps, (1280,720))
    #out_lineL   = cv2.VideoWriter(output_lineL, fourcc, fps, (300,300))
    #out_lineR   = cv2.VideoWriter(output_lineR, fourcc, fps, (300,300))
    
   # out     = cv2.VideoWriter(output_video, fourcc, fps, (1280,720))


while True: # CAMERA CAPTURE only

    #------------------------------------------------------------------------
    # COUNTER    
    if counter%frame_interval != 0: 
        counter+=frame_increment
        continue
    #------------------------------------------------------------------------
    # COUNTER
    #success, frame = cap.read()  # VIDEO RECORD only
    
    #if success and counter%10 == 0: # VIDEO RECORD only
    #if counter%10 == 0: # CAMERA CAPTURE only
    
    print("counter : {}".format(str(counter)))
    #------------------------------------------------------------------------
    # FRAME
    # Capture frame from the camera
    frame               = picam2.capture_array("main")
    #------------------------------------------------------------------------
    # Zones creation
    
    # lineL, lineR zones
    lineL               = frame[y1_line:y2_line, x1_lineL:x2_lineL]
    lineR               = frame[y1_line:y2_line, x1_lineR:x2_lineR]
    
    # Convert in BGR  for OpenCV record
    lineL_bgr           = cv2.cvtColor(lineL, cv2.COLOR_RGB2BGR)
    lineR_bgr           = cv2.cvtColor(lineR, cv2.COLOR_RGB2BGR)
    
    
    # pinsL, pinsR zones
    pinsL               = frame[y1_pins:y2_pins, x1_pinsL:x2_pinsL]
    pinsR               = frame[y1_pins:y2_pins, x1_pinsR:x2_pinsR]
    pinsL_bgr           = cv2.cvtColor(pinsL, cv2.COLOR_RGB2BGR)
    pinsR_bgr           = cv2.cvtColor(pinsR, cv2.COLOR_RGB2BGR)

    if counter == 0: 
        bg_lineL        = lineL_bgr
        bg_lineR        = lineR_bgr
    #------------------------------------------------------------------------
    # DETECT MOVE IN LINEL OR LINER
    zones           = {"lineL": lineL_bgr, "lineR": lineR_bgr}
    backgrounds     = {"lineL": bg_lineL, "lineR": bg_lineR}
    
    for zone_name, zone in zones.items():
        
        # Don't detect if already detected
        if motion_detected[zone_name] : 
            continue

        filename_zone       = "output/1_zone_" + zone_name + ".png"
        filename_diff       = "output/2_diff_" + zone_name + ".png"
        filename_gray       = "output/3_grayscale_" + zone_name + ".png"
        filename_blur       = "output/4_blurred_" + zone_name + ".png"
        filename_thresh     = "output/5_thresholded_" + zone_name + ".png"
        filename_contours   = "output/6_contours_" + zone_name + ".png"
        # DEBUG, FOCUS ON ONE SPECIFIC ZONE
        #if zone_name != "lineL":
        #    continue
        

        diff        = cv2.absdiff(zone, backgrounds[zone_name])  # Différence avec l'arrière-plan        
        gray        = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)    # Conversion en niveaux de gris
        blur        = cv2.GaussianBlur(gray, (5, 5), 0)         # Flou pour réduire le bruit
        _, thresh   = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)  # Seuil pour obtenir une image binaire
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Trouver les contours
        contour_image = diff.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        # Write videos
        if args.record == "on":
            #out_lineL.write(diff)
            out_pins.write(pinsL_bgr)

        if args.picture_move == "on":
            cv2.imwrite("output/output_pinsL.png",pinsR_bgr)

        for contour in contours:
            # Check if contourArea is enough big
            if cv2.contourArea(contour) > contourArea:

                # Check if there is not already detected this line
                if motion_detected[zone_name] == False:

                    # Update move detection status
                    motion_detected[zone_name] = True
                    print("Move detected on: {}".format(zone_name))

                    # Save frame number of frame detection
                    motion_detected[zone_name+"_frame_detected"] = counter

                    if args.picture_move == "on":
                        cv2.imwrite(filename_zone,zone)
                        cv2.imwrite(filename_diff,diff)
                        cv2.imwrite(filename_gray,gray)
                        cv2.imwrite(filename_blur,blur)
                        cv2.imwrite(filename_thresh,thresh)
                        cv2.imwrite(filename_contours, contour_image)
                        
                    if args.stop_on_detect == "on":
                        stopFrame   = True
                    break

        # Update last background
        backgrounds[zone_name] = zone

    # Move detected and asked to stop
    if stopFrame == True :
        break
 
    #------------------------------------------------------------------------   
    # INFERENCE
    if motion_detected["lineL"]:

        # Check when inference has to be make
        if counter >= (motion_detected["lineL_frame_detected"] + nbr_frame_before_inference):
            if motion_detected["lineL_frame_inference"] == False:
                motion_detected["lineL_frame_inference"] = True
                print("Inference on pinsL")

        # Check when detection is finished
        if counter >= (motion_detected["lineL_frame_detected"] + motion_nbr_frame_to_skip):
             # Make inference on pinsL
            print("End of detection action")
            motion_detected["lineL"] = False
    #------------------------------------------------------------------------
    if motion_detected["lineR"]:

        # Check when inference has to be make
        if counter >= (motion_detected["lineR_frame_detected"] + nbr_frame_before_inference):
            if motion_detected["lineR_frame_inference"] == False:
                motion_detected["lineR_frame_inference"] = True
                print("Inference on pinsR")

        # Check when detection is finished
        if counter >= (motion_detected["lineR_frame_detected"] + motion_nbr_frame_to_skip):
             # Make inference on pinsR
            print("End of detection action")
            motion_detected["lineR"] = False      
    #------------------------------------------------------------------------
    # RECORDS
    # Record necessary
    if args.record == "on":
        #out_main.write(frame)
        #out_lineL.write(lineL_bgr)
        #out_lineR.write(lineR_bgr)
        out_pins.write(frame)
    #------------------------------------------------------------------------
    # COUNTER AND NEXT LOOP
    if counter > int(args.nbframes):
        break
    counter+=frame_increment

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#------------------------------------------------------------------------
# Releases resources
if args.record == "on":
    #out_main.release()
    #out_lineL.release()
    #out_lineR.release()
    out_pins.release()
#------------------------------------------------------------------------
# END
cv2.destroyAllWindows()