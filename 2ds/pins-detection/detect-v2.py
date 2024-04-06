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
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

import requests



from utils.utils import detect_objects_from_image, lines_and_pins_zones

import numpy as np

import cv2
import argparse
#------------------------------------------------------------------------
# VARIABLES
import json
with open('../config/config.json', 'r') as f:
    config      = json.load(f)

# Définir le taux de frames
fps             = 30
output_video    = "output/output2.avi"
counter         = 0
variance_blurry = 10
api_token       = config['api_token']    

bg_lineL        = None
bg_lineR        = None
#------------------------------------------------------------------------
# CONFIG CAMERA
# picam2          = Picamera2()
# picam2.video_configuration.controls.FrameRate = fps

# video_config    = picam2.create_video_configuration()
# picam2.configure(video_config)
#print(f"Actual frame rate: {actualFps} fps")
#------------------------------------------------------------------------
# ARGUMENTS
parser          = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--record', type=str, choices=['on', 'off'], default='off', help='Record video output (on/off)')
parser.add_argument('--inference', type=str, choices=['on', 'off'], default='on', help='Make inference (on/off)')
parser.add_argument('--allow_blurry', type=str, choices=['on', 'off'], default='on', help='Allow blurry (on/off)')
parser.add_argument('--nbframes', type=int, default=1000, help='nber of frames to record')
args            = parser.parse_args()


#------------------------------------------------------------------------
# MODEL & declaration of the Interpreter
model_path          = "model/training_3/detect.tflite"
labels_file         = "model/training_3/labelmap.txt"
interpreter         = make_interpreter(model_path)
interpreter.allocate_tensors()
interpreter_infos                   = {}
interpreter_infos["input_details"]  = interpreter.get_input_details()
interpreter_infos["output_details"] = interpreter.get_output_details()
interpreter_infos["height"]         = interpreter_infos["input_details"][0]['shape'][1]
interpreter_infos["width"]          = interpreter_infos["input_details"][0]['shape'][2]
interpreter_infos["float_input"]    = (interpreter_infos["input_details"][0]['dtype'] == np.float32)


# labels
with open(labels_file, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


#------------------------------------------------------------------------
# FUNCTIONS
# Check if image is blurry
def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = np.var(laplacian)
    return variance


# Import packages
import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
#from tensorflow.lite.python.interpreter import Interpreter

import matplotlib
import matplotlib.pyplot as plt






# Function to get the MAC address
def get_mac_address():
    # get the MAC address of the default interface (usually eth0 for wired, wlan0 for wireless)
    try:
        mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 2 * 6, 2)][::-1])
        print(mac)
        return mac
    except Exception as e:
        print(f"An error occurred while fetching MAC Address: {e}")
        return None

# Sending a POST request
def send_post_request(api_token, image_path):

    # Endpoint and URL details
    url = "https://nuasix.hibowl.com/rest/api.php/hibowlset/shot/save"

    # Body data
    data = {
        'api_token': api_token,
        'mac_address': get_mac_address(),
        'lane' :  'left',
        'nbr_pins_up' : 10,
    }

    # Prepare the image for the POST request
    with open(image_path, 'rb') as f:
      files = {'img_frame': f}
        
      # Sending the POST request
      response = requests.post(url, data=data, files=files)
    
    # Printing the response (optional)
    print(response.text)
    return response




#------------------------------------------------------------------------
# Connect to the cam (or start the video) and start the Loop


# picam2.start()

if args.record == "on":
    fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    #out     = cv2.VideoWriter(output_video, fourcc, fps, (1280,720)) # Assuming frame size is 640x480. Adjust accordingly.
    out     = cv2.VideoWriter(output_video, fourcc, fps, (600,600)) # Assuming frame size is 640x480. Adjust accordingly.


#while True: # CAMERA CAPTURE only
cap = cv2.VideoCapture('output/output1.avi') # VIDEO RECORD only
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)
# 


boule_just_detected         = False # say if boule was just detected
frames_btween_bowl_inf      = 0 # counter for nber of frames btween boule and inference
frames_after_inf            = 9999 # counter for nber of frames after the inference was made (init to 9999 for logic purpose see below)
inference_just_done         = False # boolean that say if inference was just done



images_with_inf             = [] # openCV bounding boxed images
nbreQuille_for_img          = [] # nberQuille for each of the image 
counter_inferences          = 0



while cap.isOpened(): # VIDEO RECORD only


    #------------------------------------------------------------------------
    # COUNTER
    success, frame = cap.read()  # VIDEO RECORD only
    
    if counter%50 == 0:
        print("counter : {}".format(str(counter)))

    #if counter%2 == 0: # CAMERA CAPTURE only
    if success: # VIDEO RECORD only

        #------------------------------------------------------------------------
        # FRAME (from recorded video )
        img                             = frame

        #------------------------------------------------------------------------
        # when INFERENCE allowed , DO 5 consecutive INFERENCES, and take the max

        while counter_inferences < 5 and counter%10 == 0:

            lineL, lineR, pinsL, pinsR  = lines_and_pins_zones(config, img)
            pinsL_bgr                   = cv2.cvtColor(pinsL, cv2.COLOR_RGB2BGR)

            processed_image, detections = detect_objects_from_image(interpreter, interpreter_infos, labels, pinsL_bgr, min_conf=0.2)

            images_with_inf.append(processed_image)
            
            nbreQuille_for_img.append(len(detections))

            counter_inferences+=1


        if(len(nbreQuille_for_img) == 5):

            index_of_max_quilles = np.argmax(np.array(nbreQuille_for_img))

            print("detected {} pins ".format(str(nbreQuille_for_img[index_of_max_quilles])))
            image_with_bounding_boxes = images_with_inf[index_of_max_quilles]

            opencv_image = cv2.cvtColor(np.array(image_with_bounding_boxes), cv2.COLOR_RGB2BGR)

            # print(type(opencv_image)) # <class 'numpy.ndarray'>
            # print(opencv_image.shape) #  (600, 600, 3)

            # print(type(array))# <class 'numpy.ndarray'>
            # print(array.shape)# (720, 1280, 3)
            
            if args.record == "on":
                out.write(opencv_image)
                #out.write(array)

            nbreQuille_for_img=[]
            image_with_bounding_boxes=[]
            counter_inferences=0


        
        # continue

        # if (frames_after_inf > config['nbr_frame_to_skip'] and boule_just_detected  == False):
        #     print("waiting for BOULE")
        #     inference_just_done=False
        #     frames_btween_bowl_inf = 0
            
        #     boule_just_detected = random.choice([True, False, False, False, False]) # FUNCTION CALL FOR BOULE DETECTION !!

        # if boule_just_detected:
        #     print("Boule JUST detected")
        #     frames_btween_bowl_inf+=1
        #     frames_after_inf = 0


        # if frames_btween_bowl_inf > config['nbr_frame_before_inference']:
        #     boule_just_detected = False
        #     if not inference_just_done:

        #         print("INFERENCE NOW")

        #         #------------------------------------------------------------------------
        #         # Check blurry
        #         #------------------------------------------------------------------------
        #         variance    = is_blurry(array)
                
        #         if variance < variance_blurry:
        #             print("L'image est floue")
        #             if args.allow_blurry == "off":
        #                 continue


        #         #------------------------------------------------------------------------
        #         # RETRIEVE THE ZONES for inference
        #         #------------------------------------------------------------------------
                
        #         lineL, lineR, pinsL, pinsR = lines_and_pins_zones(array,  y1_line, y2_line, x1_lineL, x2_lineL, \
        #                             x1_lineR, x2_lineR, y1_pins, y2_pins, x1_pinsL, \
        #                         x2_pinsL, x1_pinsR, x2_pinsR)

        #         pinsL_bgr           = cv2.cvtColor(pinsL, cv2.COLOR_RGB2BGR)
                
        #         # print(frame.shape) # numpy array, <720 1280 3>
                
                
        #         if args.inference == "on":
        #             # Image avec les bounding boxes
        #             #------------------------------------------------------------------------
        #             # ACTUAL INFERENCE
        #             processed_image = detect_objects_from_image(interpreter, interpreter_infos, labels, pinsL_bgr, min_conf=0.5)
        #         else:
        #             # Image SANS les bounding boxes
        #             processed_image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
        #             processed_image = processed_image.convert('RGB')
                    

        #         # reformat de l'image pour OpenCV
        #         opencv_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)

        #         # Montrer l'image sur écran (à utiliser uniquement si écran directement branché sur le raspberry)
        #         #cv2.imshow('Processed image', opencv_image)

        #         #écrit l'image actuelle dans le répertoire courant
        #         cv2.imwrite("current_img.png", opencv_image)

        #         # envoi de la requête POST
        #         #send_post_request(api_token, "current_img.png")

        #         # # enregistrement de la vidéo
        #         if args.record == "on":
        #             out.write(opencv_image)





        #         inference_just_done = True

        #     print("Skipping FRAMES after INFERENCE")
        #     frames_after_inf+=1


    # Stop tout, après X nombre de frames
    #if counter>1000:
    if counter > int(args.nbframes):
        break


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    counter+=1

if args.record == "on":
    print("WENT HERE")
    out.release()
#------------------------------------------------------------------------
# END
cv2.destroyAllWindows()