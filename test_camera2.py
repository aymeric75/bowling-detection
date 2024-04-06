#!/usr/bin/python3
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

import time

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

import numpy as np


import cv2

import argparse

import requests
import uuid
import socket


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



picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

fps = picam2.framerate
print(f"Frame rate: {fps} fps")



parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--record', type=str, choices=['on', 'off'], default='off', help='Record video output (on/off)')
parser.add_argument('--nbframes', type=int, default=1000, help='nber of frames to record')
args = parser.parse_args()


#model="../test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
model="./detect.tflite"
labels_file="../test_data/coco_labels.txt"
theinput="../test_data/grace_hopper.bmp"
output="/home/pi/Desktop/pycoral/examples/grace_hopper_processed.bmp"
count=1 # number of inferences to perform (see the loop in make_inference def)
threshold=0.4




def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')

# check if the box object is a valid one (i.e. contains xmin etc)
def is_valid_bbox(obj):
    """Check if the object is a valid BBox."""
    return all(hasattr(obj, attr) for attr in ['xmin', 'ymin', 'xmax', 'ymax'])


def make_inference(frame_nb, frame, interpreter):

  # instanciate the frame as an Image object
  image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 
  # resize the image and zero-pad it to fit into the model's input tensor
  _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

  print('----INFERENCE TIME----')
  # print('Note: The first inference is slow because it includes',
  #       'loading the model into Edge TPU memory.')
  for _ in range(count):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    objs = detect.get_objects(interpreter, threshold, scale)
    #print('%.2f ms' % (inference_time * 1000))

  print('-------RESULTS--------')
  if not objs:
    print('No objects detected')

  ## affichage des objets trouvés
  for obj in objs:
    print(labels.get(obj.id, obj.id))
    print('  id:    ', obj.id)
    print('  score: ', obj.score)
    print('  bbox:  ', obj.bbox)


  print("IMAGE ")
  print(image.size) # height , width

  # x in 0-1280
  # y in 0-720

  # count all boxes with a specific id
  # objs = list of objects : [ Object(id: 0, score: 0.7, ..), {id: 0, score: 0.89, ..} ETC]
  # numpy array of dicts, count 

  count_persons = sum(1 for obj in objs if obj.id == 0)

  width_center = 1280 // 2


  left_bbox_list = [bbox for bbox in objs if is_valid_bbox(bbox) and bbox.xmax <= width_center]

  right_bbox_list = [bbox for bbox in objs if is_valid_bbox(bbox) and bbox.xmax > width_center]


  #print(len(left_bbox_list))


  print("count persons : {}".format(str(count_persons)))

  # Appels API

  image = image.convert('RGB')
  draw_objects(ImageDraw.Draw(image), objs, labels)

  return image



labels = read_label_file(labels_file) if labels_file else {}
interpreter = make_interpreter(model)
interpreter.allocate_tensors()

picam2.start()


counter=0

if args.record == "on":
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (1280,720)) # Assuming frame size is 640x480. Adjust accordingly.



#cap = cv2.VideoCapture('Bowling1-CUT.mp4') # VIDEO RECORD only

#while cap.isOpened(): # VIDEO RECORD only
while True: # CAMERA CAPTURE only


    #success, frame = cap.read()  # VIDEO RECORD only
    
    #if success and counter%10 == 0: # VIDEO RECORD only
    if counter%10 == 0: # CAMERA CAPTURE only
    
      print("counter : {}".format(str(counter)))

      array = picam2.capture_array("main") # uncomment when using camera
      #array = frame # array from recorded video

      #print(array.shape) # (720, 1280, 4)

      # Image avec les bounding boxes
      processed_image  = make_inference(counter, array, interpreter)

      # reformat de l'image pour OpenCV
      opencv_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)

      # Montrer l'image sur écran (à utiliser uniquement si écran directement branché sur le raspberry)
      #cv2.imshow('Processed image', opencv_image)
    
      # écrit l'image actuelle dans le répertoire courant
      cv2.imwrite("current_img.png", opencv_image)

      # envoi de la requête POST
      #send_post_request("SmUjiWZfS7t9zVvmGO84jKtxvbrIG9", "current_img.png")

      # enregistrement de la vidéo
      if args.record == "on":
          out.write(opencv_image)

    #if counter>1000:
    if counter > int(args.nbframes):
        break

    counter+=1


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if args.record == "on":
    out.release()
cv2.destroyAllWindows()
