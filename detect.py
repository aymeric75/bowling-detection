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

import numpy as np

import cv2
import argparse
#------------------------------------------------------------------------
# VARIABLES
# Définir le taux de trames
fps             = 30
count           = 1 # number of inferences to perform (see the loop in make_inference def)
threshold       = 0.4
output_video    = "output/output2.avi"
counter         = 0
variance_blurry = 10
api_token       = "SmUjiWZfS7t9zVvmGO84jKtxvbrIG9"
#------------------------------------------------------------------------
# CONFIG CAMERA
picam2          = Picamera2()
picam2.video_configuration.controls.FrameRate = fps

video_config    = picam2.create_video_configuration()
picam2.configure(video_config)
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
# MODEL
model_path           = "model/training_1/detect.tflite"
labels_file     = "model/training_1/labelmap.txt"
labels          = read_label_file(labels_file) if labels_file else {}
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()


# USED IN THE tflite_detect_images
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
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






################# TESTS ################################



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

### Define function for inferencing with TFLite model and displaying results

def tflite_detect_images(modelpath, frame, lblpath, min_conf=0.5):


  # Randomly select test images
  #images_to_test = random.sample(images, num_test_images)

  # Loop over every image and perform detection

  # Load image and resize to expected shape [1xHxWx3]
  #image = cv2.imread(frame)
  image = frame
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  imH, imW, _ = image.shape
  image_resized = cv2.resize(image_rgb, (width, height))
  input_data = np.expand_dims(image_resized, axis=0)

  # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
  if float_input:
      input_data = (np.float32(input_data) - input_mean) / input_std

  # Perform the actual detection by running the model with the image as input
  interpreter.set_tensor(input_details[0]['index'],input_data)
  interpreter.invoke()

  # Retrieve detection results
  boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
  classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
  scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
 
  #detections = []

  # Loop over all detections and draw detection box if confidence is above minimum threshold
  for i in range(len(scores)):
      if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

          # Get bounding box coordinates and draw box
          # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
          ymin = int(max(1,(boxes[i][0] * imH)))
          xmin = int(max(1,(boxes[i][1] * imW)))
          ymax = int(min(imH,(boxes[i][2] * imH)))
          xmax = int(min(imW,(boxes[i][3] * imW)))



          cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

          # Draw label
          object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
          label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
          labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
          label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
          cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
          cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

          #detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

  #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

  # plt.figure(figsize=(12,16))
  # plt.imshow(image)
  # plt.savefig("finalresult.png") 
  # #plt.show()

  return image
  


















################################## TESTS END #################################################























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

# Dra objets
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




# Make Inference
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

  #current_time = datetime.datetime.now()
  #formatted_time = current_time.strftime("%H:%M:%S.%f")[:-3]
  #print('  T :    ', formatted_time)

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

  count_topPins = sum(1 for obj in objs if obj.id == 0)

  width_center = 1280 // 2


  left_bbox_list = [bbox for bbox in objs if is_valid_bbox(bbox) and bbox.xmax <= width_center]

  right_bbox_list = [bbox for bbox in objs if is_valid_bbox(bbox) and bbox.xmax > width_center]


  #print(len(left_bbox_list))


  print("count topPins : {}".format(str(count_topPins)))

  # Appels API

  image = image.convert('RGB')
  draw_objects(ImageDraw.Draw(image), objs, labels)

  return image
#------------------------------------------------------------------------
# EXECUTE FRAMES
picam2.start()

if args.record == "on":
    fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    out     = cv2.VideoWriter(output_video, fourcc, fps, (1280,720)) # Assuming frame size is 640x480. Adjust accordingly.


#cap = cv2.VideoCapture('output/output1.avi') # VIDEO RECORD only
#while cap.isOpened(): # VIDEO RECORD only
while True: # CAMERA CAPTURE only



    #------------------------------------------------------------------------
    # COUNTER
    #success, frame = cap.read()  # VIDEO RECORD only
    
    #if success and counter%10 == 0 and counter > 0: # VIDEO RECORD only
    if counter%2 == 0: # CAMERA CAPTURE only
      
      print("counter : {}".format(str(counter)))
      #------------------------------------------------------------------------
      # FRAME
      # frame from recorded video 
      array       = picam2.capture_array("main") # uncomment when using camera (CAMERA CAPTURE only)

      #------------------------------------------------------------------------
      # Check blurry
      variance    = is_blurry(array)
      
      if variance < variance_blurry:
          print("L'image est floue")
          
          if args.allow_blurry == "off":
              continue

      #------------------------------------------------------------------------
      # INFERENCE
      #print(array.shape) # (720, 1280, 4)

      
      # print(frame.shape) # numpy array, <720 1280 3>
     
      # Image avec les bounding boxes
      if args.inference == "on":
        processed_image = tflite_detect_images(model_path, array, labels_file, min_conf=0.5)
      else:
        processed_image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
        processed_image = processed_image.convert('RGB')

      
      

      # reformat de l'image pour OpenCV
      opencv_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)

      # Montrer l'image sur écran (à utiliser uniquement si écran directement branché sur le raspberry)
      #cv2.imshow('Processed image', opencv_image)
    
      #écrit l'image actuelle dans le répertoire courant
      cv2.imwrite("current_img.png", opencv_image)

      # envoi de la requête POST
      #send_post_request(api_token, "current_img.png")

      # # enregistrement de la vidéo
      if args.record == "on":
        out.write(opencv_image)

      #if counter>1000:
      if counter > int(args.nbframes):
          break


      # Break the loop if 'q' is pressed
      if cv2.waitKey(1) & 0xFF == ord("q"):
          break

    counter+=1

if args.record == "on":
    out.release()
#------------------------------------------------------------------------
# END
cv2.destroyAllWindows()