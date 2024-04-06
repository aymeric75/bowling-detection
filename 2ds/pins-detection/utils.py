import cv2
import numpy as np



def lines_and_pins_zones(frame, y1_line, y2_line, x1_lineL, x2_lineL, \
                         x1_lineR, x2_lineR, y1_pins, y2_pins, x1_pinsL, \
                        x2_pinsL, x1_pinsR, x2_pinsR
                   ):

    # lineL, lineR zones
    lineL               = frame[y1_line:y2_line, x1_lineL:x2_lineL]
    lineR               = frame[y1_line:y2_line, x1_lineR:x2_lineR]
    
    # pinsL, pinsR zones
    pinsL               = frame[y1_pins:y2_pins, x1_pinsL:x2_pinsL]
    pinsR               = frame[y1_pins:y2_pins, x1_pinsR:x2_pinsR]
   

    return lineL, lineR, pinsL, pinsR

### Define function for inferencing with TFLite model and displaying results

def detect_objects_from_image(interpreter, interpreter_infos, labels, frame, min_conf=0.5):

    output_details = interpreter_infos['output_details']
    input_details = interpreter_infos['input_details']
    height = interpreter_infos['height']
    width = interpreter_infos['width']
    float_input = interpreter_infos['float_input']
    input_mean = 127.5
    input_std = 127.5

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

    detections = []

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

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

    # test1 Aymeric
    #print("detected {} pins : ".format(len(detections)))


    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # plt.figure(figsize=(12,16))
    # plt.imshow(image)
    # plt.savefig("finalresult.png") 
    # #plt.show()

    return image, detections

 #------------------------------------------------------------------------

# Détection d'un objet / mouvement dans une zone spécifique
# Retourne la zone trouvée
def detect_move(config,last_frame,frame):
    if not np.any(last_frame): # if last_frame is empty return False
        return False
    diff        = cv2.absdiff(frame, last_frame)            # Différence avec l'arrière-plan        
    gray        = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)    # Conversion en niveaux de gris
    blur        = cv2.GaussianBlur(gray, (5, 5), 0)         # Flou pour réduire le bruit

    # Seuil pour obtenir une image binaire
    _, thresh   = cv2.threshold(blur, config['threshold'], 255, cv2.THRESH_BINARY)  

    # Trouver les contours => contour = une zone pixellisé étrangère => un objet détecté
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    contour_image = diff.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Parcours les contours trouvés 
    for contour in contours:
        # Check if contourArea is enough big
        if cv2.contourArea(contour) > config['contour_area']:
            return True    
        
    return False
  