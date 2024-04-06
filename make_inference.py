

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



# Draw objets
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