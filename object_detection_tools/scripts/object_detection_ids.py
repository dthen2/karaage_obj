# coding: utf-8
# 日本語
# デフォルトの読み込みファイルを設定し直した
# 一部のインデントを直したら・・・
# モザイク機能は削除
# Object Detection Demo
import argparse
import cv2
import numpy as np
import os
import sys
import time
import tensorflow as tf

# 深田のＩＤネットワーク
import id_network as ids
import copy

from distutils.version import StrictVersion

try:
  if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
except:
  pass

# Path to label and frozen detection graph. This is the actual model that is used for the object detection.
parser = argparse.ArgumentParser(description='object detection tester, using webcam or movie file')
parser.add_argument('-l', '--labels', default='coco-labels-paper.txt', help="default: 'coco-labels-paper.txt'")
parser.add_argument('-m', '--model',  default='frozen_inference_graph.pb', help="default: 'frozen_inference_graph.pb'")
parser.add_argument('-d', '--device', default='normal_cam', help="normal_cam, jetson_nano_raspi_cam, jetson_nano_web_cam, raspi_cam, or video_file. default: 'normal_cam'") # normal_cam / jetson_nano_raspi_cam / jetson_nano_web_cam
parser.add_argument('-i', '--input_video_file', default='', help="Input video file")

args = parser.parse_args()

detection_graph = tf.Graph()

colors = [
  (0, 0, 255),
  (0, 64, 255),
  (0, 128, 255),
  (0, 192, 255),
  (0, 255, 255),
  (0, 255, 192),
  (0, 255, 128),
  (0, 255, 64),
  (0, 255, 0),
  (64, 255, 0),
  (128, 255, 0),
  (192, 255, 0),
  (255, 255, 0),
  (255, 192, 0),
  (255, 128, 0),
  (255, 64, 0),
  (255, 0, 0),
  (255, 0, 64),
  (255, 0, 128),
  (255, 0, 192),
  (255, 0, 255),
  (192, 0, 255),
  (128, 0, 255),
  (64, 0, 255),
]

def load_graph():
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args.model, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# Load a (frozen) Tensorflow model into memory.
print('Loading graph...')
detection_graph = load_graph()
print('Graph is loaded')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with detection_graph.as_default():
  tf_sess = tf.Session(config = tf_config)
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)

  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

def run_inference_for_single_image(image, graph):
  # Run inference
  output_dict = tf_sess.run(tensor_dict,
                          feed_dict={image_tensor: image})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.int64)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  return output_dict

if args.input_video_file != "":
  # WORKAROUND
  print("[Info]hoge --input_video_file has an argument. so --device was replaced to 'video_file'.")
  args.device = "video_file"

# Switch camera according to device
if args.device == 'normal_cam':
  cam = cv2.VideoCapture(0)
elif args.device == 'jetson_nano_raspi_cam':
  GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx \
    ! videoconvert \
    ! appsink drop=true sync=false'
  cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER) # Raspi cam
elif args.device == 'jetson_nano_web_cam':
  cam = cv2.VideoCapture(1)
elif args.device == 'raspi_cam':
  from picamera.array import PiRGBArray
  from picamera import PiCamera
  cam = PiCamera()
  cam.resolution = (640, 480)
  stream = PiRGBArray(cam)
elif args.device == 'video_file':
  cam = cv2.VideoCapture(args.input_video_file)
else:
  print('[Error] --device: wrong device')
  parser.print_help()
  sys.exit()

# Moved from main() #
def scan_detected_object(img, output_dict, score_sh):
    object_list_new = []
    for i in range(output_dict['num_detections']):
        class_id = output_dict['detection_classes'][i]
        if class_id < len(labels):
          label = labels[class_id]
        else:
          label = 'unknown'
        cid = class_id % len(colors) # % は剰余演算子
        color = colors[cid]

        detection_score = output_dict['detection_scores'][i]
        if detection_score > score_sh:
          # Define bounding box
          h, w, c = img.shape
          box = (output_dict['detection_boxes'][i] * np.array([h, w,  h, w])).astype(np.int)

          # 追加部分、各物体の位置とサイズ
          x_object = (box[1]+box[3])/2.0
          y_object = (box[2]+box[0])/2.0
          size_x_object = box[3] - box[1]
          size_y_object = box[2] - box[0]
          object_list_new.append({"x": x_object, "y": y_object, "size_x": size_x_object, "size_y": size_y_object,\
            "label":label, "box": box, "detection_score": detection_score, "color": color})
    return object_list_new

# Moved from main() #
def put_bounding_box(img, object_list_new):
    for obj in object_list_new:
        box   =    obj["box"]
        label =    obj["label"]
        objID =    obj["ID"]
        color =    obj["color"]
        detection_score = obj["detection_score"]
        fonsScale = 0.5

        # Draw bounding box
        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), color, 3)

        # Put label near bounding box
        information = 'ID:%d %s: %.1f%%' % (objID, label, detection_score * 100.0)
        cv2.putText(img, information, (box[1] + 8, box[2] - 8), \
          cv2.FONT_HERSHEY_SIMPLEX, fonsScale, color, 1, cv2.LINE_AA)

#### defined by Fukada ######
def give_IDs(idnet, projects, object_list_new, object_list_previous):
  for item in object_list_new:
    x = item["x"]
    y = item["y"]
    min_size = 10
    acceptable_d2 = max(min_size, max(item["size_x"], item["size_y"]))**2
    d_posi_min_2 = 100000000000000000.0
    infID = 0
    for prv_item in object_list_previous:
      x_prv = prv_item["x"]
      y_prv = prv_item["y"]
      dx = x - x_prv
      dy = y - y_prv
      d_position_2 = dx**2 + dy**2
      if d_position_2 < d_posi_min_2:
        d_posi_min_2 = d_position_2
        infID = prv_item["ID"]
    if infID != 0 and d_posi_min_2 < acceptable_d2:
      item["ID"] = infID
    else:
      obj = copy.copy(projects)
      item["ID"] = idnet.register_ID(obj)
  return object_list_new


if __name__ == '__main__':
  idnet = ids.ID_NetWork()
  project_ID = idnet.register_ID({"data_type": "project"}) # 1
  app_ID =     idnet.register_ID({"data_type": "app"})     # 2
  work_ID =    idnet.register_ID({"data_type": "work"})    # 3
  projects = {"data_type": "point", "project_ID": project_ID, "app_ID": app_ID, "work_ID": work_ID}

  score_sh = 0.5
  labels = ['blank']
  with open(args.labels,'r') as f:
    for line in f:
      labels.append(line.rstrip())

  object_list_previous = [] # 追加部分
  while True:
    if args.device == 'raspi_cam':
      cam.capture(stream, 'bgr', use_video_port=True)
      img = stream.array
    else:
      ret, img = cam.read()
      if not ret:
        print('error')
        break

    h, w, c = img.shape
    key = cv2.waitKey(1)
    if key == 27: # when ESC key is pressed break
        break

    img_bgr = cv2.resize(img, (300, 300))
    # convert bgr to rgb
    image_np = img_bgr[:,:,::-1]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    start = time.time()
    # コアのobject_detection はここ
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    elapsed_time = time.time() - start

    # それぞれの特性取得
    object_list_new = scan_detected_object(img, output_dict, score_sh)

    # 追加部分： ＩＤ付与
    object_list_previous = give_IDs(idnet, projects, object_list_new, object_list_previous)

    put_bounding_box(img, object_list_new)
    speed_info = '%s: %.3f' % ('fps', 1.0/elapsed_time)
    cv2.putText(img, speed_info, (10,50), \
      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('detection result', img)
    # while True ループここまで

    if args.device == 'raspi_cam':
      stream.seek(0)
      stream.truncate()

  tf_sess.close()
  cam.release()
  cv2.destroyAllWindows()
