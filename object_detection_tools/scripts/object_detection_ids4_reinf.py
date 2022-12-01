# coding: utf-8
# 日本語
# デフォルトの読み込みファイルを設定し直した
# 一部のインデントを直したら・・・
# モザイク機能は削除
# 再infの機能を復活した。2022March29 object_detection_ids4_reinf.py
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

labels = ['blank']

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

###################################################
# Moved and modified from main() #
def scan_detected_object(img, output_dict, time_now, score_sh):
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
            box = (output_dict['detection_boxes'][i] * np.array([h, w,  h, w])).astype(int)

            # 追加部分、各物体の位置とサイズ
            x_object = (box[1]+box[3])/2.0
            y_object = (box[2]+box[0])/2.0
            size_x_object = box[3] - box[1]
            size_y_object = box[2] - box[0]
            object_list_new.append({"x": x_object, "y": y_object, "size_x": size_x_object, "size_y": size_y_object,\
                  "label":label, "box": box, "detection_score": detection_score, "color": color, "time": time_now})
    return object_list_new

def create_bounding_box(box, x, y, size_x, size_y):
    box[0] = int(y - size_y/2.0)
    box[1] = int(x - size_x/2.0)
    box[2] = int(y + size_y/2.0)
    box[3] = int(x + size_x/2.0)
    return box

# Moved and modified from main() #
def put_bounding_box(img, object_list_new, v_scale):
    for obj in object_list_new:
        box = obj["box"]
        # 二度手間だが、box演算を独立させるための処理
        box = create_bounding_box(box, obj['x'], obj['y'], obj['size_x'], obj['size_y'])
        fontScale = 0.5
        if obj.get("Missed", False):
            thickness = 1
        else:
            thickness = 3
        # Draw bounding box
        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), obj["color"], thickness)
        # Put label near bounding box
        information = 'ID:%d %s: %.1f%%' % (obj["ID"], obj["label"], obj["detection_score"] * 100.0)
        cv2.putText(img, information, (box[1] + 8, box[2] - 8), \
            cv2.FONT_HERSHEY_SIMPLEX, fontScale, obj["color"], 1, cv2.LINE_AA)
        onotext = obj.get("onomatope",'')
        if onotext != '':
            cv2.putText(img, onotext, (int((box[1] + box[3])/2-50), int((box[2] + box[0])/2-20)), \
                cv2.FONT_HERSHEY_SIMPLEX, fontScale*2, obj["color"], 1, cv2.LINE_AA)
        
        x_c = int(obj['x'])
        y_c = int(obj['y'])
        x_v = x_c + int(obj.get('Vx',0)*v_scale)
        y_v = y_c + int(obj.get('Vy',0)*v_scale)
        cv2.arrowedLine(img, (x_c, y_c), (x_v, y_v), obj["color"], thickness=thickness)#, lineType=cv2.LINE_AA, shift=0)

###################################################################################
#### defined by Fukada ######

# 2022Mrch29 再inferenceを復活追加
def re_inference_image(unmatches, img, size_mag):
    output_re_inf = {'num_detections': 0, 'detection_boxes': [], 'detection_scores': [], 'detection_classes': []}
    h, w, c = img.shape
    num_img = 0
    objects_missed = []
    for unmatch in unmatches:
        x = unmatch["x"]
        y = unmatch["y"]
        size_x = unmatch["size_x"]
        size_y = unmatch["size_y"]
        sizePLSy = max(0, h*size_x/w - size_y)/2 + 24
        sizePLSx = max(0, w*size_y/w - size_x)/2 + 40
        size = max(unmatch["x"], unmatch["y"])*size_mag
        top    = int(min(h, max(0, y-size/2)))
        bottom = int(min(h, max(0, y+size/2)))
        left   = int(min(w, max(0, x-size/2)))
        right  = int(min(w, max(0, x+size/2)))
        mag_x = (right - left)/w
        mag_y = (bottom - top)/h
        img_new = img[top : bottom, left : right]
        img_bgr = cv2.resize(img_new, (300, 300))
        image_np = img_bgr[:,:,::-1]
        image_np_exp = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np_exp, detection_graph)
        # for bugfix
        img2 = cv2.resize(img_new, (int(w/5),int(w/5)))
        image_name = 'magnified image' + str(num_img)
        cv2.imshow(image_name, img2)
        num_img += 1

        detection_score_max = 0.
        num = 0
        for i in range(output_dict['num_detections']):
            detection_score = output_dict['detection_scores'][i]
            if detection_score > detection_score_max:
                detection_score_max = detection_score
                num = i
                d_box = output_dict['detection_boxes'][i] * np.array([mag_y, mag_x, mag_y, mag_x]) + [top/h, left/w, top/h, left/w]
        if detection_score_max > 0.3 :
            output_re_inf['detection_scores'].append(output_dict['detection_scores'][num])
            output_re_inf['detection_classes'].append(output_dict['detection_classes'][num])
            output_re_inf['detection_boxes'].append(d_box)
            output_re_inf['num_detections'] += 1
        else:
            objects_missed.append(unmatch)
    return output_re_inf, objects_missed


# object_detection において２重に検出することがあるので、除去する
# duplication_scale が大きいほど、除去率が高い（間違って除去する可能性）
def remove_duplication(object_list, duplication_scale):
    duplication = []
    i = 0
    for item_a in object_list:
        acceptable_diff2 = (item_a["size_x"]**2 + item_a["size_y"]**2)*duplication_scale
        j = 0
        for item_s in object_list:
            dx = item_a["x"] - item_s["x"]
            dy = item_a["y"] - item_s["y"]
            dsx = item_a["size_x"] - item_s["size_x"]
            dsy = item_a["size_y"] - item_s["size_y"]
            diff2 = dx**2 + dy**2 + dsx**2 + dsy**2
            if i != j and diff2 < acceptable_diff2 and item_a["detection_score"] < item_s["detection_score"]:
                duplication.append(i)
                break
            j += 1
        i += 1
    for delete in reversed(duplication):
        del object_list[delete]
    return object_list

def return_distance(mutch):
    return(mutch["dist_ratio"])

# ID付与 
# accept_ratio が大きいほど大きな位置ズレを許容
def give_IDs(idnet, projects, object_list_new, object_list_previous, time_now, accept_ratio):
    obj = copy.copy(projects)
    min_size = 10
    objects_missed = []
    mutch_list = []
    p = 0
    for prv_item in object_list_previous:
        i = 0
        for item in object_list_new:
            dx  = item["x"] - prv_item["x"]
            dy  = item["y"] - prv_item["y"]
            dsx = item["size_x"] - prv_item["size_x"]
            dsy = item["size_y"] - prv_item["size_y"]
            distance_2 = dx**2 + dy**2 + dsx**2 + dsy**2
            acceptable_d2 = max(max(min_size, max(    item["size_x"],     item["size_y"]))**2,\
                                max(min_size, max(prv_item["size_x"], prv_item["size_y"]))**2)
            mutch = {"dist_ratio": distance_2/acceptable_d2, 'No_new': i, "No_prev": p}
            mutch_list.append(mutch)
            i += 1
        p += 1
    mutch_list.sort(key = return_distance, reverse = True)
    for mutch in mutch_list:
        if mutch['dist_ratio'] < accept_ratio:
            i = mutch['No_new']
            p = mutch['No_prev']
            if object_list_new[i].get('assained',False) == False \
               and object_list_previous[p].get('Found', False) == False:
                object_list_new[i]['ID'] = object_list_previous[p]['ID']
                object_list_new[i]['assained'] = True
                object_list_previous[p]['Found'] = True
                object_list_previous[p]['Missed'] = False
    for prv_item in object_list_previous:
        if prv_item.get('Found', False) == False: # 今回非検出
            if prv_item.get('Missed', False) == False: # 前回検出
                prv_item['Missed_time'] = time_now # 前回検出で今回非検出なら時刻を更新
            prv_item['Missed'] = True
            objects_missed.append(prv_item)
    for item in object_list_new:
        if item.get('ID', False) == False:
            item['ID'] = idnet.register_ID(obj)
    return object_list_new, objects_missed

# 速度推定
def infer_velocity(object_list_new, object_list_previous, last_time, time_now, filt_time):
    for prv_item in object_list_previous:
        vx_prv = prv_item.get('Vx',0.0)
        vy_prv = prv_item.get('Vy',0.0)
        for item in object_list_new:
            if prv_item['ID'] == item['ID'] \
              and prv_item.get('missed',False) == False: # 前回データが推定値の場合はやらない
                dtime = time_now - last_time
                ratio = min(1.0, dtime/filt_time)
                item['Vx'] = (1.0 - ratio)*vx_prv + ratio*(item['x'] - prv_item['x'])/dtime
                item['Vy'] = (1.0 - ratio)*vy_prv + ratio*(item['y'] - prv_item['y'])/dtime
    return object_list_new

# 速度推定値に基づいて検出できてないアイテムの位置を補正する
def corr_missed_position(objects_missed, last_time, time_now):
    for item in objects_missed:
        item['x'] = item['x'] + item.get('Vx',0.0)*(time_now - last_time)
        item['y'] = item['y'] + item.get('Vy',0.0)*(time_now - last_time)
    return objects_missed

# 検出できていない アイテムの推定を retain_time 秒保持する
def retain_missed_items(object_list_new, objects_missed, time_now, retain_time):
    object_list_previous = copy.copy(object_list_new)
    for item in objects_missed:
        time_diff = time_now - item['Missed_time']
        if time_diff < retain_time:
            object_list_previous.append(item)
    return object_list_previous
### Fukada's algorithm end 
####################################################################################

##########################################
# メイン関数のはじめの部分を独立
#
def object_detection_init():
#if __name__ == '__main__':

    #labels = ['blank']
    with open(args.labels,'r') as f:
        for line in f:
            labels.append(line.rstrip())

    # 録画
    if args.device == 'raspi_cam':
        cam.capture(stream, 'bgr', use_video_port=True)
        img = stream.array
    else:
        ret, img = cam.read()
    frame_rate = 24.0 
    y_size, x_size, c = img.shape
    record_size = (x_size, y_size)
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('record_video.mp4', fmt, frame_rate, record_size)
    return writer

############################################
# 無限ループの部分を関数化
#                   writer, idnet, projects, object_list_previous, last_time=time.time(), onomatope_list=[])
def object_detection( writer, idnet, projects, object_list_previous, last_time, onomatope_list):
    stutus = True
    if args.device == 'raspi_cam':
        cam.capture(stream, 'bgr', use_video_port=True)
        img = stream.array
    else:
        ret, img = cam.read()
        if not ret:
            print('error')
            stutus = False

    key = cv2.waitKey(1)
    if key == 27: # when ESC key is pressed break
        stutus = False

    img_bgr = cv2.resize(img, (300, 300))
    # convert bgr to rgb
    image_np = img_bgr[:,:,::-1]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    start = time.time()

    # コアの object_detection はここ
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    elapsed_time = time.time() - start

    # 検出結果それぞれの特性取得
    time_now = time.time()
    object_list_new = scan_detected_object(img, output_dict, time_now, score_sh=0.5)

    # 深田追加部分 ： ＩＤ付与
    object_list_new = remove_duplication(object_list_new, duplication_scale=0.1)
    object_list_new, objects_missed = give_IDs(idnet, projects, object_list_new, object_list_previous, time_now, accept_ratio=0.5)
    # re_inf
    output_re_inf, objects_missed2 = re_inference_image(objects_missed, img, size_mag=1.5)
    object_list_new2 = scan_detected_object(img, output_re_inf, time_now, score_sh=0.1)
    object_list_new += object_list_new2
    object_list_new, objects_missed2 = give_IDs(idnet, projects, object_list_new, object_list_previous, time_now, accept_ratio=0.5)
    # re_inf end
    object_list_new = infer_velocity(object_list_new, object_list_previous, last_time, time_now, filt_time=0.8) # filt_time 0.6->1.2->0.8
    objects_missed = corr_missed_position(objects_missed, last_time, time_now)
    object_list_previous = retain_missed_items(object_list_new, objects_missed, time_now, retain_time=4.0) # retain_time 3->4
    object_list_previous = remove_duplication(object_list_previous, duplication_scale=0.1)
    for onomatope in onomatope_list:
        for object in object_list_previous:
            if object["ID"]==onomatope[0]:
                object["onomatope"] = onomatope[1]

    put_bounding_box(img, object_list_previous, v_scale=1.0)
    speed_info = '%s: %.3f' % ('fps', 1.0/elapsed_time)
    cv2.putText(img, speed_info, (10,50), \
      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('detection result', img)
    writer.write(img) # 録画
    
    if args.device == 'raspi_cam':
        stream.seek(0)
        stream.truncate()
    return stutus, object_list_previous, time_now

def object_detection_end(writer):
    writer.release()    # 録画ファイルを閉じる
    tf_sess.close()     # global 変数
    cam.release()       # global 変数
    cv2.destroyAllWindows() # global 変数

if __name__ == '__main__':
    # prepare ID_network
    idnet = ids.ID_NetWork()
    project_ID = idnet.register_ID({"data_type": "project"}) # 1
    app_ID =     idnet.register_ID({"data_type": "app"})     # 2
    work_ID =    idnet.register_ID({"data_type": "work"})    # 3
    projects = {"data_type": "point", "project_ID": project_ID, "app_ID": app_ID, "work_ID": work_ID}
    # prepare ID_network end
    writer = object_detection_init()
    object_list_previous = []
    while True:
        status, object_list_previous, time_now = object_detection(writer, idnet, projects, object_list_previous, last_time=time.time(), onomatope_list=[])
        if status == False:
            break
    object_detection_end(writer)
    #writer.release() # 録画ファイルを閉じる
    #tf_sess.close()
    #cam.release()
    #cv2.destroyAllWindows()
