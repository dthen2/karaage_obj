# 深田のＩＤネットワーク
import id_network2 as ids
import copy
import textreaderGR as tx

# ファイル名の文字列
FILENAME_TOP = 'case1/frame_'
RES_10 = '00000'
RES_100 = '0000'
RES_1000 = '000'
RES_10000 = '00'
RES_100000 = '0'
RES_1000000 = ''

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
            if i != j and diff2 < acceptable_diff2 and item_a.get("detection_score",0.0) < item_s.get("detection_score",0.0):
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
# object_list_new のキー      "x", "y", "size_x", "size_y", が必要で、 'ID', 'assained' を新たに書き込む
# object_list_previous のキー "x", "y", "size_x", "size_y", 'ID', が必須で、 'Missed', 'Found', 'Missed_time'
def give_IDs(idnet, projects, object_list_new, object_list_previous, time_now, accept_ratio, class_factor=0.2):
    obj = copy.copy(projects)
    min_size = 10
    objects_missed = []
    mutch_list = []
    p = 0
    for prv_item in object_list_previous:
        i = 0
        for item in object_list_new:
            dc  = min(1, max(-1, item.get('class', 0) - prv_item.get('class', 0)))
            dx  = item["x"] - prv_item["x"]
            dy  = item["y"] - prv_item["y"]
            dsx = item["size_x"] - prv_item["size_x"]
            dsy = item["size_y"] - prv_item["size_y"]
            distance_2 = dx**2 + dy**2 + dsx**2 + dsy**2 + dc**2*class_factor
            acceptable_d2 = max(max(min_size, max(    item["size_x"],     item["size_y"]))**2,\
                                max(min_size, max(prv_item["size_x"], prv_item["size_y"]))**2)
            mutch = {"dist_ratio": distance_2/acceptable_d2, 'No_new': i, "No_prev": p}
            mutch_list.append(mutch)
            i += 1
        p += 1
    # マッチング度合いでソートし、高い順にマッチングしていく
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


def filenameSTR(num):
    top = FILENAME_TOP
    numstr = str(num)
    if num < 10:
        res = RES_10
    elif num < 100:
        res = RES_100
    elif num < 1000:
        res = RES_1000
    elif num < 10000:
        res = RES_10000
    elif num < 100000:
        res = RES_100000
    else:
        res = RES_1000000
    filename = top + res + numstr + '.txt'
    return filename

#                idns, project, object_list_previous, last_time, 
def text2object(idnet, projects, object_list_previous, frame, time_now, last_time):
    filename = filenameSTR(frame)
    object_list_new, status = tx.reader(filename)
    object_list_new = remove_duplication(object_list_new, duplication_scale=0.1)
    object_list_new, objects_missed = give_IDs(idnet, projects, object_list_new, object_list_previous, time_now, accept_ratio=0.5, class_factor=10000.)
    object_list_new = infer_velocity(object_list_new, object_list_previous, last_time, time_now, filt_time=0.8) # filt_time 0.6->1.2->0.8
    objects_missed = corr_missed_position(objects_missed, last_time, time_now)
    object_list_previous = retain_missed_items(object_list_new, objects_missed, time_now, retain_time=4.0) # retain_time 3->4
    object_list_previous = remove_duplication(object_list_previous, duplication_scale=0.1)
    return status, object_list_previous, time_now

if __name__ == '__main__':

    idnet = ids.ID_NetWork()
    project_ID = idnet.register_ID({"data_type": "project"}) # 1
    app_ID =     idnet.register_ID({"data_type": "app"})     # 2
    work_ID =    idnet.register_ID({"data_type": "work"})    # 3
    projects = {"data_type": "point", "project_ID": project_ID, "app_ID": app_ID, "work_ID": work_ID}
    # prepare ID_network end
    object_list_previous = []
    time_now = 0.0
    sampling_time = 0.1 #sec
    last_time = time_now
    frame = 0

    while True:
        time_now += sampling_time
        status, object_list_previous, time_ = text2object(idnet, projects, object_list_previous, frame, time_now, last_time)
        if status == False:
            break
        last_time = copy.copy(time_now)
        frame += 1
        print(object_list_previous)