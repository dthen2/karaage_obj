from email.mime import base
import copy
import time
import math
import numpy as np
#import object_detection_ids4 as obj
import give_idsGR as giv
import point_cloud6 as pcd
import memory_space6 as memory
import id_network2 as ids
import category2 as cat
import cogni_lingu2 as cog
import field as fld

##########################################################################
# オノマトペ、関係性の出力形式を刷新
# memory_space と、id_network、それに点群を駆使するアルゴリズムの初試作
# 認知言語学っぽい認知のアップデートに合わせて、更新してVer.2 とした
# GRファクトリーのデータを読み込んで評価するシステムとしてVer7をベースに制作
# Ver.7 と共に、ファイルに記録する仕組みを追加 in Apply4
# Ver.6 に当たって、認知言語学っぽく動きを評価するApply6を追加。
# この出力（Attentionを伴うeventで、データはディクショナリ形式の評価）を使って、
#   Apply2 の時系列を切る仕様
#   Apply4 に表示する仕様
# を追加
#
# 時系列を、n+1時限目に時間を入れる仕様の見直し(これではnumpy使えない)
# 点群のマッチング点検
# 点群のnumpy化（そのためにpoint_cloud6を作った）
# を実施

#########################################################################
#    GRファクトリー用のパラメータ集
# 2022Apr13 「作業場所」を点として登録。空間スケールの単位は画像のピクセル
PLACES = []#    {'x': 400., 'y': 300, 'size': 50., 'class': 'Place1'}\
#            , {'x': 900., 'y': 300, 'size': 50., 'class': 'Place1'}\
#            , {'x': 400., 'y': 600, 'size': 50., 'class': 'Place1'}\
#            , {'x': 900., 'y': 600, 'size': 50., 'class': 'Place1'}\
#        ]
# 画面スケールのパラメータは textreaderGR.py に入ってるので注意
# フィールドのパラメータ。Apply1で入力しているが、使うのは Appl6, Appl7 の認知
TIME_RESOLUTION = 0.1 #sec
SPACE_RESOLUTION = 40.0 #pixel
# フレーム毎に画像を止める時間 sec 大きく設定するとスローモーションになる
PAUSE_TIME = 0.001
# 動画のフレームレートの逆数 sec
SAMPLING_TIME = 0.05
# 最初に読み込むデータのフレームナンバー
INITIAL_FRAME = 0
# 画面サイズ
PIXEL_SIZE_X = 1280
PIXEL_SIZE_Y = 720
########################################################################

# Apply1: karaageさんobject_detection
# Apply2: 少数点群を時系列の多数点群に変換
# Apply3: 少数点群のフーリエ変換
# Apply4: 別のグラフィック画面に画き出し、ファイルに記録する
# Apply5: 多数点群を、カテゴリーと付き合わせる
# Apply6: 認知言語学っぽく動きを評価
# Apply7: 認知言語学っぽく関係性を評価

time_glov = 1.0 # ここをゼロにすると、time_now == False 判定になってしまうので注意
real_time=False

def time_g(real_time=False):
    if real_time:
        return time.time()
    else:
        return time_glov

##########################################################
# 登場しなくなったアイテムを除去する。worksリストのメンバーにtime_orgがあれば動作する。
# ここでtime_orgはメモリー空間から得たアイテムのタイムスタンプ。
# これがretain_time（秒）以上更新されていないものを除去する
def remove_unused_works(retain_time, works):
    for n in reversed(range(len(works))):
        if time_g() > works[n].time_org + retain_time:
            del works[n]

##########################################################
# Karaageさんのobject_detectionを処理するアプリ
# 物体検出のリストを、少数点群としてポスト。
# オノマトペを取り込む仕様追加 2022Jan27
class Apply1:
    def __init__(self, memory_space, idnet, projectID, fields):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": 0.0,'additional_str':'Object detection'})
        self.workID = idnet.register_ID({"data_type":"work","time_stamp": 0.0})
        self.fieldID = idnet.register_ID({"data_type":"field","time_stamp": 0.0})
        f = fld.Field(idnet, fields, self.fieldID)  # fieldの登録 2022Mar25 追加
        f.time_resolution = TIME_RESOLUTION#0.1 #sec
        f.space_resolution = SPACE_RESOLUTION#40.0 #pixel
        f.max_x = PIXEL_SIZE_X
        f.max_y = PIXEL_SIZE_Y
        self.dataID = idnet.register_ID({"data_type":"point_cloud_small_num","time_stamp": 0.0})
        self.project = {"data_type": "point", "projectID": projectID, "appID": self.appID, "workID": self.workID}
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.object_list_previous = []
        self.status = False     # ESCキーが押されたことの判定
        self.point_cloud = pcd.PointCloud(2, xs=[], points=[])
        self.data_on_memory = memory.Data_On_Memory(projectID, self.appID, self.workID, self.fieldID, self.dataID)
        self.data_on_memory.work_origin_IDs = [self.workID]
        self.last_time = time_g()-0.1    # 2022Feb21
        self.plases = PLACES    # 2022Apr13 作業場所の表
        for place in self.plases:
            place['ID'] = idnet.register_ID({"data_type":"point","time_stamp": 0.0})

    def app_main1(self, frame):
        self.status, self.object_list_previous, self.last_time = giv.text2object(self.idns, self.project, self.object_list_previous, frame, time_now = time_g(), last_time=self.last_time)
        new_points = []
        participant_IDs = []
        for object in self.object_list_previous:
            participant_IDs.append(object["ID"])
            new_point = pcd.Point_F(2, time_g())
            new_point.ID = object['ID']
            new_point.x = [object['x'],  PIXEL_SIZE_Y-object['y']] 
            new_point.v = [object.get('Vx',0.0), -object.get('Vy',0.0)]
            new_point.radius = (object['size_x']+object['size_y'])/4.
            if object.get('class') != None:
                new_point.attribute.append(object['class'])
            new_points.append(new_point)
        # 2022Apr13追加。「作業場所」を点群として加える
        for place in self.plases:
            participant_IDs.append(place["ID"])
            new_point = pcd.Point_F(2)
            new_point.ID = place['ID']
            new_point.x = [place['x'],  place['y']] 
            new_point.v = [0.0, 0.0]
            new_point.radius = place['size']    
            new_point.attribute.append(place['class'])
            new_points.append(new_point)
        # 点群を更新
        self.point_cloud.points = new_points
        self.point_cloud.renew_cloud()

        self.data_on_memory.data_type = "point_cloud_small_num"
        self.data_on_memory.participant_IDs = participant_IDs
        self.data_on_memory.time_stamp = time_g()   # タイムスタンプ
        self.data_on_memory.data = self.point_cloud               
        upload_data_list = [self.data_on_memory]    # このアプリでは、点群1つだけをアップロードするので、このような形になる

        # 司書にアップロード依頼をする
        self.memory_space.librarian_counter_up(self.appID, upload_data_list)
        return self.status

#    def apply1_end(self):
#        obj.object_detection_end(self.writer)

####################################
# 検出物体のリストから、時系列データを生成するアプリ
# 元データが少数点群なので、一点づつ取り出し、その時系列を多数点群として生成する
# ここで、point は Point_F クラスの点
class Work_in_app2:
    def __init__(self, time_org, dim, ID, x, idns, work_origin_ID):
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time_g()})            # このアプリでは、単点毎にworkを生成する
        self.time_org = time_org        # 最新入力データのタイムスタンプ
        self.data_orgID = ID            # 入力データのID
        self.fieldID = 0                # 入力データのものを継承する（例）が、初期化には入れない
        self.work_origin_ID = work_origin_ID
        #self.data_org = data_org        # 入力データ
        #self.data_created = []          # 生成した手持ちデータのリスト
        self.points = pcd.PointCloud(dim, xs=[x], points=[])
        self.points.ID = idns.register_ID({"data_type":"time_series_point_cloud_small_num","time_stamp": time_org})
        self.points.times.append(time_org)  #2022Mar21 追加 
        self.event = {}                 # 2022Mar15 追加
        self.event_time = 0             # 2022Mar15 追加
        self.time_begin = 0             # 2022Mar15 追加
        self.is_new = True              # 2022Mar15 追加

class Apply2:
    event_dead_time = 3.       # 2022Mar15 追加 イベントが連続するとみなす時間
    def __init__(self, memory_space, idnet, projectID, fields):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time_g(),'additional_str':'Point to time_series_cloud'})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
        # データの要求条件作成。ディクショナリのリストである
        # ここで、"projectID" と、"appID"(自分自身のID) の2つは絶対に必要なキーなので忘れないように
        self.data_aqcuire_conditions = [{ "aq_data_types": ["point_cloud_small_num"], "projectID": self.projectID, "appID": self.appID},\
                                        { "aq_data_types": ["event"],                 "projectID": self.projectID, "appID": self.appID}]
        self.memory_space.set_aq_conditions(self.data_aqcuire_conditions)

    # メモリー空間とのやりとりも含めたプロセス本体。
    def app_main2(self):
        # データ置き場を空にする
        self.aq_datas_list = []
        # 司書にデータ取得依頼をする
        self.memory_space.librarian_counter_down(self.appID, self.aq_datas_list)
        # 前回にメモリー空間からもらったデータをスキャンし、work 毎に整理する。
        # self.aq_datas_list は、Data_On_Memory形式データのリストになってるはず
        for data in self.aq_datas_list:
            if data.data_type == "event": # 2022Mar15追加。App5の出力
                #for event in data.data:
                for work in self.works :
                    if data.dataID == work.data_orgID: # ここでは、点群の評価としてのイベントのみ表示するので、一致しなければ破棄する
                        work.event = data.data
                        if data.time_stamp > work.event_time + self.event_dead_time:
                            work.event_time = data.time_stamp
        for data in self.aq_datas_list:
            if data.data_type == "point_cloud_small_num":
                for point in data.data.points: # データが少数点群である前提
                    x = copy.copy(point.x)
                    #x.append(data.time_stamp) # 次元をオーバーした所を時刻とする事で、時系列データとする・・・廃止2022Mar21
                    new_work_created = True
                    previousID = 0
                    for work in self.works :
                        if point.ID == work.data_orgID: # 現有ID一致しなければ、new_work_createdがON・・・一つの例
                            # イベントで切る 2022Mar15追加仕様
                            #  Punctuality が大きく、元の点群がself.event_dead_time以上古く、イベントがself.event_dead_timeより新しい
                            if work.event.get("Punctuality",0.) > 0.5 and \
                               time_g() - work.time_begin > self.event_dead_time and \
                               time_g() - work.event_time < self.event_dead_time:
                                new_work_created = True
                                previousID = work.points.ID
                                work.is_new = False
                                #print("Cut Cut Cut ID=%d" % work.data_orgID)
                            else:
                                new_work_created = False
                                if work.time_org < data.time_stamp and work.is_new:
                                    work.points.xs.append(x) # 入力pointは点で、work.pointsは多数点群。注意
                                    work.points.times.append(data.time_stamp)   # 2022Mar21 仕様変更
                                    work.points.renew_cloud()
                                    work.time_org = data.time_stamp
                    if new_work_created :
                        # 新しい点を検出したら新しいworkを立ち上げ、新しい時系列多数点群を創成する
                        new_work = Work_in_app2(data.time_stamp, point.dim, point.ID, x, self.idns, data.workID)
                        new_work.fieldID = data.fieldID
                        new_work.time_begin = time_g()   # 2022Mar21 追加
                        self.works.append(new_work)
                        relation_property_dat = {"data_type": "point", "projectID": self.projectID, "appID": self.appID, "workID": new_work.workID, "relation_IDs": []}
                        self.idns.register_relation(point.ID,     new_work.points.ID, 1.0, "processed", relation_property_dat)
                        # work同士の関係性の記述必要か？
                        relation_property_dat = {"data_type": "work", "projectID": self.projectID, "appID": self.appID, "workID": new_work.workID, "relation_IDs": []}
                        self.idns.register_relation(data.workID, new_work.workID,    1.0, "processed", relation_property_dat)
                        # 切った点群の連続性を serise の関係性として登録
                        if previousID != 0:
                            relation_property_dat = {"data_type": "time_series_point_cloud_large_num", "projectID": self.projectID, "appID": self.appID, "workID": new_work.workID, "relation_IDs": []}
                            self.idns.register_relation(previousID, new_work.points.ID, 1.0, "series", relation_property_dat)
        # 7秒以上更新されなかったら、そのworkを除去する
        remove_unused_works(7., self.works)
                    
        upload_data_list = []
        for work in self.works : # work 毎に処理を走らせる
            # データ処理本体
            # アップロードする新しいデータ作成
            data = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.points.ID, time_g())
            data.data = work.points
            data.time_stamp = time_g()
            data.data_type = "time_series_point_cloud_large_num"
            data.participant_IDs = [work.data_orgID]
            data.work_origin_IDs = [work.work_origin_ID]
            upload_data_list.append(data)
        
        # 司書にアップロード依頼をする
        self.memory_space.librarian_counter_up(self.appID, upload_data_list)


####################################
# 検出物体のリストから、各物体の動きを評価するアプリ
# 元データが少数点群なので、一点づつ取り出し、評価
# ここで、point は Point_F クラスの点
# 作成途上 2022Jan24
import fourier_trans as flt
class Work_in_app3:
    k = 60.
    FREQUENCY_LIST =  [10./k, 12.5/k, 16.0/k, 20./k, 25./k, 30./k,  40./k,  50./k, 60./k]
    m = 0.3
    decay_time_list = [50.*m, 40.*m,  31.*m,  25.*m, 20.*m, 16.7*m, 12.5*m, 10.*m, 8.3*m]
    filt_time_list =  [1., 3., 10., 30., 100., 300., 1000., 3000., 10000.]

    def __init__(self, time_org, dim, ID, x, idns, work_origin_ID, fields, fieldID):
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time_g()}, time_g())            # このアプリでは、単点毎にworkを生成する
        self.time_org = time_org        # 最新入力データのタイムスタンプ
        self.data_orgID = ID            # 入力データのID
        self.fieldID = fieldID                  # 入力データのものを継承する（例）
        field = fields.fields_dic[fieldID]
        self.fl_min = field.space_resolution    # 検出閾値 2022Mar25 変更
        self.work_origin_ID = work_origin_ID
        self.flx = flt.Fourier(self.FREQUENCY_LIST, self.decay_time_list, time_new=time_g())
        self.fly = flt.Fourier(self.FREQUENCY_LIST, self.decay_time_list, time_new=time_g())
        self.plx = flt.Impulse(self.filt_time_list, filt_ratio=5.0, time_new=time_g())
        self.ply = flt.Impulse(self.filt_time_list, filt_ratio=5.0, time_new=time_g())

class Apply3:
    def __init__(self, memory_space, idnet, projectID, fields):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time_g(),'additional_str':'Fourier examination'})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
        self.fields = fields
        # データの要求条件作成。ディクショナリのリストである
        # ここで、"projectID" と、"appID"(自分自身のID) の2つは絶対に必要なキーなので忘れないように
        self.data_aqcuire_conditions = [{ "aq_data_types": ["point_cloud_small_num"], "projectID": self.projectID, "appID": self.appID}]
        memory_space.set_aq_conditions(self.data_aqcuire_conditions)

    # メモリー空間とのやりとりも含めたプロセス本体。
    def app_main3(self):
        # データ置き場を空にする
        self.aq_datas_list = []
        # 司書にデータ取得依頼をする
        self.memory_space.librarian_counter_down(self.appID, self.aq_datas_list)
        # 前回にメモリー空間からもらったデータをスキャンし、work 毎に整理する。
        # self.aq_datas_list は、Data_On_Memory形式データのリストになってるはず
        for data in self.aq_datas_list:
            for point in data.data.points: # データが少数点群である前提
                x = copy.deepcopy(point.x)
                new_work_created = True
                for work in self.works :
                    if point.ID == work.data_orgID: # 現有ID一致しなければ、new_work_createdがON・・・一つの例
                        new_work_created = False
                        if work.time_org < data.time_stamp:
                            work.time_org = data.time_stamp
                            # フーリエ処理プロセス
                            work.flx.fourier_trans(imput=x[0], time_now=data.time_stamp)
                            work.fly.fourier_trans(imput=x[1], time_now=data.time_stamp)
                            work.plx.impulse(imput=x[0], time_now=data.time_stamp)
                            work.ply.impulse(imput=x[1], time_now=data.time_stamp)
                            #print(data.time_stamp)
                if new_work_created :
                    # 新しい点を検出したら新しいworkを立ち上げ、新しいフーリエ処理プロセスを開始する
                    new_work = Work_in_app3(data.time_stamp, point.dim, point.ID, x, self.idns, data.workID, self.fields, data.fieldID)
                    self.works.append(new_work)
                    # 関係性の記述を忘れずに
                    relation_property_dat = {"data_type": "evaluation", "projectID": self.projectID, "appID": self.appID, "workID": new_work.workID, "relation_IDs": []}
                    self.idns.register_relation(point.ID,     new_work.workID, 1.0, "processed", relation_property_dat)
                    # work同士の関係性の記述必要か？
                    relation_property_dat = {"data_type": "work", "projectID": self.projectID, "appID": self.appID, "workID": new_work.workID, "relation_IDs": []}
                    self.idns.register_relation(data.workID, new_work.workID, 1.0, "processed", relation_property_dat)
                    new_work.flx.fourier_trans(imput=x[0], time_now=data.time_stamp)
                    new_work.fly.fourier_trans(imput=x[1], time_now=data.time_stamp)
                    new_work.plx.impulse(imput=x[0], time_now=data.time_stamp)
                    new_work.ply.impulse(imput=x[1], time_now=data.time_stamp)
                    #print(data.time_stamp)
        # 更新されないデータは除去
        remove_unused_works(0.2, self.works)

        upload_data_list = []
        for work in self.works : # work 毎に処理を走らせる
            # データ処理本体
            # アップロードする新しいデータ作成
            data = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.data_orgID, time_g())
            x0 = int(work.flx.components[0]['Real']**2 + work.flx.components[0]['Imagin']**2) # フンワフンワ
            x1 = int(work.flx.components[1]['Real']**2 + work.flx.components[1]['Imagin']**2) # フワフワ
            x2 = int(work.flx.components[2]['Real']**2 + work.flx.components[2]['Imagin']**2) # ユラユラ
            x3 = int(work.flx.components[3]['Real']**2 + work.flx.components[3]['Imagin']**2) # ユラユラ2
            x4 = int(work.flx.components[4]['Real']**2 + work.flx.components[4]['Imagin']**2) # フラフラ
            x5 = int(work.flx.components[5]['Real']**2 + work.flx.components[5]['Imagin']**2) # フラフラ2
            x6 = int(work.flx.components[6]['Real']**2 + work.flx.components[6]['Imagin']**2) # ぐらぐら
            x7 = int(work.flx.components[7]['Real']**2 + work.flx.components[7]['Imagin']**2) # ぐらぐら2
            x8 = int(work.flx.components[8]['Real']**2 + work.flx.components[8]['Imagin']**2) # ブルブル
            #print(work.flx.components)
            #print(work.fl_min)
            onomatodict = {"FunwaFunwa":x0, "FuwaFuwa":x1, "YuuraYuura":x2,"YuraYura":x3,\
                          "FuuraFuura":x4, "FuraFura":x5, "GuraGura":x6, "GataGata":x7,"BuruBuru":x8}
            
            data.data_type = "evaluation"
            fl_max = max(x0,x1,x2,x3,x4,x5,x6,x7,x8,work.fl_min)
            if x0 == fl_max:
                onomatopoeia = "FunwaFunwa"
            elif x1 == fl_max:
                onomatopoeia = "FuwaFuwa"
            elif x2 == fl_max:
                onomatopoeia = "YuuraYuura"
            elif x3 == fl_max:
                onomatopoeia = "YuraYura"
            elif x4 == fl_max:
                onomatopoeia = "FuuraFuura"
            elif x5 == fl_max:
                onomatopoeia = "FuraFura"
            elif x6 == fl_max:
                onomatopoeia = "GuraGura"
            elif x7 == fl_max:
                onomatopoeia = "GataGata"
            elif x8 == fl_max:
                onomatopoeia = "BuruBuru"
            else:
                onomatopoeia = ""

            # 動作確認のための表示文字列
            #txt = 'ID:' + str(work.data_orgID)+' '+str(x0) +' '+str(x1)+' '+str(x2)+' '+str(x3)+' '+str(x4)+' '+str(x5)+' '+str(x6)+' '+str(x7)+' '+str(x8)
            #if work.data_orgID==14:
            #    print(txt)
            #    print(onomatopoeia)
            ### 動作確認コード end ###
            
            data.data = {"onomatopoeia": onomatopoeia, "onomatodict":onomatodict}
            data.time_stamp = time_g()
            data.work_origin_IDs = [work.work_origin_ID]
            data.participant_IDs = [work.data_orgID]
            upload_data_list.append(data)

        # 司書にアップロード依頼をする
        self.memory_space.librarian_counter_up(self.appID, upload_data_list)

####################################
# メモリー空間の情報をグラフィックに画き出すアプリ
# これはメモリー空間から読む一方で、書き込みをしない
# 2022Jan31現在、participantsが少数点群の場合は各点の位置と速度ベクトル、それに評価値（文字列）、時系列多数点群の場合は軌跡としてグラフィック表示
from matplotlib import pylab as plt
class Work_in_app4:
    def __init__(self, time_org, dim, ID, point_or_cloud, idns, data_type="Single_point"):
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time_g()})            # このアプリでは、単点毎にworkを生成する
        self.time_org = time_org        # 最新入力データのタイムスタンプ
        self.data_orgID = ID            # 入力データのID
        self.fieldID = 0                # 入力データのものを継承する（例）が、初期化には入れない
        self.point = point_or_cloud     # 例外的に、単点または点群が置かれる
        self.cloud_orgID = 0            # 点群の起源となる点のIDを遡って記載
        self.data_type = data_type
        self.event = {}                 # 2022Mar15 追加。認知言語学っぽい評価値  
        self.event_time = 0.            # 2022Mar15 追加。認知言語学っぽい評価値はイベントとして出力されてるので、そのタイムスタンプ
        self.event_workIDs = []         # 2022Mar25 追加。同じアイテムに異なるアプリからイベント検出された場合にそなえる
        self.onomatopoeia = ''
        self.onomatodict = {}           # 2022Apr26 追加
        self.color = 0 #color #'red'
        self.mark = 'o'

    # 2022Apr8 追加 
    def highestCogni(self):
        if type(self.event) == dict:
            maxval = 0.0
            for key in self.event.keys():
                if self.event[key] > maxval:
                    maxval = self.event[key]
                    keymax = key
            if maxval > 0.0:
                return keymax
            else:
                return ''
        else:
            return ''

    # 2022Apr26 追加 
    def cogniTexts(self):
        texts = ''
        if type(self.event) == dict:
            for key in self.event.keys():
                text = ', ' + key + ':%.2f' % self.event[key]
                texts += text
        return texts

class Apply4:
    def __init__(self, memory_space, idnet, projectID, fields):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time_g(),'additional_str':'Data drawing'})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
        self.fig = plt.figure(figsize=(11.0, 4.5))
        self.ax = self.fig.add_subplot(1,1,1)
        self.fig.show()
        #self.max_x = -200000
        #self.min_x = 200000
        #self.max_y = -200000
        #self.min_y = 200000
        self.fields = fields
        self.scale_x = 600
        self.scale_y = 400
        # データの要求条件作成。ディクショナリのリストである
        # ここで、"projectID" と、"appID"(自分自身のID) の2つは絶対に必要なキーなので忘れないように
        self.data_aqcuire_conditions = [{ "aq_data_types": ["point_cloud_small_num"], "projectID": self.projectID, "appID": self.appID},\
                                        { "aq_data_types": ["evaluation"],            "projectID": self.projectID, "appID": self.appID},\
                                        { "aq_data_types": ["point_cloud_large_num"], "projectID": self.projectID, "appID": self.appID},\
                                        { "aq_data_types": ["event"],                 "projectID": self.projectID, "appID": self.appID},\
                                        { "aq_data_types": ["time_series_point_cloud_large_num"], "projectID": self.projectID, "appID": self.appID}]
        memory_space.set_aq_conditions(self.data_aqcuire_conditions)
        self.file = open('recordGR.txt', 'a')

    def colors_list(self, id):
        colors = ['red', 'green', 'blue', 'orange', 'cyan', 'Magenta', 'gold']
        cid = id % len(colors) # % は剰余演算子
        color = colors[cid]
        return color

    #def max_scale(self, x, r):
    #    self.max_x = max(self.max_x, x[0]+r)
    #    self.min_x = min(self.min_x, x[0]-r)
    #    self.max_y = max(self.max_y, x[1]+r)
    #    self.min_y = min(self.min_y, x[1]-r)
    #    self.scale_x = self.max_x - self.min_x
    #    self.scale_y = self.max_y - self.min_y

    def print_ID(self, work, x, radius, indicateID, attribute=[], frame=0):
        f_text = 'frame=%d, time=%.2f, ' % (frame, work.time_org)
        text = 'ID:' + str(work.data_orgID)
        f_text += text + ', x=%.2f, y=%.2f, r=%.2f' % (x[0], x[1], radius)
        if attribute != []:
            f_text += ', Class:%s' % str(attribute[0])
        x_sift = self.scale_x/60
        y_sift = self.scale_y/4000
        if indicateID:
            self.ax.text(x[0]-x_sift, x[1]-y_sift*(radius+100), text, size=10, color=work.color)
        y_sift_onomatope = y_sift*(radius+100)*0.3
        self.ax.text(    x[0]-x_sift, x[1]+y_sift_onomatope, work.onomatopoeia, size=10, color=work.color)
        if work.onomatopoeia != '':
            for onokey in work.onomatodict.keys():
                f_text += ', ' + onokey + ':%.2f' % (work.onomatodict[onokey])
        sift_tx = 0
        for event_key in work.event.keys():
            if work.event_time > time_g()-0.2:
                text_e = event_key + ':%.2f' % work.event[event_key] #' + str(work.event[event_key])
                sift_tx += 19   # 行間
                self.ax.text( x[0]-x_sift, x[1]+y_sift_onomatope + sift_tx, text_e, size=10, color=work.color)
                f_text += ', ' + text_e
        f_text += '\n'
        if indicateID:
            self.file.write(f_text)

    def print_relation(self, work, text, frame):
        text_f = 'frame=%d, time=%.2f' % (frame, work.time_org)
        text_f += ', eventID:%d' % work.data_orgID
        text_f += ', ID1:%d, ID2:%d' % (work.point[0].ID, work.point[1].ID) + text + '\n'
        self.file.write(text_f)


    # メモリー空間とのやりとりも含めたプロセス本体。
    def app_main4(self, frame):
        # データ置き場を空にする
        self.aq_datas_list = []
        # 司書にデータ取得依頼とアップロード依頼をする
        self.memory_space.librarian_counter_down(self.appID, self.aq_datas_list)

        # 前回にメモリー空間からもらったデータをスキャンし、work 毎に整理する。
        # self.aq_datas_list は、Data_On_Memory形式データのリストになってるはず
        for data in self.aq_datas_list:
            if data.data_type == "event": # 2022Mar15追加。App5の出力
                #for event in data.data:
                new_work_created = True
                for work in self.works :
                    if data.dataID == work.data_orgID: # ここでは、点群の評価としてのイベントのみ表示するので、一致しなければ破棄する
                        evnet_updated = False
                        new_work_created = False
                        for workID in work.event_workIDs:
                            if data.workID == workID:
                                work.event = data.data      # 同じworkからのイベントデータであれば上書きする
                                work.event_time = data.time_stamp
                                if work.data_type=="Relation":
                                    work.time_org = data.time_stamp
                                evnet_updated = True
                        if evnet_updated != True:
                            work.event.update(data.data)    # 2022Mar25 追加。異なるworkからのイベントであれば、update機能で追加
                            work.event_time = data.time_stamp
                            work.event_workIDs.append(data.workID)
                # 2022Apr8 追加
                if new_work_created:
                    if len(data.participant_IDs) >= 2: # とりあえず、2点間の関係だけ扱う
                        ID1 = data.participant_IDs[0]
                        ID1_found = False
                        ID2 = data.participant_IDs[1]
                        ID2_found = False
                        for work in self.works :
                            if ID1 == work.data_orgID:
                                ID1_found = True
                                work1 = work
                            if ID2 == work.data_orgID:
                                ID2_found = True
                                work2 = work
                        if ID1_found and ID2_found:
                            color = self.colors_list(data.dataID)
                            new_work = Work_in_app4(data.time_stamp, 2, data.dataID, [work1.point, work2.point], self.idns, data_type="Relation")
                            self.scale_x = max(self.scale_x, self.fields.fields_dic[data.fieldID].max_x)
                            self.scale_y = max(self.scale_y, self.fields.fields_dic[data.fieldID].max_y)
                            new_work.color = color
                            new_work.fieldID = data.fieldID
                            new_work.event = data.data
                            self.works.append(new_work)
            if data.data_type == "point_cloud_small_num":
                for point in data.data.points: # データが少数点群である前提
                    new_work_created = True
                    for work in self.works :
                        if point.ID == work.data_orgID: # 現有ID一致しなければ、new_work_createdがON・・・一つの例
                            new_work_created = False
                            if work.time_org < data.time_stamp:
                                work.point = copy.copy(point)
                                work.time_org = data.time_stamp
                    if new_work_created :
                        # 新しい点を検出したら新しいworkを立ち上げ、
                        color = self.colors_list(point.ID)
                        new_work = Work_in_app4(data.time_stamp, point.dim, point.ID, copy.copy(point), self.idns)
                        self.scale_x = max(self.scale_x, self.fields.fields_dic[data.fieldID].max_x)
                        self.scale_y = max(self.scale_y, self.fields.fields_dic[data.fieldID].max_y)
                        new_work.color = color
                        new_work.fieldID = data.fieldID
                        self.works.append(new_work)
            if data.data_type == "time_series_point_cloud_large_num" or data.data_type == "point_cloud_large_num":
                # データが時系列多数点群の場合
                new_work_created = True
                for work in self.works :
                    if data.dataID == work.data_orgID: # 現有ID一致しなければ、new_work_createdがON・・・一つの例
                        new_work_created = False
                        if work.time_org < data.time_stamp:
                            work.point = copy.copy(data.data) # ごっそり更新する
                            work.time_org = data.time_stamp
                if new_work_created :
                    # 新しい点群を検出したら新しいworkを立ち上げ、
                    # 点群の起源を探り、色を同じにする
                    new_work = Work_in_app4(data.time_stamp, data.data.dim, data.dataID, copy.copy(data.data), self.idns, "Cloud")
                    self.scale_x = max(self.scale_x, self.fields.fields_dic[data.fieldID].max_x)
                    self.scale_y = max(self.scale_y, self.fields.fields_dic[data.fieldID].max_y)
                    new_work.fieldID = data.fieldID
                    relatedIDs = self.idns.related_fromIDs(data.dataID)
                    color_determined = False
                    for relatedID in relatedIDs:
                        prop_org = self.idns.get_property(relatedID['fromID'])
                        if prop_org['data_type'] == 'point':
                            cloud_orgID = relatedID['fromID']
                            color = self.colors_list(cloud_orgID)
                            color_determined = True
                    if color_determined == False:
                        color = self.colors_list(data.dataID)
                        new_work.cloud_orgID = data.dataID
                    else:
                        new_work.cloud_orgID = cloud_orgID
                    new_work.color = color
                    self.works.append(new_work)
        for data in self.aq_datas_list:
            if data.data_type == "evaluation":
                for work in self.works :
                    if data.dataID == work.data_orgID: 
                        # オノマトペのデータ形式にディクショナリ追加
                        work.onomatopoeia = copy.copy(data.data.get("onomatopoeia",''))
                        work.onomatodict = copy.copy(data.data.get("onomatodict", {}))

        # 0.2秒以上更新されなかったら、そのworkを除去する
        remove_unused_works(0.2, self.works)

        plt.cla()
        self.ax.set_xlim(0,self.scale_x)
        self.ax.set_ylim(0,self.scale_y)
        for work in self.works:
            if work.data_type=="Single_point": 
                x = work.point.x
                # 枠サイズ確保のため
                #self.max_scale(x, 0.)
                # 点の描画
                size = work.point.radius**2*2.
                self.ax.scatter(x[0],x[1],c = 'none', edgecolors=work.color, alpha=1.0, s=size, marker=work.mark)
                self.ax.set_xlim(0,1820)
                self.ax.set_ylim(0,720)
                # 速度ベクトルの描画
                v = np.dot(1.,work.point.v)
                x_list = [x[0], x[0]+v[0]]
                y_list = [x[1], x[1]+v[1]]
                self.ax.plot(x_list, y_list, color = work.color)
                # IDの記述
                self.print_ID(work, x, work.point.radius, True, work.point.attribute, frame)
            elif work.data_type=="Cloud":
                #if work.data_orgID == work.cloud_orgID: # 元のIDが検出できなかった場合はIDを表示する
                #print(work.data_orgID)
                x = work.point.centre
                r = math.sqrt(work.point.cloud_radius)
                # 枠サイズ確保のため
                #self.max_scale(x, r)
                if work.data_orgID == work.cloud_orgID: # 元のIDが検出できなかった場合はIDを表示する
                    self.print_ID(work, x, r, True, [], frame)
                else:
                    self.print_ID(work, x, r, False, [], frame)

                xx=[]
                xy=[]
                for x in work.point.xs:
                    xx.append(x[0])
                    xy.append(x[1])
                self.ax.plot(xx,xy,color=work.color,linestyle="solid")
                #following_point = False
                #for x in work.point.xs:
                #    if following_point:
                #        self.ax.plot([x_pre[0],x[0]],[x_pre[1],x[1]],color=work.color,linestyle="solid")
                #    x_pre = copy.deepcopy(x)
                #    following_point = True
            # 2022Apr8 追加
            elif work.data_type=="Relation":
                x1 = work.point[0].x[0]
                y1 = work.point[0].x[1]
                x2 = work.point[1].x[0]
                y2 = work.point[1].x[1]
                self.ax.plot([x1,x2],[y1,y2],color=work.color,linestyle="dotted")
                text_p = work.highestCogni()   # 仕様変更 2022Apr26
                text_r = work.cogniTexts()
                self.ax.text( (x1+x2)/2, (y1+y2)/2, text_p, size=10, color=work.color)
                self.print_relation(work, text_r, frame)

        # 枠サイズ確保のため
        #if (self.max_x>-100000 and self.min_x<100000):
        #    self.ax.scatter(self.max_x,self.max_y,c = 'none', edgecolors='w', alpha=1.0, s=1, marker=',')
        #    self.ax.scatter(self.min_x,self.min_y,c = 'none', edgecolors='w', alpha=1.0, s=1, marker=',')
            #self.scale_x = self.max_x - self.min_x
            #self.scale_y = self.max_y - self.min_y
        self.fig.show()
        plt.pause(0.001)

        # データ置き場を空にする
        #self.aq_datas_list = []
        # 司書にデータ取得依頼とアップロード依頼をする
        #memory_space.librarian_counter(self.appID, self.data_aqcuire_conditions, [], self.aq_datas_list)

####################################
# 多数点群を、カテゴリーと付き合わせるアプリ
class Work_in_app5:
    def __init__(self, workID, time_org, dim, dataID, field, data, work_origin_ID, data_orgType):
        self.workID = workID
        self.time_org = time_org    # 入力データのタイムスタンプ
        self.data_orgID = dataID    # 入力データのID
        self.fieldID = field        # 入力データのものを継承する（例）が、初期化には入れない
        self.work_origin_ID = work_origin_ID
        self.data_orgType = data_orgType
        self.data = data      # 入力データ
        self.categoryNum = -1       # 一致探索中のカテゴリーの番号（リスト中の位置）。探索モード1つ数値を挙げるので-1でスタート
        self.prototypeID = 0        # 一致探索中のプロトタイプの元データのID
        self.pnum = 0               # 一致探索中のプロトタイプのリスト中番号
        self.snum = 0               # 一致探索中のステレオタイプのリスト中番号
        self.categoryID = 0         # カテゴリーのID。prototypeIDと混同しないように！
        self.ref_points = 0         # ステレオタイプまたはプロトタイプの点群を格納する
        self.dim = dim
        self.closest = 0            #pcd.Closest(dim, self.data_xs, self.ref_xs)  # point_cloud5.py にあるマッチングクラス
        self.counter1 = 0   # 乱数探索の回数カウンタ
        self.count1_max = 100 # 乱数探索の上限回数
        self.counter2 = 0   # グラジェントアプローチの回数カウンタ
        self.count2_max = 2000 # グラジェントアプローチの上限回数
        self.mode = 0       # 0ならカテゴリーの探索、1なら乱数探索、2ならグラジェントアプローチ
        self.mat = 0.0      # マッチング評価値
        self.is_converge = False# マッチング収束
        self.d_x = []
        self.d_rot = []
        self.d_mag = []
        for n in range(dim):
            self.d_x.append(0.0)
            self.d_rot.append(0.0)
            self.d_mag.append(0.0)
        self.x_scale   = 0.0
        self.rot_scale = 0.0 
        self.mag_scale = 0.0

    def reset_finding(self):
        self.mat = 0.0      # マッチング評価値
        self.is_converge = False# マッチング収束
        self.d_x = []
        self.d_rot = []
        self.d_mag = []
        for n in range(self.dim):
            self.d_x.append(0.0)
            self.d_rot.append(0.0)
            self.d_mag.append(0.0)
        self.x_scale   = 0.0
        self.rot_scale = 0.0 
        self.mag_scale = 0.0

class Apply5:
    DISSMISSED = -1
    CATEGORY_FINDING = 0
    RANDOM_APPROACH = 1
    GRADIENT_APPROACH = 2
    def __init__(self, memory_space, idnet, projectID, fields, categories):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.cats = categories
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time_g(),'additional_str':'Category mutching'})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
        # データの要求条件作成。ディクショナリのリストである
        # 少数点群も含めちゃうところがミソ
        self.data_aqcuire_conditions = [{ "aq_data_types": ["time_series_point_cloud_large_num",\
                                                            "point_cloud_large_num",\
                                                            "point_cloud_small_num"],\
                                          "projectID": self.projectID, "appID": self.appID, "include_self_app": False}]
        self.memory_space.set_aq_conditions(self.data_aqcuire_conditions)
        self.sampling_time = 5.0    # データ取得を行う周期
        self.time_last = time_g()

    # グラジェントに基づいて、点群のマッチングを行う
    def gradient_approach(self, search_num_g, work):
        work.closest.trans_best_to_default()
        for i in range(search_num_g):
            diff_rot, diff_x, diff_mag = work.closest.closest_point_diff(work.resolution, work.rot_scale, work.x_scale, work.mag_scale)
            for n in range(work.dim):
                work.d_x[n]   = diff_x[n]
                work.d_rot[n] = diff_rot[n]
                work.d_mag[n] = diff_mag[n]
            work.mat, work.is_converge = work.closest.closest_point_approach( work.resolution, work.mat, work.d_x, work.d_rot, work.d_mag)
            work.closest.trans_best_to_default()
            if work.is_converge:
                break
        work.counter2 += search_num_g

    # メモリー空間からのデータ収集。このルーティンは長いサンプリングタイム(self.sampling_time)置きに実行する
    def app_main5_aq(self):
        time_now = time_g()
        if time_now > (self.time_last+self.sampling_time):
            self.time_last = time_now
            # データ置き場を空にする
            self.aq_datas_list = []
            # 司書にデータ取得依頼とアップロード依頼をする
            self.memory_space.librarian_counter_down(self.appID, self.aq_datas_list)

            # メモリー空間からもらったデータをスキャンし、探索中でなければ新しいworkを立ち上げて追加。
            for data in self.aq_datas_list:
                working = False
                n = 0
                old_data_work = -1
                for work in self.works:
                    if work.data_orgID == data.dataID:
                        if data.time_stamp > work.time_org:
                            old_data_work = n
                            break
                        else:
                            working = True
                            break
                    n += 1
                # 探索中であっても、データのタイムスタンプが更新されていたら、破棄して新たなワークを立ち上げる 2022Mar22
                if old_data_work > -1:
                    del self.works[old_data_work]
                if working == False:
                    workID = self.idns.register_ID({"data_type":"work", "time_stamp": time_g()})
                    newwork = Work_in_app5(workID, data.time_stamp, data.data.dim, data.dataID, data.fieldID, copy.deepcopy(data.data), data.workID, data.data_type)
                    self.works.append(newwork)

    # カテゴリーとの一致を探索する。このルーティンは短いサンプリングタイムで実行
    def app_main5_infer(self):
        upload_data_list = []
        for work in self.works:
            # 先ずは一致探索するカテゴリーを決める
            # カテゴリーを順番（ランダムにしたい）にチェックし、そのステレオタイプに、データタイプの一致するものがあるか確認
            if work.mode == self.CATEGORY_FINDING:
                work.categoryNum, snum, pnum = self.cats.find_same_type_category(["time_series_point_cloud_large_num",\
                                                  "point_cloud_large_num",\
                                                  "point_cloud_small_num"], work.categoryNum+1)
                if work.categoryNum >= 0 and len(work.data.xs) > 1 and work.data.cloud_radius > 0.:
                    work.categoryID = self.cats.categories[work.categoryNum]["ID"]
                    if snum < 0:
                        work.ref_points  = self.cats.categories[work.categoryNum]["prototypes"][pnum]["data"]
                        work.snum = snum
                    else:
                        work.ref_points  = self.cats.categories[work.categoryNum]["stereotypes"][snum]["data"]
                        work.prototypeID = self.cats.categories[work.categoryNum]["prototypes"][pnum]["dataID"]
                        work.pnum = pnum
                    work.closest = pcd.Closest(work.dim, work.data.xs, work.ref_points.xs )
                    cloud_tmp = pcd.PointCloud(2, xs=copy.deepcopy(work.data.xs), points=[])
                    cloud_tmp.points_line_sort()
                    if cloud_tmp.cloud_radius == 0.0:
                        work.mode = self.DISSMISSED
                    else:
                        work.x_scale   = cloud_tmp.dist_mean*0.1
                        work.rot_scale = cloud_tmp.dist_mean/cloud_tmp.cloud_radius*0.1
                        work.mag_scale = cloud_tmp.dist_mean/cloud_tmp.cloud_radius*0.1
                        work.resolution = cloud_tmp.dist_mean/10. 
                        # モード切替
                        work.mode = self.RANDOM_APPROACH
                        #print('ID=%d dist_mean=%.3f' % (work.data_orgID, cloud_tmp.dist_mean))
                else :
                    work.mode = self.DISSMISSED

            # 先ず、乱数探索のモード
            # 一致度 mat は、最大で1になるように再設計すべき
            matching_srsh_rand = 0.8
            if work.mode == self.RANDOM_APPROACH:
                search_num_r = 10
                work.mat = work.closest.closest_point_search( search_num=search_num_r, resolution=work.resolution, matching_func_best=work.mat)
                work.counter1 += search_num_r
                if work.mat >= matching_srsh_rand or work.counter1 > work.count1_max:
                    work.mode = self.GRADIENT_APPROACH

            # グラジェントアプローチ
            if work.mode == self.GRADIENT_APPROACH:
                search_num_g = 5
                self.gradient_approach(search_num_g, work)
                if work.data_orgID==15:
                    print('In gradient approach ID: %d. mat=%.3f %s' % (work.data_orgID, work.mat, str(work.is_converge)))

            # 一致を検出したら、関係性登録
            # 表示用に自分で投げたデータを再び処理してしまうので、後で対策すること
            # matching_srsh_grad を小さくすれば、テスト用に何でもマッチングするようにできる
            matching_srsh_grad = 0.5
            if work.is_converge and work.mat >= matching_srsh_grad: 
                if work.prototypeID >= 0:
                    relation_IDs = [work.prototypeID]
                    cat_data_type = self.cats.categories[work.categoryNum]["prototypes"][work.pnum]["data_type"]
                else:
                    relation_IDs = [] 
                    cat_data_type = self.cats.categories[work.categoryNum]["stereotypes"][work.snum]["data_type"]
                relation_property_dat = {"data_type": work.data_orgType, "projectID": self.projectID, "appID": self.appID, "workID": work.workID, "relation_IDs": relation_IDs}
                self.idns.register_relation(work.data_orgID,  work.categoryID, work.mat, "matched", relation_property_dat)
                # 一致検出したら、そのプロトタイプorステレオタイプをメモリー空間に投げる
                property_dat = {"data_type":  "point_cloud_large_num", "time_stamp": time_g(), \
                                "projectID":  self.projectID, "appID":  self.appID, "workID": work.workID, \
                                "relation_IDs":   [work.data_orgID, work.categoryID], \
                                "additional_str": "matched category"}
                newID = self.idns.register_ID(property_dat)
                data_on_memory = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, newID)
                data_on_memory.data_type = cat_data_type
                data_on_memory.time_stamp = time_g()
                data_on_memory.participant_IDs = [work.data_orgID] + relation_IDs
                new_xs = work.closest.coordinate_antitransformation(work.ref_points.xs)
                data_on_memory.data = pcd.PointCloud( work.ref_points.dim, new_xs, points=[])
                upload_data_list.append(copy.copy(data_on_memory))    # 
                print("Category matching obtained for ID=%d" % work.data_orgID)
                # さらに評価値をメモリー空間に投入。暫定的に"onomatopoeia"を使っている
                eval_data = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.data_orgID, time_g())
                eval_data.data_type = "evaluation"
                category_txt = "Category ID is %-2d" % work.categoryID
                eval_data.data = {"onomatopoeia": category_txt}
                eval_data.time_stamp = time_g()
                eval_data.work_origin_IDs = [work.work_origin_ID]
                upload_data_list.append(eval_data)
                work.mode = self.CATEGORY_FINDING   # また次のカテゴリーの探索に入る
                work.reset_finding()
            elif work.is_converge : # 探索収束はしたがマッチしていない場合
                work.mode = self.CATEGORY_FINDING   # また次のカテゴリーの探索に入る
                work.reset_finding()
            elif work.counter2 > work.count2_max:
                # タイムアップでカテゴリーを替えてやり直し
                work.mode = self.CATEGORY_FINDING

        # 20秒以上更新されなかったら、そのworkを除去する
        remove_unused_works(20., self.works)
        print("Num of infer = %d" % len(self.works))
        # 司書にアップロード依頼をする
        self.memory_space.librarian_counter_up(self.appID, upload_data_list)
#########
                
####################################
# 認知言語学っぽく、少数点群の各点の動きを評価するアプリ
# ここで、point は Point_F クラスの点
class Work_in_app6:
    #filter_time = 0.30 #sec
    def __init__(self, time_org, point, ID, idns, work_origin_ID, fields, fieldID):
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time_g()})            # このアプリでは、単点毎にworkを生成する
        self.time_org = time_org        # 最新入力データのタイムスタンプ
        self.data_orgID = ID            # 入力データのID
        self.fieldID = fieldID          # 入力データのものを継承する
        field = fields.fields_dic[fieldID]
        self.work_origin_ID = work_origin_ID
        self.cogObj = cog.Detected_Object(point, field.time_resolution, time_g())
        self.cogObj.STANDARD_SPEED = field.space_resolution # 今回の速度スケールに合わせたモノ
        self.cogObj.A_STANDARD = field.space_resolution     # 今回の速度スケールに合わせたモノ
        
class Apply6:
    def __init__(self, memory_space, idnet, projectID, fields):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.fields = fields
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time_g(),'additional_str':'Cognitive Lingistics for single point'})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
        # データの要求条件作成。ディクショナリのリストである
        # ここで、"projectID" と、"appID"(自分自身のID) の2つは絶対に必要なキーなので忘れないように
        self.data_aqcuire_conditions = [{ "aq_data_types": ["point_cloud_small_num"], "projectID": self.projectID, "appID": self.appID}]
        self.memory_space.set_aq_conditions(self.data_aqcuire_conditions)

    # メモリー空間とのやりとりも含めたプロセス本体。
    def app_main6(self):
        # データ置き場を空にする
        self.aq_datas_list = []
        # 司書にデータ取得依頼をする
        self.memory_space.librarian_counter_down(self.appID, self.aq_datas_list)
        # 前回にメモリー空間からもらったデータをスキャンし、work 毎に整理する。
        # self.aq_datas_list は、Data_On_Memory形式データのリストになってるはず
        for data in self.aq_datas_list:
            for point in data.data.points: # データが少数点群である前提
                new_work_created = True
                for work in self.works :
                    if point.ID == work.data_orgID: # 現有ID一致しなければ、new_work_createdがON・・・一つの例
                        new_work_created = False
                        if work.time_org < data.time_stamp:
                            work.cogObj.point= point
                            work.time_org = data.time_stamp
                if new_work_created and point.ID != 0:
                    # 新しい点を検出したら新しいworkを立ち上げ
                    new_work = Work_in_app6(data.time_stamp, point, point.ID, self.idns, data.workID, self.fields, data.fieldID)
                    self.works.append(new_work)
                    # work同士の関係性の記述必要か？
                    relation_property_dat = {"data_type": "work", "projectID": self.projectID, "appID": self.appID, "workID": new_work.workID, "relation_IDs": []}
                    self.idns.register_relation(data.workID, new_work.workID,    1.0, "processed", relation_property_dat)
        # 0.2秒以上更新されなかったら、そのworkを除去する
        remove_unused_works(0.2, self.works)
                    
        upload_data_list = []
        for work in self.works : # work 毎に処理を走らせる
            # データ処理本体
            work.cogObj.objects_kinesis(work.cogObj.point, work.time_org)
            # アップロードする新しいデータ作成
            if work.cogObj.Attention > 0.5:
                data = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.data_orgID, time_g())
                data.data = {"Kinesis": work.cogObj.Kinesis}
                if work.cogObj.Punctuality > cog.COGNI_SRESH:
                    data.data["Punctuality"] = work.cogObj.Punctuality
                if work.cogObj.AntiPunctuality > cog.COGNI_SRESH:
                    data.data["AntiPunctuality"] = work.cogObj.AntiPunctuality
                if work.cogObj.Volitionality > cog.COGNI_SRESH:
                    data.data["Volitionality"] = work.cogObj.Volitionality
                data.Attention_self = work.cogObj.Attention
                data.time_stamp = work.time_org #cogObj.Attention_time
                data.data_type = "event"
                data.participant_IDs = [work.data_orgID]
                data.work_origin_IDs = [work.work_origin_ID]
                upload_data_list.append(data)
        
        # 司書にアップロード依頼をする
        if upload_data_list != []:
            self.memory_space.librarian_counter_up(self.appID, upload_data_list)

####################################
# 認知言語学っぽく、少数点群の各点の関係性を評価するアプリ
# ここで、point は Point_F クラスの点
class Work_in_app7:
    filter_time = 1.0 #sec
    def __init__(self, time_org1, point1, ID1, work_origin_ID1,     \
                       time_org2, point2, ID2, work_origin_ID2, fieldID, idns, fields):
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time_g()})            # このアプリでは、単点の組み合わせ毎にworkを生成する
        self.time_org1 = time_org1        # 最新入力データのタイムスタンプ
        self.time_org2 = time_org2        # 最新入力データのタイムスタンプ
        self.time_org = max(time_org1, time_org2)
        self.data_orgID1 = ID1            # 入力データのID ここで、常に ID1 > ID2 にする
        self.data_orgID2 = ID2            # 入力データのID
        self.fieldID = fieldID              # 入力データのものを継承する
        field = fields.fields_dic[fieldID] # 2022Mar25 追加
        self.work_origin_ID1 = work_origin_ID1
        self.work_origin_ID2 = work_origin_ID2
        self.cogObj1 = cog.Detected_Object(point1, self.filter_time, time_g())
        self.cogObj2 = cog.Detected_Object(point2, self.filter_time, time_g())
        self.cogObj1.STANDARD_SPEED = field.space_resolution # 今回の速度スケールに合わせたモノ
        self.cogObj2.STANDARD_SPEED = field.space_resolution # 今回の速度スケールに合わせたモノ
        self.cogObj1.A_STANDARD = field.space_resolution     # 今回の速度スケールに合わせたモノ
        self.cogObj2.A_STANDARD = field.space_resolution     # 今回の速度スケールに合わせたモノ
        self.cogRelation = cog.Relation(self.cogObj1, self.cogObj2, self.filter_time, time_g())
        self.event_ID = 0                # work 作成時点ではIDは設定しない（まだ関係が「有る」とは言えない）

class Apply7:
    def __init__(self, memory_space, idnet, projectID, fields):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.fields = fields
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time_g(),'additional_str':'Cognitive Lingistics for two points relation'})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
        # データの要求条件作成。ディクショナリのリストである
        # ここで、"projectID" と、"appID"(自分自身のID) の2つは絶対に必要なキーなので忘れないように
        self.data_aqcuire_conditions = [{ "aq_data_types": ["point_cloud_small_num"], "projectID": self.projectID, "appID": self.appID}]
        self.memory_space.set_aq_conditions(self.data_aqcuire_conditions)

    # メモリー空間とのやりとりも含めたプロセス本体。
    def app_main7(self):
        # データ置き場を空にする
        self.aq_datas_list = []
        # 司書にデータ取得依頼をする
        self.memory_space.librarian_counter_down(self.appID, self.aq_datas_list)
        # 前回にメモリー空間からもらったデータをスキャンし、work 毎に整理する。
        # self.aq_datas_list は、Data_On_Memory形式データのリストになってるはず
        points = []
        for data in self.aq_datas_list:
            for point in data.data.points: # データが少数点群である前提
                points.append({"point":point, "field": data.fieldID, "time": data.time_stamp, "workID": data.workID})

        for pointA in points:
            for pointB in points:
                if pointA["point"].ID > pointB["point"].ID:
                    point1 = pointA
                    point2 = pointB
                else:
                    point1 = pointB
                    point2 = pointA
                new_work_created = True
                for work in self.works :
                    if point1["point"].ID == work.data_orgID1 and point2["point"].ID == work.data_orgID2: # 現有ID一致しなければ、new_work_createdがON・・・一つの例
                        new_work_created = False # 現有workに一致したら、データ更新
                        if work.time_org1 < point1["time"]:
                            work.cogObj1.point= point1["point"]
                            work.time_org1 = point1["time"]
                        if work.time_org2 < point2["time"]:
                            work.cogObj2.point= point2["point"]
                            work.time_org2 = point2["time"]
                        work.time_org = max(work.time_org1, work.time_org2)

                if new_work_created and point1["field"] == point2["field"] and point1["point"].ID != point2["point"].ID:
                    # 新しい組み合わせ検出したら新しいworkを立ち上げ、新しい関係探索を開始
                    # ただし、fieldID が一致していること
                    #                       time_org2       point,           ID,                 work_origin_ID2, idns)
                    new_work = Work_in_app7(point1["time"], point1["point"], point1["point"].ID, point1["workID"],\
                                            point2["time"], point2["point"], point2["point"].ID, point2["workID"], point1["field"], self.idns, fields)
                    self.works.append(new_work)
                    # work同士の関係性の記述必要か？
                    relation_property_dat = {"data_type": "work", "projectID": self.projectID, "appID": self.appID, "workID": new_work.workID, "relation_IDs": []}
                    self.idns.register_relation(point1["workID"], new_work.workID,    1.0, "processed", relation_property_dat)
                    self.idns.register_relation(point2["workID"], new_work.workID,    1.0, "processed", relation_property_dat)
        # 0.2秒以上更新されなかったら、そのworkを除去する
        remove_unused_works(0.2, self.works)
                    
        upload_data_list = []
        for work in self.works : # work 毎に処理を走らせる
            # データ処理本体
            cogni_dict = work.cogRelation.relation_evaluation(work.cogObj1, work.cogObj2, work.time_org)    # 2022Apr27 時刻を変更
            # アップロードする新しいデータ作成
            if work.cogRelation.Attention > 0.5 :#or work.cogRelation.Contact: # Contact追加2022Apr6->ContactでAttention立てるようにしたので不要
                relation_property_dat = {"data_type": "event", "projectID": self.projectID, "appID": self.appID, "workID": work.workID, "relation_IDs": []}
                #strength = max(work.cogRelation.Attention, float(work.cogRelation.Contact))# ContactでAttention立てるようにしたので不要
                #if work.cogRelation.Contact:# ディクショナリ統合したので不要
                #    cogni_dict['Contact'] = 1.0
                fromID = work.data_orgID1
                toID = work.data_orgID2
                self.idns.register_relation(fromID, toID, work.cogRelation.Attention, "interaction", relation_property_dat)
                # 関係ではなく、イベントとしてデータ登録
                # ここで、データIDの考え方は要再考（App6も同じ)
                if work.event_ID == 0:
                    work.event_ID = self.idns.register_ID({"data_type":"event", "time_stamp": time_g()})
                updata = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.event_ID, time_g())
                updata.data = cogni_dict
                #if work.cogObj.Punctuality > cog.COGNI_SRESH:
                #    updata.data["Punctuality"] = work.cogObj.Punctuality
                #if work.cogObj.AntiPunctuality > cog.COGNI_SRESH:
                #    updata.data["AntiPunctuality"] = work.cogObj.AntiPunctuality
                #if work.cogObj.Volitionality > cog.COGNI_SRESH:
                #    updata.data["Volitionality"] = work.cogObj.Volitionality
                updata.Attention_self = work.cogRelation.Attention
                updata.time_stamp = work.time_org # cogRelation.Attention_time
                updata.data_type = "event"
                updata.participant_IDs = [work.data_orgID1, work.data_orgID2]
                updata.work_origin_IDs = [work.work_origin_ID1,work.work_origin_ID2]
                upload_data_list.append(updata)
        
        # 司書にアップロード依頼をする
        if upload_data_list != []:
            self.memory_space.librarian_counter_up(self.appID, upload_data_list)

######################################################
# モニタ画面
import tkinter as tk
class Monitor:
    def __init__(self, idnet):
        self.applies = []
        for id in idnet.IDs:
            prop = idnet.get_property(id)
            if prop["data_type"]=='app':
                self.applies.append({'ID':id, 'add_str':prop.get('additional_str','')})
        appnum = len(self.applies)
        y_geo_str = str(100+20*appnum)
        self.root = tk.Tk()
        self.root.geometry('500x'+y_geo_str)
        self.root.title('memory space status')
        self.time_last = time.time()

    def monitor(self, memory_space):
        sampling_time = 'Sampling time = %.2f  ' % (time.time() - self.time_last)
        self.time_last = time.time()
        label = tk.Label(self.root, text=sampling_time)
        label.place(x = 10, y= 10)

        memory_size = 'Number of data on memory space = ' + str(len(memory_space.memory_space)) + '        '
        label = tk.Label(self.root, text=memory_size)
        label.place(x = 10, y= 30)
    
        keys = memory_space.upload_status.keys()
        upload_status = 'Number of uploading applies         = ' + str(len(keys)) + '        '
        label = tk.Label(self.root, text=upload_status)
        label.place(x = 10, y= 50)

        label = tk.Label(self.root, text='App : num of uploaded data')
        label.place(x = 10, y= 70)
        y = 90
        for app in self.applies:
            upload = 0
            for key in keys:
                if key==app['ID']:
                    upload = memory_space.upload_status[key]
            each_app = "ID: %-3d %-45s  upload: %-2d" % (app['ID'], app['add_str'], upload)
            label = tk.Label(self.root, text=each_app, font=("Courier",9)) # 等幅フォント
            label.place(x = 10, y=y)
            y += 20

class Monitor2:
    SYSTEM = 0
    def __init__(self, idnet):
        self.applies = {self.SYSTEM: {'add_str':'Memory manager', "samp_time": 0.0}}
        for id in idnet.IDs:
            prop = idnet.get_property(id)
            if prop["data_type"]=='app':
                self.applies[id] = {'add_str':prop.get('additional_str',''), "samp_time": 0.0}
        appnum = len(self.applies)
        y_geo_str = str(100+20*appnum)
        self.root = tk.Tk()
        self.root.geometry('600x'+y_geo_str) # ウィンドウサイズ
        self.root.title('memory space status')
        self.time_last = time.time()

    def monitor2(self, memory_space):
        line_space = 20
        y = 10
        sampling_time = 'Sampling time = %.3f  ' % (time.time() - self.time_last)
        self.time_last = time.time()
        label = tk.Label(self.root, text=sampling_time)
        label.place(x = 10, y=y)

        y += line_space
        system_time = 'Sampling time of memory manaerger = %.3f' % self.applies[self.SYSTEM]['samp_time']
        label = tk.Label(self.root, text=system_time)
        label.place(x = 10, y=y)

        y += line_space
        memory_size = 'Number of data on memory space = ' + str(len(memory_space.memory_space)) + '        '
        label = tk.Label(self.root, text=memory_size)
        label.place(x = 10, y=y)
    
        y += line_space
        upIDs = memory_space.upload_status.keys()
        upload_status = 'Number of uploading applies         = ' + str(len(upIDs)) + '        '
        label = tk.Label(self.root, text=upload_status)
        label.place(x = 10, y=y)

        y += line_space
        label = tk.Label(self.root, text="App'sID Description                                 Num of upload   Sampling time", font=("Courier",9))
        label.place(x = 10, y=y)
        y += line_space
        appIDs = self.applies.keys()
        for appID in appIDs:
            
            if appID != self.SYSTEM:
                upload = memory_space.upload_status.get(appID, 0)
                #each_app = "   -- %-45s              time: %.3f" % (self.applies[appID]['add_str'],                self.applies[appID]['samp_time'])
                each_app = "ID: %-3d %-45s upload: %-3d    time: %.3f" % (appID, self.applies[appID]['add_str'], upload, self.applies[appID]['samp_time'])
                label = tk.Label(self.root, text=each_app, font=("Courier",9)) # 等幅フォント
                label.place(x = 10, y=y)
                y += line_space

    def time_monitor(self, appID, prev_time):
        time_now = time.time()
        self.applies[appID]["samp_time"] = time_now - prev_time
        return time_now
        

########################################
# カテゴリーの登録
def init_category(idnet, projectID):
    categories = cat.Categories(idnet, projectID, "AImind5_categories") # 2022Feb4追加、カテゴリー
    # cat.SAMPLE_CLOUD は「の」の字。多数点群をサンプルカテゴリーとして登録。
    prototypes = [  {"Data":cat.SAMPLE_CLOUD,   "Description": "no no ji"}, \
                    {"Data":cat.SAMPLE_CLOUD_A, "Description": "A no ji"}, \
                    {"Data":cat.SAMPLE_CLOUD_B, "Description": "B no ji"}, \
                    {"Data":cat.SAMPLE_CLOUD_C, "Description": "C no ji"}]
    for prototype in prototypes:
        data = pcd.PointCloud(2, xs=prototype["Data"], points=[])
        id = idnet.register_ID({"data_type":"category","time_stamp": time_g()})
        type = { \
            "data_type":  "point_cloud_large_num", \
            "dataID":     id, \
            "data":       data,  \
            "time_stamp": time_g(), \
            "appID":      0, \
            "workID":     0,  \
            "stength":    1.0}
        category_data = {"ID": id, "prototypes": [type], "stereotypes": []}
        categories.register_category(category_data, prototype["Description"])
    return categories

########################################
if __name__ == "__main__":
    idnet = ids.ID_NetWork()
    memory_space = memory.Memory_space()
    projectID = idnet.register_ID({"data_type":"project","time_stamp": time_g()}) # 本当は、それなりのプロセスを経て新規プロジェクト生成
    fields = fld.Fields()
    categories = init_category(idnet, projectID)

    apply1 = Apply1(memory_space, idnet, projectID, fields)
    apply2 = Apply2(memory_space, idnet, projectID, fields)
    apply3 = Apply3(memory_space, idnet, projectID, fields)
    apply4 = Apply4(memory_space, idnet, projectID, fields)
    apply5 = Apply5(memory_space, idnet, projectID, fields, categories)
    apply6 = Apply6(memory_space, idnet, projectID, fields)
    apply7 = Apply7(memory_space, idnet, projectID, fields)
    #moni = Monitor2(idnet)
    frame = INITIAL_FRAME

    while(True): # 本当は、タイマー機能でやる。後ほど
        prev_time = time.time()
        memory_space.memory_space_manerger(time_g())
        memory_space.librarian_swapper_before_scanner()
        memory_space.librarian_scanner(time_g())
        memory_space.librarian_swapper_after_scanner()
        #prev_time = moni.time_monitor(moni.SYSTEM, prev_time)
        
        status = apply1.app_main1(frame)
        #prev_time = moni.time_monitor(apply1.appID, prev_time)
        apply2.app_main2()
        #prev_time = moni.time_monitor(apply2.appID, prev_time)
        apply3.app_main3()
        #prev_time = moni.time_monitor(apply3.appID, prev_time)
        apply4.app_main4(frame)
        #prev_time = moni.time_monitor(apply4.appID, prev_time)
        #apply5.app_main5_aq()
        #apply5.app_main5_infer()
        #prev_time = moni.time_monitor(apply5.appID, prev_time)
        apply6.app_main6()
        #prev_time = moni.time_monitor(apply6.appID, prev_time)
        apply7.app_main7()
        #prev_time = moni.time_monitor(apply7.appID, prev_time)

        #moni.monitor2(memory_space)
        time_glov += SAMPLING_TIME
        frame += 1
        plt.pause(PAUSE_TIME)

        if status == False :
            break
    apply4.file.close()
    #apply1.apply1_end()
        
        

