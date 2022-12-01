from email.mime import base
import object_detection_ids4 as obj
import point_cloud6 as pcd
import memory_space6 as memory
import id_network2 as ids
import category2 as cat
import cogni_lingu as cog
import copy
import time
import math
import numpy as np

##########################################################################
# memory_space と、id_network、それに点群を駆使するアルゴリズムの初試作
# Ver.6 に当たって、認知言語学っぽく動きを評価するApply6を追加。
# この出力（Attentionを伴うeventで、データはディクショナリ形式の評価）を使って、
#   Apply2 の時系列を切る仕様
#   Apply4 に表示する仕様
# を追加した

## この先やること ##
# 時系列を、n+1時限目に時間を入れる仕様の見直し(これではnumpy使えない)
# 点群のマッチング点検
# 点群のnumpy化（そのためにpoint_cloud6を作った）

######################################
# Apply1: karaageさんobject_detection
# Apply2: 少数点群を時系列の多数点群に変換
# Apply3: 少数点群のフーリエ変換
# Apply4: 別のグラフィック画面に画き出すだけ
# Apply5: 多数点群を、カテゴリーと付き合わせる
# Apply6: 認知言語学っぽく動きを評価

#######################################
# 登場しなくなったアイテムを除去する。worksリストのメンバーにtime_orgがあれば動作する。
# ここでtime_orgはメモリー空間から得たアイテムのタイムスタンプ。
# これがretain_time（秒）以上更新されていないものを除去する
def remove_unused_works(retain_time, works):
    for n in reversed(range(len(works))):
        if time.time() > works[n].time_org + retain_time:
            del works[n]

##########################################################
# Karaageさんのobject_detectionを処理するアプリ
# 物体検出のリストを、少数点群としてポスト。
# オノマトペを取り込む仕様追加 2022Jan27
class Apply1:
    def __init__(self, memory_space, idnet, projectID):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time.time(),'additional_str':'Object detection'})
        self.workID = idnet.register_ID({"data_type":"work","time_stamp": time.time()})
        self.fieldID = idnet.register_ID({"data_type":"field","time_stamp": time.time()})
        self.dataID = idnet.register_ID({"data_type":"point_cloud_small_num","time_stamp": time.time()})
        #print('object detection ID =' +str(self.dataID))
        self.project = {"data_type": "point", "projectID": projectID, "appID": self.appID, "workID": self.workID}
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        #self.works = []
        self.data_aqcuire_conditions = [{ "aq_data_types": ["evaluation"], "projectID": self.projectID, "appID": self.appID}] # データ取得条件のリスト
        memory_space.set_aq_conditions(self.data_aqcuire_conditions)
        self.writer = obj.object_detection_init()
        self.object_list_previous = []
        self.status = False     # ESCキーが押されたことの判定
        self.point_cloud = pcd.PointCloud(2, xs=[], points=[])
        self.data_on_memory = memory.Data_On_Memory(projectID, self.appID, self.workID, self.fieldID, self.dataID)
        self.data_on_memory.work_origin_IDs = [self.workID]
        ### テスト用多数点群。「の」の字の多数点群
        self.point_cloud_large = pcd.PointCloud(2, xs=cat.SAMPLE_CLOUD2, points=[])
        self.point_cloud_large.ID = idnet.register_ID({"data_type":"point_cloud_large_num","time_stamp": time.time()})
        dataID2 = idnet.register_ID({"data_type":"point_cloud_large_num","time_stamp": time.time()})
        self.data_on_memory2 = memory.Data_On_Memory(projectID, self.appID, self.workID, self.fieldID, dataID2)
        #print('nonoji ID=' + str(dataID2))
        self.data_on_memory2.data_type = "point_cloud_large_num"
        self.data_on_memory2.participant_IDs = []
        self.data_on_memory2.work_origin_IDs = [self.workID]
        self.data_on_memory2.data = self.point_cloud_large  
        self.last_time = time.time()-0.1    # 2022Feb21
        #print(self.data_on_memory2.data.cloud_radius)
        #print(self.point_cloud_large.cloud_radius)        
        

    def app_main1(self):
        # データ置き場を空にする
        self.aq_datas_list = []
        # 司書にデータ取得依頼とアップロード依頼をする
        self.memory_space.librarian_counter_down(self.appID, self.aq_datas_list)
        # オノマトペを取り込む仕様追加 2022Jan27
        onomatope_list = []
        for data in self.aq_datas_list:
            onomatope_list.append([data.dataID, data.data.get("onomatopoeia",'')])
        # Karaageさんのobject_detectionを走らせる
        self.status, self.object_list_previous, self.last_time = obj.object_detection(self.writer, self.idns, self.project, self.object_list_previous, last_time=self.last_time, onomatope_list=onomatope_list)
        new_points = []
        participant_IDs = []
        for object in self.object_list_previous:
            participant_IDs.append(object["ID"])
            new_point = pcd.Point_F(2)
            new_point.ID = object['ID']
            new_point.x = [object['x'],  700.-object['y']] 
            new_point.v = [object.get('Vx',0.0), -object.get('Vy',0.0)]
            new_point.radius = (object['size_y']+object['size_y'])/2.
            new_points.append(new_point)
        # 点群を更新
        self.point_cloud.points = new_points
        self.point_cloud.renew_cloud()

        self.data_on_memory.data_type = "point_cloud_small_num"
        self.data_on_memory.participant_IDs = participant_IDs
        self.data_on_memory.time_stamp = time.time()   # タイムスタンプ
        self.data_on_memory.data = self.point_cloud               
        upload_data_list = [self.data_on_memory]    # このアプリでは、点群1つだけをアップロードするので、このような形になる

        ########## テスト用にイレギュラーなアップロード 「の」の字の多数点群を投じる
        self.data_on_memory2.time_stamp = time.time()   # タイムスタンプ              
        upload_data_list.append(self.data_on_memory2)
        #upload_data_list = [self.data_on_memory,self.data_on_memory2]

        # 司書にアップロード依頼をする
        self.memory_space.librarian_counter_up(self.appID, upload_data_list)
        return self.status

    def apply1_end(self):
        obj.object_detection_end(self.writer)

####################################
# 検出物体のリストから、時系列データを生成するアプリ
# 元データが少数点群なので、一点づつ取り出し、その時系列を多数点群として生成する
# ここで、point は Point_F クラスの点
class Work_in_app2:
    def __init__(self, time_org, dim, ID, x, idns, work_origin_ID):
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time.time()})            # このアプリでは、単点毎にworkを生成する
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
    def __init__(self, memory_space, idnet, projectID):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time.time(),'additional_str':'Point to time_series_cloud'})
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
                    for work in self.works :
                        if point.ID == work.data_orgID: # 現有ID一致しなければ、new_work_createdがON・・・一つの例
                            # イベントで切る 2022Mar15追加仕様
                            #  Punctuality が大きく、元の点群がself.event_dead_time以上古く、イベントがself.event_dead_timeより新しい
                            if work.event.get("Punctuality",0.) > 0.5 and \
                               time.time() - work.time_begin > self.event_dead_time and \
                               time.time() - work.event_time < self.event_dead_time:
                                new_work_created = True
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
                        new_work.time_begin = time.time()   # 2022Mar21 追加
                        self.works.append(new_work)
                        relation_property_dat = {"data_type": "point", "projectID": self.projectID, "appID": self.appID, "workID": new_work.workID, "relation_IDs": []}
                        self.idns.register_relation(point.ID,     new_work.points.ID, 1.0, "processed", relation_property_dat)
                        # work同士の関係性の記述必要か？
                        relation_property_dat = {"data_type": "work", "projectID": self.projectID, "appID": self.appID, "workID": new_work.workID, "relation_IDs": []}
                        self.idns.register_relation(data.workID, new_work.workID,    1.0, "processed", relation_property_dat)
        # 3秒以上更新されなかったら、そのworkを除去する
        remove_unused_works(3., self.works)
                    
        upload_data_list = []
        for work in self.works : # work 毎に処理を走らせる
            # データ処理本体
            # アップロードする新しいデータ作成
            data = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.points.ID)
            data.data = work.points
            data.time_stamp = time.time()
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

    def __init__(self, time_org, dim, ID, x, idns, work_origin_ID):
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time.time()})            # このアプリでは、単点毎にworkを生成する
        self.time_org = time_org        # 最新入力データのタイムスタンプ
        self.data_orgID = ID            # 入力データのID
        self.fieldID = 0                # 入力データのものを継承する（例）が、初期化には入れない
        self.work_origin_ID = work_origin_ID
        self.flx = flt.Fourier(self.FREQUENCY_LIST, self.decay_time_list, time_new=time.time())
        self.fly = flt.Fourier(self.FREQUENCY_LIST, self.decay_time_list, time_new=time.time())
        self.plx = flt.Impulse(self.filt_time_list, filt_ratio=5.0, time_new=time.time())
        self.ply = flt.Impulse(self.filt_time_list, filt_ratio=5.0, time_new=time.time())

class Apply3:
    def __init__(self, memory_space, idnet, projectID):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time.time(),'additional_str':'Fourier examination'})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
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
                if new_work_created :
                    # 新しい点を検出したら新しいworkを立ち上げ、新しいフーリエ処理プロセスを開始する
                    new_work = Work_in_app3(data.time_stamp, point.dim, point.ID, x, self.idns, data.workID)
                    new_work.fieldID = data.fieldID
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
        # 更新されないデータは除去
        remove_unused_works(2., self.works)

        upload_data_list = []
        for work in self.works : # work 毎に処理を走らせる
            # データ処理本体
            # アップロードする新しいデータ作成
            data = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.data_orgID)
            x0 = int(work.flx.components[0]['Real']**2 + work.flx.components[0]['Imagin']**2) # フンワフンワ
            x1 = int(work.flx.components[1]['Real']**2 + work.flx.components[1]['Imagin']**2) # フワフワ
            x2 = int(work.flx.components[2]['Real']**2 + work.flx.components[2]['Imagin']**2) # ユラユラ
            x3 = int(work.flx.components[3]['Real']**2 + work.flx.components[3]['Imagin']**2) # ユラユラ2
            x4 = int(work.flx.components[4]['Real']**2 + work.flx.components[4]['Imagin']**2) # フラフラ
            x5 = int(work.flx.components[5]['Real']**2 + work.flx.components[5]['Imagin']**2) # フラフラ2
            x6 = int(work.flx.components[6]['Real']**2 + work.flx.components[6]['Imagin']**2) # ぐらぐら
            x7 = int(work.flx.components[7]['Real']**2 + work.flx.components[7]['Imagin']**2) # ぐらぐら2
            x8 = int(work.flx.components[8]['Real']**2 + work.flx.components[8]['Imagin']**2) # ブルブル
            
            data.data_type = "evaluation"
            fl_min = 10 # 本来はスケールなどから決定すべき
            fl_max = max(x0,x1,x2,x3,x4,x5,x6,x7,x8,fl_min)
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
            
            data.data = {"onomatopoeia": onomatopoeia}
            data.time_stamp = time.time()
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
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time.time()})            # このアプリでは、単点毎にworkを生成する
        self.time_org = time_org        # 最新入力データのタイムスタンプ
        self.data_orgID = ID            # 入力データのID
        self.fieldID = 0                # 入力データのものを継承する（例）が、初期化には入れない
        self.point = point_or_cloud     # 例外的に、単点または点群が置かれる
        self.cloud_orgID = 0            # 点群の起源となる点のIDを遡って記載
        self.data_type = data_type
        self.event = {}                 # 2022Mar15 追加。認知言語学っぽい評価値  
        self.event_time = 0.            # 2022Mar15 追加。認知言語学っぽい評価値はイベントとして出力されてるので、そのタイムスタンプ
        self.onomatopoeia = ''
        self.color = 0 #color #'red'
        self.mark = 'o'

class Apply4:
    def __init__(self, memory_space, idnet, projectID):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time.time(),'additional_str':'Data drawing'})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.fig.show()
        self.max_x = -200000
        self.min_x = 200000
        self.max_y = -200000
        self.min_y = 200000
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

    def colors_list(self, id):
        colors = ['red', 'green', 'blue', 'orange', 'cyan', 'Magenta', 'gold']
        cid = id % len(colors) # % は剰余演算子
        color = colors[cid]
        return color

    def max_scale(self, x, r):
        self.max_x = max(self.max_x, x[0]+r)
        self.min_x = min(self.min_x, x[0]-r)
        self.max_y = max(self.max_y, x[1]+r)
        self.min_y = min(self.min_y, x[1]-r)
        self.scale_x = self.max_x - self.min_x
        self.scale_y = self.max_y - self.min_y

    def print_ID(self, work, x, radius, indicateID):
        text = 'ID:' + str(work.data_orgID)
        x_sift = self.scale_x/60
        y_sift = self.scale_y/4000
        if indicateID:
            self.ax.text(x[0]-x_sift,max(self.min_y-y_sift*500, x[1]-y_sift*(radius+100)), text, size=10, color=work.color)
        y_sift_onomatope = y_sift*(radius+100)
        self.ax.text(    x[0]-x_sift,                           x[1]+y_sift_onomatope, work.onomatopoeia, size=10, color=work.color)
        sift_tx = 0
        for event_key in work.event.keys():
            if work.event_time > time.time()-2.:
                text = event_key + ': %.2f' % work.event[event_key] #' + str(work.event[event_key])
                sift_tx += 17
                self.ax.text(    x[0]-x_sift,                       x[1]+y_sift_onomatope + sift_tx, text, size=10, color=work.color)

    # メモリー空間とのやりとりも含めたプロセス本体。
    def app_main4(self):
        # データ置き場を空にする
        self.aq_datas_list = []
        # 司書にデータ取得依頼とアップロード依頼をする
        self.memory_space.librarian_counter_down(self.appID, self.aq_datas_list)

        # 前回にメモリー空間からもらったデータをスキャンし、work 毎に整理する。
        # self.aq_datas_list は、Data_On_Memory形式データのリストになってるはず
        for data in self.aq_datas_list:
            if data.data_type == "event": # 2022Mar15追加。App5の出力
                #for event in data.data:
                for work in self.works :
                    if data.dataID == work.data_orgID: # ここでは、点群の評価としてのイベントのみ表示するので、一致しなければ破棄する
                        work.event = data.data
                        work.event_time = data.time_stamp
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
                        work.onomatopoeia = copy.copy(data.data.get("onomatopoeia",''))

        # 2秒以上更新されなかったら、そのworkを除去する
        remove_unused_works(2., self.works)

        plt.cla()
        for work in self.works:
            if work.data_type=="Single_point": 
                x = work.point.x
                # 枠サイズ確保のため
                self.max_scale(x, 0.)
                # 点の描画
                size = work.point.radius**2/7.
                self.ax.scatter(x[0],x[1],c = 'none', edgecolors=work.color, alpha=1.0, s=size, marker=work.mark)
                # 速度ベクトルの描画
                v = np.dot(1.,work.point.v)
                x_list = [x[0], x[0]+v[0]]
                y_list = [x[1], x[1]+v[1]]
                self.ax.plot(x_list, y_list, color = work.color)
                # IDの記述
                self.print_ID(work, x, work.point.radius, True)
            elif work.data_type=="Cloud":
                #if work.data_orgID == work.cloud_orgID: # 元のIDが検出できなかった場合はIDを表示する
                #print(work.data_orgID)
                x = work.point.centre
                r = math.sqrt(work.point.cloud_radius)
                # 枠サイズ確保のため
                self.max_scale(x, r)
                if work.data_orgID == work.cloud_orgID: # 元のIDが検出できなかった場合はIDを表示する
                    self.print_ID(work, x, r, True)
                else:
                    self.print_ID(work, x, r, False)

                following_point = False
                for x in work.point.xs:
                    if following_point:
                        self.ax.plot([x_pre[0],x[0]],[x_pre[1],x[1]],color=work.color)
                    x_pre = copy.deepcopy(x)
                    following_point = True

        # 枠サイズ確保のため
        if (self.max_x>-100000 and self.min_x<100000):
            self.ax.scatter(self.max_x,self.max_y,c = 'none', edgecolors='w', alpha=1.0, s=1, marker=',')
            self.ax.scatter(self.min_x,self.min_y,c = 'none', edgecolors='w', alpha=1.0, s=1, marker=',')
            #self.scale_x = self.max_x - self.min_x
            #self.scale_y = self.max_y - self.min_y
        self.fig.show()
        plt.pause(0.01)

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
        self.d_x = []
        self.d_rot = []
        self.d_mag = []
        for n in range(dim):
            self.d_x.append(0.0)
            self.d_rot.append(0.0)
            self.d_mag.append(0.0)
        #self.diff_x_scale = 0.0
        #self.diff_rot_scale = 0.0
        #self.diff_mag_scale = 0.0
        self.x_scale   = 0.0
        self.rot_scale = 0.0 
        self.mag_scale = 0.0

class Apply5:
    DISSMISSED = -1
    CATEGORY_FINDING = 0
    RANDOM_APPROACH = 1
    GRADIENT_APPROACH = 2
    def __init__(self, memory_space, idnet, projectID, categories):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.cats = categories
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time.time(),'additional_str':'Category mutching'})
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
        self.time_last = time.time()

    def gradient_approach(self, search_num_g, work, closest):
        # ここから、グラジェントに基づいて行う
        closest.trans_best_to_default()
        for i in range(search_num_g):
            diff_rot, diff_x, diff_mag = closest.closest_point_diff(work.resolution, work.rot_scale, work.x_scale, work.mag_scale)
            #for n in range(work.dim):
            #    work.diff_x_scale += abs(diff_x[n])
            #    work.diff_rot_scale += abs(diff_rot[n])
            #    work.diff_mag_scale += abs(diff_mag[n])
            for n in range(work.dim):
                #work.d_x[n]   = work.x_scale*diff_x[n]/work.diff_x_scale*20.*2.48
                #work.d_rot[n] = work.rot_scale*diff_rot[n]/work.diff_rot_scale*400.*2.49
                #work.d_mag[n] = work.mag_scale*diff_mag[n]/work.diff_mag_scale*200.*2.49
                work.d_x[n]   = diff_x[n]
                work.d_rot[n] = diff_rot[n]
                work.d_mag[n] = diff_mag[n]
            work.mat, is_converge = closest.closest_point_approach( work.resolution, work.mat, work.d_x, work.d_rot, work.d_mag)
            closest.trans_best_to_default()
            #print(mat, is_converge)
            if is_converge:
                return work.mat, is_converge
        work.counter2 += search_num_g
        return work.mat, is_converge

    # メモリー空間からのデータ収集。このルーティンは長いサンプリングタイムで実行する
    def app_main5_aq(self):
        time_now = time.time()
        if time_now > (self.time_last+self.sampling_time):
            self.time_last = time_now
            # データ置き場を空にする
            self.aq_datas_list = []
            # 司書にデータ取得依頼とアップロード依頼をする
            self.memory_space.librarian_counter_down(self.appID, self.aq_datas_list)

            # メモリー空間からもらったデータをスキャンし、探索中でなければ新しいworkを立ち上げて追加。
            for data in self.aq_datas_list:
                working = False
                for work in self.works:
                    if work.data_orgID == data.dataID:
                        working = True
                        break
                if working == False:
                    workID = self.idns.register_ID({"data_type":"work", "time_stamp": time.time()})
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
                        work.ref_points = self.cats.categories[work.categoryNum]["prototypes"][pnum]["data"]
                        work.snum = snum
                    else:
                        work.ref_points = self.cats.categories[work.categoryNum]["stereotypes"][snum]["data"]
                        work.prototypeID = self.cats.categories[work.categoryNum].prototypes[pnum]["dataID"]
                        work.pnum = pnum
                    work.closest = pcd.Closest(work.dim, work.data.xs, work.ref_points.xs )
                    cloud_tmp = pcd.PointCloud(2, xs=copy.deepcopy(work.data.xs), points=[])
                    cloud_tmp.points_line_sort()
                    work.x_scale   = cloud_tmp.dist_mean*0.1
                    if cloud_tmp.cloud_radius == 0.0:
                        work.mode = self.DISSMISSED
                    else:
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
            is_converge = False
            if work.mode == self.GRADIENT_APPROACH:
                search_num_g = 5
                work.mat, is_converge = self.gradient_approach(search_num_g, work, work.closest)
                if work.data_orgID>0:
                    print('In gradient approach ID: %d. mat=%.3f %s' % (work.data_orgID, work.mat, str(is_converge)))

            # 一致を検出したら、関係性登録
            # 表示用に自分で投げたデータを再び処理してしまうので、後で対策すること
            # matching_srsh_grad を小さくすれば、テスト用に何でもマッチングするようにできる
            matching_srsh_grad = 0.5
            if is_converge and work.mat >= matching_srsh_grad:
                if work.prototypeID >= 0:
                    #print("work ID= %d" % work.data_orgID)
                    #print("prototype ID=%d" % work.prototypeID)
                    #print("categoryNum =%d" % work.categoryNum)
                    #print("pnum = %d" % work.pnum)
                    relation_IDs = [work.prototypeID]
                    cat_data_type = self.cats.categories[work.categoryNum]["prototypes"][work.pnum]["data_type"]
                else:
                    relation_IDs = [] 
                    cat_data_type = self.cats.categories[work.categoryNum]["stereotypes"][work.snum]["data_type"]
                relation_property_dat = {"data_type": work.data_orgType, "projectID": self.projectID, "appID": self.appID, "workID": work.workID, "relation_IDs": relation_IDs}
                self.idns.register_relation(work.data_orgID,  work.categoryID, work.mat, "matched", relation_property_dat)
                # 一致検出したら、そのプロトタイプorステレオタイプをメモリー空間に投げる
                property_dat = {"data_type":  "point_cloud_large_num", \
                                "time_stamp": time.time(), \
                                "projectID":  self.projectID, \
                                "appID":      self.appID,  \
                                "workID":     work.workID, \
                                "relation_IDs":   [work.data_orgID, work.categoryID], \
                                "additional_str": "matched category"}
                newID = self.idns.register_ID(property_dat)
                data_on_memory = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, newID)
                data_on_memory.data_type = cat_data_type
                data_on_memory.time_stamp = time.time()
                data_on_memory.participant_IDs = [work.data_orgID] + relation_IDs
                #print(data_on_memory.data)
                #sift = pcd.cloud_centre(work.data.xs, 2)
                #sift_ref = pcd.cloud_centre(work.ref_points.xs, 2)
                #sift[0] = sift[0]-sift_ref[0]
                #sift[1] = sift[1]-sift_ref[1]
                #print(sift)
                #new_xs = pcd.shift_cloud(2, work.ref_points.xs, sift)
                new_xs = work.closest.coordinate_antitransformation(work.ref_points.xs)
                data_on_memory.data = pcd.PointCloud( work.ref_points.dim, new_xs, points=[])
                upload_data_list.append(copy.copy(data_on_memory))    # 
                print("Category matching obtained for ID=%d" % work.data_orgID)
                # さらに評価値をメモリー空間に投入。暫定的に"onomatopoeia"を使っている
                data = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.data_orgID)
                data.data_type = "evaluation"
                category_txt = "Category ID is %-2d" % work.categoryID
                data.data = {"onomatopoeia": category_txt}
                data.time_stamp = time.time()
                data.work_origin_IDs = [work.work_origin_ID]
                upload_data_list.append(data)
                work.mode = self.CATEGORY_FINDING   # また次のカテゴリーの探索に入る
            elif work.counter2 > work.count2_max:
                # タイムアップでカテゴリーを替えてやり直し
                work.mode = self.CATEGORY_FINDING

        # 20秒以上更新されなかったら、そのworkを除去する
        remove_unused_works(20., self.works)

        # 司書にアップロード依頼をする
        self.memory_space.librarian_counter_up(self.appID, upload_data_list)
#########
                
####################################
# 認知言語学っぽく、少数点群の各点の動きを評価するアプリ
# ここで、point は Point_F クラスの点
class Work_in_app6:
    filter_time = 0.7 #sec
    def __init__(self, time_org, point, ID, idns, work_origin_ID):
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time.time()})            # このアプリでは、単点毎にworkを生成する
        self.time_org = time_org        # 最新入力データのタイムスタンプ
        self.data_orgID = ID            # 入力データのID
        self.fieldID = 0                # 入力データのものを継承する（例）が、初期化には入れない
        self.work_origin_ID = work_origin_ID
        self.cogObj = cog.Detected_Object(point, self.filter_time)
        self.cogObj.STANDARD_SPEED = 10.0 # 今回の速度スケールに合わせたモノ
        self.cogObj.A_STANDARD = 6.     # 今回の速度スケールに合わせたモノ
        
class Apply6:
    def __init__(self, memory_space, idnet, projectID):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time.time(),'additional_str':'Cognitive Lingistics for single point'})
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
                    # 新しい点を検出したら新しいworkを立ち上げ、新しい時系列多数点群を創成する
                    new_work = Work_in_app6(data.time_stamp, point, point.ID, self.idns, data.workID)
                    new_work.fieldID = data.fieldID
                    self.works.append(new_work)
                    # work同士の関係性の記述必要か？
                    relation_property_dat = {"data_type": "work", "projectID": self.projectID, "appID": self.appID, "workID": new_work.workID, "relation_IDs": []}
                    self.idns.register_relation(data.workID, new_work.workID,    1.0, "processed", relation_property_dat)
        # 3秒以上更新されなかったら、そのworkを除去する
        remove_unused_works(3., self.works)
                    
        upload_data_list = []
        for work in self.works : # work 毎に処理を走らせる
            # データ処理本体
            work.cogObj.objects_kinesis(work.time_org)
            # アップロードする新しいデータ作成
            if work.cogObj.Attention > 0:
                data = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.data_orgID)
                data.data = {"Punctuality": work.cogObj.Punctuality, \
                            "AntiPunctuality": work.cogObj.AntiPunctuality, \
                            "Kinesis": work.cogObj.Kinesis, \
                            "Volitionality": work.cogObj.Volitionality}
                data.Attention_self = work.cogObj.Attention
                data.time_stamp = work.cogObj.Attention_time
                data.data_type = "event"
                data.participant_IDs = [work.data_orgID]
                data.work_origin_IDs = [work.work_origin_ID]
                upload_data_list.append(data)
        
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

########################################
if __name__ == "__main__":
    idnet = ids.ID_NetWork()
    memory_space = memory.Memory_space()
    projectID = idnet.register_ID({"data_type":"project","time_stamp": time.time()}) # 本当は、それなりのプロセスを経て新規プロジェクト生成
    categories = cat.Categories(idnet, projectID, "AImind5_categories") # 2022Feb4追加、カテゴリー

    ########### 「の」の字の多数点群をサンプルカテゴリーとして登録。本来はアプリでやる
    id_nonoji = idnet.register_ID({"data_type":"category","time_stamp": time.time()})
    nonoji_data = pcd.PointCloud(2, xs=cat.SAMPLE_CLOUD, points=[])
    nonoji_type = { \
            "data_type":  "point_cloud_large_num", \
            "dataID":     id_nonoji, \
            "data":       nonoji_data,  \
            "time_stamp": time.time(), \
            "appID":     0, \
            "workID":    0,  \
            "stength":    1.0}
    category_data = {"ID": id_nonoji, "prototypes": [nonoji_type], "stereotypes": []}
    categories.register_category(category_data, "no no ji")
    ######## サンプルカテゴリー修了

    apply1 = Apply1(memory_space, idnet, projectID)
    apply2 = Apply2(memory_space, idnet, projectID)
    apply3 = Apply3(memory_space, idnet, projectID)
    apply4 = Apply4(memory_space, idnet, projectID)
    apply5 = Apply5(memory_space, idnet, projectID, categories)
    apply6 = Apply6(memory_space, idnet, projectID)
    moni = Monitor(idnet)

    while(True): # 本当は、タイマー機能でやる。後ほど
        memory_space.memory_space_manerger()
        memory_space.librarian_swapper_before_scanner()
        memory_space.librarian_scanner()
        memory_space.librarian_swapper_after_scanner()

        status = apply1.app_main1()
        apply2.app_main2()
        apply3.app_main3()
        apply4.app_main4()
        apply5.app_main5_aq()
        apply5.app_main5_infer()
        apply6.app_main6()

        moni.monitor(memory_space)

        if status == False :
            break

    apply1.apply1_end()
        
        

