import object_detection_ids3 as obj
import point_cloud5 as pcd
import memory_space2 as memory
import id_network as ids
import copy
import time
import numpy as np

##########################################################################
# memory_space と、id_network、それに点群を駆使するアルゴリズムの初試作

##########################################################
# Karaageさんのobject_detectionを処理するアプリ
# 物体検出のリストを、少数点群としてポスト。
class Apply1:
    def __init__(self, memory_space, idnet, projectID):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time.time()})
        self.workID = idnet.register_ID({"data_type":"work","time_stamp": time.time()})
        self.fieldID = idnet.register_ID({"data_type":"field","time_stamp": time.time()})
        self.dataID = idnet.register_ID({"data_type":"point_cloud_small_num","time_stamp": time.time()})
        self.project = {"data_type": "point", "project_ID": projectID, "app_ID": self.appID, "work_ID": self.workID}
        #self.aq_datas_list = []     # メモリ空間から得たデータの置き場・・・このアプリでは不要
        #self.works = []
        self.data_aqcuire_conditions = [] # データ取得条件のリスト
        self.writer = obj.object_detection_init()
        self.object_list_previous = []
        self.status = False     # ESCキーが押されたことの判定
        self.point_cloud = pcd.PointCloud(2, xs=[], points=[])
        self.data_on_memory = memory.Data_On_Memory(projectID, self.appID, self.workID, self.fieldID, self.dataID)
        self.data_on_memory.work_origin_IDs = [self.workID]

    def app_main1(self):
        # Karaageさんのobject_detectionを走らせる
        self.status, self.object_list_previous = obj.object_detection(self.writer, self.idns, self.project, self.object_list_previous, last_time=time.time())
        new_points = []
        participant_IDs = []
        for object in self.object_list_previous:
            participant_IDs.append(object["ID"])
            # 2022Jan24 消えた点も保持してしまうので仕様変更
            #not_found = True
            #for point in self.point_cloud.points:
            #    if object['ID'] == point.ID:
            #        point.x = [object['x'],  object['y']] 
            #        point.v = [object.get('Vx',0.0), object.get('Vy',0.0)]
            #        point.radius = (object['size_y']+object['size_y'])/2.
            #        not_found = False
            #if not_found:
            #    new_point = pcd.Point_F(2)
            #    new_point.ID = object['ID']
            #    new_point.x = [object['x'],  object['y']] 
            #    new_point.v = [object.get('Vx',0.0), object.get('Vy',0.0)]
            #    new_point.radius = (object['size_y']+object['size_y'])/2.
            #    new_points.append(new_point)
            new_point = pcd.Point_F(2)
            new_point.ID = object['ID']
            new_point.x = [object['x'],  700.-object['y']] 
            new_point.v = [object.get('Vx',0.0), -object.get('Vy',0.0)]
            new_point.radius = (object['size_y']+object['size_y'])/2.
            new_points.append(new_point)
        #self.point_cloud.points += new_points
        self.point_cloud.points = new_points

        self.data_on_memory.data_type = "point_cloud_small_num"
        self.data_on_memory.participant_IDs = participant_IDs
        self.data_on_memory.time_stamp = time.time()   # タイムスタンプ
        self.data_on_memory.data = self.point_cloud               
        upload_data_list = [self.data_on_memory]    # このアプリでは、点群1つだけをアップロードするので、このような形になる

        # このアプリはメモリー空間からのデータ取得をしない。よって、データ置き場は常に空
        aq_datas_list = []
        # 司書にデータ取得依頼とアップロード依頼をする
        memory_space.librarian_counter(self.appID, self.data_aqcuire_conditions, upload_data_list, aq_datas_list)
        return self.status

    def apply1_end(self):
        obj.object_detection_end(self.writer)

####################################
# 検出物体のリストから、時系列データを生成するアプリ
# 元データが少数点群なので、一点づつ取り出し、その時系列を多数点群として生成する
# ここで、point は Point_F クラスの点
class Work_in_app2:
    def __init__(self, time_org, dim, ID, x, idns):
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time.time()})            # このアプリでは、単点毎にworkを生成する
        self.time_org = time_org        # 最新入力データのタイムスタンプ
        self.data_orgID = ID            # 入力データのID
        self.fieldID = 0                # 入力データのものを継承する（例）が、初期化には入れない
        #self.data_org = data_org        # 入力データ
        #self.data_created = []          # 生成した手持ちデータのリスト
        self.points = pcd.PointCloud(dim, xs=[x], points=[])
        self.points.ID = idns.register_ID({"data_type":"time_series_point_cloud_small_num","time_stamp": time_org})

class Apply2:
    def __init__(self, memory_space, idnet, projectID):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time.time()})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
        self.data_aqcuire_conditions = [] # データ取得条件のリスト

    # メモリー空間とのやりとりも含めたプロセス本体。
    def app_main2(self):
        # 前回にメモリー空間からもらったデータをスキャンし、work 毎に整理する。
        # self.aq_datas_list は、Data_On_Memory形式データのリストになってるはず
        for data in self.aq_datas_list:
            for point in data.data.points: # データが少数点群である前提
                x = copy.copy(point.x)
                x.append(data.time_stamp) # 次元をオーバーした所を時刻とする事で、時系列データとする
                new_work_created = True
                for work in self.works :
                    if point.ID == work.data_orgID: # 現有ID一致しなければ、new_work_createdがＯＮ・・・一つの例
                        new_work_created = False
                        work.points.xs.append(x) # 入力pointは点で、work.pointsは多数点群。注意
                        work.time_org = data.time_stamp
                if new_work_created :
                    # 新しい点を検出したら新しいworkを立ち上げ、新しい時系列多数点群を創成する
                    new_work = Work_in_app2(data.time_stamp, point.dim, point.ID, x, self.idns)
                    new_work.fieldID = data.fieldID
                    self.works.append(new_work)
                    relation_property_dat = {"data_type": "point", "project_ID": self.projectID, "app_ID": self.appID, "work_ID": new_work.workID}
                    self.idns.register_relation(point.ID, new_work.points.ID, 1.0, "processed", relation_property_dat)
                    
        upload_data_list = []
        for work in self.works : # work 毎に処理を走らせる
            # データ処理本体
            # アップロードする新しいデータ作成
            data = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.points.ID)
            #print(work.points.xs) # 動作確認
            data.data = work.points
            data.data_type = "time_series_point_cloud_large_num"
            upload_data_list.append(data)
        
        # データの要求条件作成。ディクショナリのリストである
        # ここで、"project_ID" と、"app_ID"(自分自身のID) の2つは絶対に必要なキーなので忘れないように
        self.data_aqcuire_conditions = [{ "aq_data_types": ["point_cloud_small_num"], "project_ID": self.projectID, "app_ID": self.appID}]

        # データ置き場を空にする
        self.aq_datas_list = []
        # 司書にデータ取得依頼とアップロード依頼をする
        memory_space.librarian_counter(self.appID, self.data_aqcuire_conditions, upload_data_list, self.aq_datas_list)

####################################
# 検出物体のリストから、各物体の動きを評価するアプリ
# 元データが少数点群なので、一点づつ取り出し、評価
# ここで、point は Point_F クラスの点
# 作成途上 2022Jan24
import fourier_trans as flt
class Work_in_app3:
    k = 60.
    FREQUENCY_LIST =  [10./k, 12.5/k, 16.0/k, 20./k, 25./k, 30./k,  40./k,  50./k, 60./k]
    m = 0.5
    decay_time_list = [50.*m, 40.*m,  31.*m,  25.*m, 20.*m, 16.7*m, 12.5*m, 10.*m, 8.3*m]
    filt_time_list =  [1., 3., 10., 30., 100., 300., 1000., 3000., 10000.]

    def __init__(self, time_org, dim, ID, x, idns):
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time.time()})            # このアプリでは、単点毎にworkを生成する
        self.time_org = time_org        # 最新入力データのタイムスタンプ
        self.data_orgID = ID            # 入力データのID
        self.fieldID = 0                # 入力データのものを継承する（例）が、初期化には入れない
        # 作成途上
        self.flx = flt.Fourier(self.FREQUENCY_LIST, self.decay_time_list, time_new=time.time())
        self.fly = flt.Fourier(self.FREQUENCY_LIST, self.decay_time_list, time_new=time.time())
        self.plx = flt.Impulse(self.filt_time_list, filt_ratio=5.0, time_new=time.time())
        self.ply = flt.Impulse(self.filt_time_list, filt_ratio=5.0, time_new=time.time())

class Apply3:
    def __init__(self, memory_space, idnet, projectID):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time.time()})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
        self.data_aqcuire_conditions = [] # データ取得条件のリスト

    # メモリー空間とのやりとりも含めたプロセス本体。
    def app_main3(self):
        # 前回にメモリー空間からもらったデータをスキャンし、work 毎に整理する。
        # self.aq_datas_list は、Data_On_Memory形式データのリストになってるはず
        for data in self.aq_datas_list:
            for point in data.data.points: # データが少数点群である前提
                x = copy.copy(point.x)
                #x.append(data.time_stamp) # 次元をオーバーした所を時刻とする事で、時系列データとする
                new_work_created = True
                for work in self.works :
                    if point.ID == work.data_orgID: # 現有ID一致しなければ、new_work_createdがＯＮ・・・一つの例
                        new_work_created = False
                        work.time_org = data.time_stamp
                        # フーリエ処理プロセス
                        work.flx.fourier_trans(imput=x[0], time_now=data.time_stamp)
                        work.fly.fourier_trans(imput=x[1], time_now=data.time_stamp)
                        work.plx.impulse(imput=x[0], time_now=data.time_stamp)
                        work.ply.impulse(imput=x[1], time_now=data.time_stamp)
                if new_work_created :
                    # 新しい点を検出したら新しいworkを立ち上げ、新しいフーリエ処理プロセスを開始する
                    new_work = Work_in_app3(data.time_stamp, point.dim, point.ID, x, self.idns)
                    new_work.fieldID = data.fieldID
                    self.works.append(new_work)
                    #relation_property_dat = {"data_type": "points", "project_ID": self.projectID, "app_ID": self.appID, "work_ID": new_work.workID}
                    #self.idns.register_relation(point.ID, new_work.points.ID, 1.0, "processed", relation_property_dat)
                    new_work.flx.fourier_trans(imput=x[0], time_now=data.time_stamp)
                    new_work.fly.fourier_trans(imput=x[1], time_now=data.time_stamp)
                    new_work.plx.impulse(imput=x[0], time_now=data.time_stamp)
                    new_work.ply.impulse(imput=x[1], time_now=data.time_stamp)
        
        RETAIN_TIME = 1.    # 更新されないデータは除去
        for n in reversed(range(len(self.works))):
            if time.time() > self.works[n].time_org+RETAIN_TIME :
                del self.works[n]

        upload_data_list = []
        for work in self.works : # work 毎に処理を走らせる
            # データ処理本体
            # アップロードする新しいデータ作成
            data = memory.Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.data_orgID)
            # 作成途上 
            #data.data = work.flx.components[2]['Real']**2 + work.flx.components[2]['Imagin']**2
            x0 = int(work.flx.components[0]['Real']**2 + work.flx.components[0]['Imagin']**2) # フンワフンワ
            x1 = int(work.flx.components[1]['Real']**2 + work.flx.components[1]['Imagin']**2) # フワフワ
            x2 = int(work.flx.components[2]['Real']**2 + work.flx.components[2]['Imagin']**2) # ユラユラ
            x3 = int(work.flx.components[3]['Real']**2 + work.flx.components[3]['Imagin']**2) # ユラユラ2
            x4 = int(work.flx.components[4]['Real']**2 + work.flx.components[4]['Imagin']**2) # フラフラ
            x5 = int(work.flx.components[5]['Real']**2 + work.flx.components[5]['Imagin']**2) # フラフラ2
            x6 = int(work.flx.components[6]['Real']**2 + work.flx.components[6]['Imagin']**2) # ぐらぐら
            x7 = int(work.flx.components[7]['Real']**2 + work.flx.components[7]['Imagin']**2) # ぐらぐら2
            x8 = int(work.flx.components[8]['Real']**2 + work.flx.components[8]['Imagin']**2) # ブルブル
            # 動作確認のための表示文字列
            txt = 'ID:' + str(work.data_orgID)+' '+str(x0) +' '+str(x1)+' '+str(x2)+' '+str(x3)+' '+str(x4)+' '+str(x5)+' '+str(x6)+' '+str(x7)+' '+str(x8)
            if work.data_orgID==9:
                print(txt)
            ### 動作確認コード end ###
            data.data_type = "evaluation"
            fl_min = 20. # 本来はスケールなどから決定すべき
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
            data.data = {"onomatopoeia": onomatopoeia}
            data.time_stamp = time.time()
            upload_data_list.append(data)
        
        # データの要求条件作成。ディクショナリのリストである
        # ここで、"project_ID" と、"app_ID"(自分自身のID) の2つは絶対に必要なキーなので忘れないように
        self.data_aqcuire_conditions = [{ "aq_data_types": ["point_cloud_small_num"], "project_ID": self.projectID, "app_ID": self.appID}]

        # データ置き場を空にする
        self.aq_datas_list = []
        # 司書にデータ取得依頼とアップロード依頼をする
        memory_space.librarian_counter(self.appID, self.data_aqcuire_conditions, upload_data_list, self.aq_datas_list)

####################################
# メモリー空間の情報をグラフィックに画き出すアプリ
# これはメモリー空間から読む一方で、書き込みをしない
# 作成途上 2022Jan25
from matplotlib import pylab as plt
class Work_in_app4:
    def __init__(self, time_org, dim, ID, point, idns, color):
        self.workID = idns.register_ID({"data_type":"work", "time_stamp": time.time()})            # このアプリでは、単点毎にworkを生成する
        self.time_org = time_org        # 最新入力データのタイムスタンプ
        self.data_orgID = ID            # 入力データのID
        self.fieldID = 0                # 入力データのものを継承する（例）が、初期化には入れない
        self.point = point
        self.onomatopoeia = ''
        self.color = color #'red'
        self.mark = 'o'

class Apply4:
    def __init__(self, memory_space, idnet, projectID):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time.time()})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
        self.data_aqcuire_conditions = [] # データ取得条件のリスト
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.fig.show()
        self.max_x = -200000
        self.min_x = 200000
        self.max_y = -200000
        self.min_y = 200000
        self.scale_x = 600
        self.scale_y = 400

    def colors_list(self, id):
        colors = ['red', 'green', 'blue', 'orange', 'cyan', 'Magenta', 'gold']
        cid = id % len(colors) # % は剰余演算子
        color = colors[cid]
        return color

    # メモリー空間とのやりとりも含めたプロセス本体。
    def app_main4(self):
        # 前回にメモリー空間からもらったデータをスキャンし、work 毎に整理する。
        # self.aq_datas_list は、Data_On_Memory形式データのリストになってるはず
        for data in self.aq_datas_list:
            if data.data_type == "point_cloud_small_num":
                for point in data.data.points: # データが少数点群である前提
                    new_work_created = True
                    for work in self.works :
                        if point.ID == work.data_orgID: # 現有ID一致しなければ、new_work_createdがＯＮ・・・一つの例
                            new_work_created = False
                            work.point = copy.copy(point)
                            work.time_org = data.time_stamp
                    if new_work_created :
                        # 新しい点を検出したら新しいworkを立ち上げ、
                        color = self.colors_list(point.ID)
                        new_work = Work_in_app4(data.time_stamp, point.dim, point.ID, copy.copy(point), self.idns, color)
                        new_work.fieldID = data.fieldID
                        new_work.point = copy.copy(point)
                        self.works.append(new_work)
        for data in self.aq_datas_list:
            if data.data_type == "evaluation":
                for work in self.works :
                    if data.data_ID == work.data_orgID: 
                        work.onomatopoeia = copy.copy(data.data.get("onomatopoeia",''))
                        #work.time_org = data.time_stamp

        RETAIN_TIME = 1.    # 更新されないデータは除去
        for n in reversed(range(len(self.works))):
            if time.time() > self.works[n].time_org+RETAIN_TIME :
                del self.works[n]

        plt.cla()
        for work in self.works:
            if work.time_org > (time.time() - 0.7): # 念のための処置
                x = work.point.x
                # 枠サイズ確保のため
                self.max_x = max(self.max_x, x[0])
                self.min_x = min(self.min_x, x[0])
                self.max_y = max(self.max_y, x[1])
                self.min_y = min(self.min_y, x[1])
                # 点の描画
                size = work.point.radius**2/7.
                self.ax.scatter(x[0],x[1],c = 'none', edgecolors=work.color, alpha=1.0, s=size, marker=work.mark)
                # 速度ベクトルの描画
                v = np.dot(1.,work.point.v)
                left = np.array([x[0], x[0]+v[0]])
                height = np.array([x[1], x[1]+v[1]])
                self.ax.plot(left, height, color = work.color)
                # ＩＤの記述
                text = 'ID:' + str(work.data_orgID)
                x_sift = self.scale_x/60
                y_sift = self.scale_y/4000
                self.ax.text(x[0]-x_sift,max(self.min_y-y_sift*500, x[1]-y_sift*(work.point.radius+100)), text, size=10, color=work.color)
                self.ax.text(x[0]-x_sift,                           x[1]+y_sift*(work.point.radius+100), work.onomatopoeia, size=10, color=work.color)
        # 枠サイズ確保のため
        if (self.max_x>-100000 and self.min_x<100000):
            self.ax.scatter(self.max_x,self.max_y,c = 'none', edgecolors='w', alpha=1.0, s=1, marker=',')
            self.ax.scatter(self.min_x,self.min_y,c = 'none', edgecolors='w', alpha=1.0, s=1, marker=',')
            self.scale_x = self.max_x - self.min_x
            self.scale_y = self.max_y - self.min_y
        self.fig.show()
        plt.pause(0.01)

        # データの要求条件作成。ディクショナリのリストである
        # ここで、"project_ID" と、"app_ID"(自分自身のID) の2つは絶対に必要なキーなので忘れないように
        self.data_aqcuire_conditions = [{ "aq_data_types": ["point_cloud_small_num"], "project_ID": self.projectID, "app_ID": self.appID},\
                                        { "aq_data_types": ["evaluation"], "project_ID": self.projectID, "app_ID": self.appID}]

        # データ置き場を空にする
        self.aq_datas_list = []
        # 司書にデータ取得依頼とアップロード依頼をする
        memory_space.librarian_counter(self.appID, self.data_aqcuire_conditions, [], self.aq_datas_list)


if __name__ == "__main__":
    idnet = ids.ID_NetWork()
    memory_space = memory.Memory_space()
    projectID = idnet.register_ID({"data_type":"project","time_stamp": time.time()}) # 本当は、それなりのプロセスを経て新規プロジェクト生成

    apply1 = Apply1(memory_space, idnet, projectID)
    apply2 = Apply2(memory_space, idnet, projectID)
    apply3 = Apply3(memory_space, idnet, projectID)
    apply4 = Apply4(memory_space, idnet, projectID)

    while(True): # 本当は、タイマー機能でやる。後ほど
        memory_space.memory_space_manerger()
        memory_space.librarian_swapper_before_scanner()
        memory_space.librarian_scanner()
        memory_space.librarian_swapper_after_scanner()

        status = apply1.app_main1()
        apply2.app_main2()
        apply3.app_main3()
        apply4.app_main4()

        if status == False :
            break

    apply1.apply1_end()
        
        

