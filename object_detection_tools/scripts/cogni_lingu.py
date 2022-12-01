# 認知言語学の知見を使って関係性および関係性を評価するモノ
# 2022Feb21 test8.py から抽出
#from matplotlib import pylab as plt
from matplotlib import pyplot as plt
#import matplotlib.animation as animation
import numpy as np
import math
import random
import copy
import json
import time
import point_cloud6 as pcd
import id_network2 as ids

ID_NOT_ASSIGNED = -1
COGNI_SRESH = 0.5
AVOID_ZERO = 0.00000000001

# ベクトルの絶対値を1にする
# np.linalg.norm はベクトルの絶対値
# 入力xはnumpyarrayでること
def normalize(x, minx = AVOID_ZERO):
    return(np.array(x)/max(np.linalg.norm(x), minx))

#############################################################
# 2022Feb21 再構築
# Point_F オブジェクトに認知言語学パラメータ等を加えたモノ
class Detected_Object:
    def __init__(self, point, filter_time):
        self.filter_time = filter_time
        self.point = point  # Point_F オブジェクト
        self.Vfilt = 0.*np.array(point.v)
        self.V_last = self.Vfilt
        self.Acc = 0.*self.Vfilt       # 加速度 単に次元を合わせるために速度かけるゼロ
        self.Vnor = normalize(self.Vfilt)   # 正規化した速度ベクトル
        self.Anor = normalize(self.Acc)     # 正規化した加速度ベクトル
        self.VdotAnor = np.dot(self.Vnor, self.Anor)    # 正規化した速度と正規化した加速度ベクトルの内積（向きを変える運動すると値が小さくなる。正なら加速、負なら減速）
        #self.ID = ID_NOT_ASSIGNED
        #self.Face_x = 0.0     #2019/Sep/24 
        #self.Face_y = 0.0     #2019/Sep/24 
        #self.Size = 0.0       #2019/Sep/24 
        #self.Body_boundary = []      #2019/Oct/3 ブツの領域を言う
        # Body_boundary の構造は、{"Centre": {"x": 値, "y": 値}, "Radius": 値} のリストで、Ground の g_elements と同じ
        self.Aspect = ID_NOT_ASSIGNED  # 2019/Sep/24 認知言語学でいう所の相(p.101)
        self.Category = ID_NOT_ASSIGNED   #2019/Sep/24
        #self.Vx = 0.0
        #self.Vy = 0.0
        #self.dVx = 0.0
        #self.dVy = 0.0
        self.Attention = 0
        self.Attention_time = 0.0   # Attention 値を増やした時のタイムスタンプ
        #self.continuity = False
        #self.ID_double = 0
        self.Punctuality = 0.0      # 瞬時性
        self.AntiPunctuality = 0.0  # アンチ瞬時性 動作性の急減
        self.Kinesis = 0.0          # 動作性
        self.Volitionality = 0.0    # 意図性。関係性に与えるものと、個体に与えるものが有る
        self.STANDARD_SPEED = 0.40 # m/sec
        self.A_STANDARD = 0.5 # m/sec2 加速度
        self.last_time = time.time()
        self.filter_ratio_min = 0.3 # 2022Apr6 追加


    # 検出物体の動作性を評価
    # 1．運動していること
    # 2．加速度運動していること
    # 3．それまでと違う変化があったこと
    # を検出。3の「変化」に対してAttentionを発出し、セグメンテーションの基本情報とするなどに使う
    # ここで時定数は重要で、異なる時定数のセットを多数用意すると良いだろう
    def objects_kinesis(self, time_now):
        filter_ratio = max(self.filter_ratio_min, min(1., (time_now - self.last_time)/self.filter_time))
        deltaT = max(AVOID_ZERO, time_now - self.last_time)
        self.Vfilt = (1. - filter_ratio)*self.Vfilt + filter_ratio*np.array(self.point.v)
        self.Acc = (1. - filter_ratio)*self.Acc + filter_ratio*(self.Vfilt - self.V_last)
        self.Vnor = normalize(self.Vfilt)
        self.Anor = normalize(self.Acc)
        self.VdotAnor = np.dot(self.Vnor, self.Anor)
        self.V_last = copy.copy(self.Vfilt)

        # 動作性。単に速度絶対値のSTANDARD_SPEEDとの比較
        kinesis_new = (1. - filter_ratio)*self.Kinesis + filter_ratio*math.log(np.linalg.norm(self.point.v)**2/(self.STANDARD_SPEED**2)+0.000001)
        if time_now > self.last_time:
            # 動作性が急増したら、瞬時性があると判断する・・・単に動作性の微分 負の値を排する変更2019Sep16
            self.Punctuality = (1. - filter_ratio)*self.Punctuality     + filter_ratio*max(0.,max(0.0, kinesis_new) - max(0.0, self.Kinesis))/deltaT
            # 動作性が急減したら、アンチ瞬時性があると判断する・・・単に動作性の微分 2022Mar15追加
            self.AntiPunctuality = (1. - filter_ratio)*self.AntiPunctuality + filter_ratio*max(0.,max(0.0, self.Kinesis) - max(0.0, kinesis_new))/deltaT
        self.Kinesis = kinesis_new
        # 意図性を、進行方向からずれた加速度として定義する事で、Punctualityと違う意味を与える。本来の言語学の意味とは別物
        volitionality_new = (1. - abs(np.linalg.norm(self.VdotAnor)))*np.linalg.norm(self.Acc)/self.A_STANDARD
        self.Volitionality = (1. - filter_ratio)*self.Volitionality + filter_ratio*volitionality_new

        # Kinesis以外のパラメータから、注意を発出
        if self.Volitionality > COGNI_SRESH or \
           self.Punctuality > COGNI_SRESH or \
           self.AntiPunctuality > COGNI_SRESH :
            self.Attention = max(self.Volitionality, self.Punctuality, self.AntiPunctuality)
            self.Attention_time = copy.copy(time_now)
            #print("Volitionality = %.2f" % self.Volitionality)
            #print("Punctuality = %.2f" % self.Punctuality)
            #print("AntiPunctuality = %.2f" % self.AntiPunctuality)
        else:
            self.Attention = 0

        self.last_time = copy.copy(time_now)

    # 新バージョン2022Mar24
    # 時間スケールと速度（kinesis）以外は無次元化した
    # filter_time に敏感に反応して感度が変わる→感度設定はこれで行う？ 2022Mar25
    # 無次元化した結果、明らかに直観と異なる判定をしてしまうので、当面使わない
    def objects_kinesis2(self, time_now):
        filter_ratio = min(1., (time_now - self.last_time)/self.filter_time)
        deltaT = max(AVOID_ZERO, time_now - self.last_time)
        self.Vfilt = (1. - filter_ratio)*self.Vfilt + filter_ratio*np.array(self.point.v)
        self.Acc = (1. - filter_ratio)*self.Acc + filter_ratio*(self.Vfilt - self.V_last)
        self.Vnor = normalize(self.Vfilt)
        Anor_new = (self.Vfilt - self.V_last)/max(np.linalg.norm(self.Vfilt)+AVOID_ZERO, np.linalg.norm(self.V_last))/deltaT 
        self.Anor = (1. - filter_ratio)*self.Anor + filter_ratio*np.array(Anor_new) # 定義を変えた。速度で規格化し、単位は/sec
        self.VdotAnor = np.dot(self.Vnor, normalize(self.Anor))
        #print(self.Anor)

        # 動作性。単に速度絶対値のSTANDARD_SPEEDとの比較
        kinesis_new = (1. - filter_ratio)*self.Kinesis + filter_ratio*math.log(np.linalg.norm(self.point.v)**2/(self.STANDARD_SPEED**2)+0.000001)
        self.Kinesis = kinesis_new
        # 規格化加速度をそのまま瞬時性とする＝単位は/sec
        self.Punctuality = (1. - filter_ratio)*self.Punctuality + filter_ratio*max(0.,np.linalg.norm(self.Anor))*0.1
        # アンチ瞬時性の定義も変更。急減速なのでこうなる
        self.AntiPunctuality = (1. - filter_ratio)*self.AntiPunctuality + filter_ratio*max(0., -np.dot(self.Anor, self.Vnor))*0.1
        # 意図性を、進行方向からずれた加速度として定義する事で、Punctualityと違う意味を与える。本来の言語学の意味とは別物
        volitionality_new = max(0., 1. - (self.VdotAnor)**2)*np.linalg.norm(self.Anor)*0.2
        self.Volitionality = (1. - filter_ratio)*self.Volitionality + filter_ratio*volitionality_new

        # Kinesis以外のパラメータから、注意を発出
        if self.Volitionality > COGNI_SRESH or \
           self.Punctuality > COGNI_SRESH or \
           self.AntiPunctuality > COGNI_SRESH :
            self.Attention = max(self.Volitionality, self.Punctuality, self.AntiPunctuality)
            self.Attention_time = copy.copy(time_now)
            #print("Volitionality = %.2f" % self.Volitionality)
            #print("Punctuality = %.2f" % self.Punctuality)
            #print("AntiPunctuality = %.2f" % self.AntiPunctuality)
        else:
            self.Attention = 0.
        self.V_last = copy.copy(self.Vfilt)
        self.last_time = copy.copy(time_now)


MIN_DISTANCE_TO_IDENTICAL_OBJECT = 0.7
TIME_RESOLUTION = 0.05 # sec 認知の時間分解能 
MIN_SPEED = 0.3 #m/sec
##############################################################################################################
#  関係性のクラス
# ここで、point の座標系はnumpy前提として作る
PERSONAL_SPACE = 2.0 # m この距離より小さい距離での相互作用は強いとする
SPACE_RESOLUTION = 0.2
VMIN = 0.001 # m/sec 速度の認知限界

class Relation:
    def calc_relative_motion(self):
        npxa = np.array(self.pointA.x)
        npxb = np.array(self.pointB.x)
        self.MeanX = (npxa + npxb)/2                                # np.array ２つの点の中央位置
        self.DeltaX = npxb - npxa                                   # Aから見た2点間位置相対ベクトル
        self.Distance = np.linalg.norm(self.DeltaX)                 # スカラー ２点間距離
        self.DeltaXnor = normalize(self.DeltaX)                     # 正規化した２点間位置相対ベクトル
        self.DeltaV = self.VAfilt - self.VBfilt                     # A基準の（Aが速い＝正）相対速度ベクトル
        self.DeltaVnor = normalize(self.DeltaV)                     # 正規化にた相対速度ベクトル
        self.MeanV = (self.VAfilt + self.VBfilt)/2.                 # 速度平均値
        self.DeltaVrelat = self.DeltaV/max(np.linalg.norm(self.MeanV), AVOID_ZERO) # 中心速度で正規化した相対速度
        self.MeanVnor = normalize(self.MeanV)                       # 正規化した平均速度ベクトル
        vanor = np.linalg.norm(self.VAfilt)
        vbnor = np.linalg.norm(self.VBfilt)
        self.VdotVnor = np.dot(self.VAfilt, self.VBfilt)/max(vanor, vbnor, AVOID_ZERO) # 正規化した速度の内積（運動の平行度合い）
        self.DeltaXDeltaVnor = np.dot(self.DeltaXnor, self.DeltaVnor) # 正規化した相対速度と正規化した相対位置ベクトルの内積（対抗具合、または離れる具合） 
        self.DeltaXMeanVnor = np.dot(self.DeltaXnor, self.MeanVnor) # 正規化した相対速度と正規化した平均速度の内積（ゼロなら並走、先行/後追い 判別）
        #print(self.DeltaXMeanVnor)
        # DeltaX の変化は DeltaV であると仮定する
        # 追う     ・・・VdotVnorが1に近く（並走）、DeltaXMeanVnorが-1に近い（Aが追う者で、後ろに付いている）
        # 続く     ・・・VdotVnorが1に近く（並走）、DeltaXMeanVnorが1に近い（Aが追われる者で、前にいる）
        # 並ぶ     ・・・ VdotVnorが1に近く（並走）、DeltaXMeanVnorがゼロに近い（横に並んでいる）
        # 遅れる   ・・・「追う、続く」に加えて、DeltaXDeltaVnorが負
        # 追いつく ・・・「追う、続く」に加えて、DeltaXDeltaVnorが正 ”Contact" がFalse->Trueの変化があればイベント
        # 会う（イベント）”Contact" が False -> True の変化
        # 別れ（イベント）”Contact" が True -> False の変化
        # 追い越す（イベント）・・・ VdotVnorが1に近く（並走）、DeltaXMeanVnorが、負->正 の変化（逆なら追い越される）
        # 妨害する ・・・ DeltaVの絶対値が正で、Bに「減速」が現れる

    # 上記の記述に従って認知するための行列。下記の state_vector で変換すると cogni_vector となる
    relation_matrix = np.array([\
#       VdotVnor,   DeltaXMeanVnor, 1-DeltaXMeanVnor, DeltaXDeltaVnor, Contact,  VdotVnorl,   DeltaXMeanVnorl, 1-DeltaXMeanVnorl, DeltaXDeltaVnorl, Contactl 
#       ぶつかり合う
        [-0.5,       0.,             0.,               0.5,            0.,        0.,          0.,              0.,                0.,               0.],\
#       避け合う
        [-0.5,       0.,             0.,              -0.5,            0.,        0.,          0.,              0.,                0.,               0.],\
#       追う、続く
        [0.5,        0.5,            0.,               0.,             0.,        0.,          0.,              0.,                0.,               0.],\
#       追われる
        [0.5,       -0.5,            0.,               0.,             0.,        0.,          0.,              0.,                0.,               0.],\
#       共に動く
        [1.,         0.,             0.,               0.,             0.,        0.,          0.,              0.,                0.,               0.],\
#       横に並ぶ
        [0.5,        0.,             0.5,              0.,             0.,        0.,          0.,              0.,                0.,               0.],\
#       付き添う
        [0.5,        0.,             0.,               0.,             0.5,       0.,          0.,              0.,                0.,               0.],\
#       遅れる
        [0.33,       0.33,           0.,              -0.33,           0.,        0.,          0.,              0.,                0.,               0.],\
#       追いつく
        [0.33,       0.33,           0.,               0.33,           0.,        0.,          0.,              0.,                0.,               0.],\
#       会う
        [0.,         0.,             0.,               0.,            0.5,        0.,          0.,              0.,                0.,              -0.5],\
#       別れる
        [0.,         0.,             0.,               0.,           -0.5,        0.,          0.,              0.,                0.,               0.5],\
#       追い越す
        [0.5,       0.25,            0.,               0.,             0.,        0.,        -0.25,             0.,                0.,               0.],\
#       追い越される
        [0.5,      -0.25,            0.,               0.,             0.,        0.,         0.25,             0.,                0.,               0.],\
        ])

    def cogni_vec2dict(self, cogni_vector):
        cogni_dict = {  "Hit":       cogni_vector[0], \
                        "Avoid":     cogni_vector[1], \
                        "Chase":     cogni_vector[2], \
                        "Chased":    cogni_vector[3], \
                        "Co_motion": cogni_vector[4], \
                        "Sideby":    cogni_vector[5], \
                        "Attend":    cogni_vector[6], \
                        "Behind":    cogni_vector[7], \
                        "Catchup":   cogni_vector[8], \
                        "Meet":      cogni_vector[9], \
                        "Apart":     cogni_vector[10], \
                        "Overtake":  cogni_vector[11], \
                        "Overtaken": cogni_vector[12]  }
        return cogni_dict

    def state_vector(self):
        vector = np.array([\
            self.VdotVnor,\
            self.DeltaXMeanVnor,\
            (1.-abs(self.DeltaXMeanVnor))**2,\
            self.DeltaXDeltaVnor,\
            self.Contact,  \
            self.VdotVnor_last,   \
            self.DeltaXMeanVnor_last, \
            (1.-abs(self.DeltaXMeanVnor_last))**2, \
            self.DeltaXDeltaVnor_last, \
            self.Contact_last])
        return vector

    def set_last(self):
        self.VdotVnor_last = copy.copy(self.VdotVnor)
        self.DeltaXMeanVnor_last = copy.copy(self.DeltaXMeanVnor)
        self.DeltaXDeltaVnor_last = copy.copy(self.DeltaXDeltaVnor)
        self.Contact_last = copy.copy(self.Contact)

    def __init__(self, objectA, objectB, filter_time):
        self.objectA = objectA          # Detected_Object クラス
        self.objectB = objectB          # Detected_Object クラス
        self.pointA = objectA.point     # Point_F クラス
        self.pointB = objectB.point     # Point_F クラス
        self.ID = ID_NOT_ASSIGNED
        #self.IDa = pointA.ID
        #self.IDb = pointB.ID
        #self.fieldID = fieldID
        #self.categoryIDs = []     
        #self.V = []         # np.array 上記の微分
        self.VAfilt = np.array(self.pointA.v)   # フィルターをかけた速度。初期値はフィルタなしで入れる
        self.VBfilt = np.array(self.pointB.v)   # フィルターをかけた速度。初期値はフィルタなしで入れる
        self.Contact = False
        self.calc_relative_motion()
        self.set_last()
        self.last_time = time.time()
        self.filter_time = filter_time
        #self.Vy = 0.0
        #self.dV = []
        #self.dVy = 0.0
        self.DeltaXV = 0.0  # スカラー 相対間距離の変化
        self.DeltaVV = 0.0  # スカラー 正規化した相対速度 
        #self.DeltaY = 0.0
        self.Attention = 0.0
        self.Attention_in_A = False # Linkageness検出のための個別イベント検出
        self.Attention_in_B = False # Linkageness検出のための個別イベント検出
        self.Attention_sh_A = 0.0   # Linkageness検出のための個別イベント検出閾値
        self.Attention_sh_B = 0.0   # Linkageness検出のための個別イベント検出閾値
        self.Attention_time = 0.    # Attention が立った時刻
        #self.continuity = False
        #self.ID_double = 0
        self.cogni_dict = { "Hit":       0.0, \
                            "Avoid":     0.0, \
                            "Chase":     0.0, \
                            "Chased":    0.0, \
                            "Co_motion": 0.0, \
                            "Sideby":    0.0, \
                            "Attend":    0.0, \
                            "Behind":    0.0, \
                            "Catchup":   0.0, \
                            "Meet":      0.0, \
                            "Apart":     0.0, \
                            "Overtake":  0.0, \
                            "Overtaken": 0.0  }
        self.relation = {\
#                   瞬時性        ・・・アルゴリズム未作成            
                    "Punctuality":  0.0,\
#                   動作性        ・・・アルゴリズム未作成 "Sideby" で仮置き  
                    "Kinesis":      0.0,\
#                   意図性        ・・・アルゴリズム未作成                      
                    "Volitionality": 0.0,\
#                   能動性         ・・・アルゴリズム未作成 "Chase" で仮置き
                    "Agency":       0.0,\
#                   受影性         ・・・アルゴリズム未作成 "Chased" で仮置き
                    "Affectedness": 0.0,\
#                   連動性（フカダ用語）
                    "Linkageness":  0.0,\
#                   一体性（フカダ用語）
                    "Unity":        0.0,\
#                   同一性（フカダ用語）・・・アルゴリズム未作成
                    "Identicality": 0.0,\
#                   類似性（フカダ用語）・・・アルゴリズム未作成
                    "Similarity":   0.0,\
                    "Time": time.time()}
        self.ATTENTION_RELEASE_TIME  = .20   # Attention を解除する時間
        self.TIME_LINKAGE = .10             # 連動性有りとみなすAttentionの時間差

    def linkage_evaluation(self, time_now):
        # 連動性 Linkageness （フカダ造語）を評価
        # ここで、過去にさかのぼるので、メモリ空間から引っ張ってきたattention付きアイテムに限って評価できる
        # ON/OFF 判定になっているが、ゆくゆくはアナログ値にしたい
        #print(self.objectA.Attention)
        if self.objectA.Attention > self.Attention_sh_A:
            #print("RelAtt")
            self.Attention_in_A = True
            self.objectA.Attention_time = time_now
        elif self.objectA.Attention == 0.0 \
            and self.objectA.Attention_time + self.ATTENTION_RELEASE_TIME< time_now:
            self.Attention_in_A = False
        if self.objectB.Attention > self.Attention_sh_B:
            #print("RelAtt")
            self.Attention_in_B = True
            self.objectB.Attention_time = time_now
        elif self.objectB.Attention == 0.0 \
            and self.objectB.Attention_time + self.ATTENTION_RELEASE_TIME < time_now:
            self.Attention_in_B = False
        if self.Attention_in_A and self.Attention_in_B:
            # Aが先に動く-> AのAttention_timeの方が古い＝小さい
            time_diff = self.objectB.Attention_time - self.objectA.Attention_time
            sig = np.sign(time_diff)
            if sig == 0.:
                sig = 1.
            #print("RelAtt A=%.2f, B=%.2f, diff=%1f" % (self.objectB.Attention_time, self.objectA.Attention_time, (np.sign(time_diff))))
            self.relation["Linkageness"] = max(0., (self.TIME_LINKAGE - abs(time_diff))/self.TIME_LINKAGE)*sig
        else:
            self.relation["Linkageness"] = 0.
        #print("link=%.2f" % self.relation["Linkageness"])

    # 連動性 Linkageness （フカダ造語）を評価
    # アナログ値にしたver.2
    # 連続でAttentionが立っていると、全部「linkageと判定してしまう。
    # やはり個々のイベント検出に閾値を設けて、その閾値を調整するしかない。
    # という事で、この関数は不採用
    def linkage_evaluation2(self, time_now):
        decay = min(1., max(0., 1. - (time_now - self.last_time)/self.ATTENTION_RELEASE_TIME))
        if self.Attention_in_A < self.objectA.Attention:
            self.objectA.Attention_time = time_now  # ピークを過ぎたところで時刻設定
        self.Attention_in_A = max(self.objectA.Attention, self.Attention_in_A*decay)
        if self.Attention_in_B < self.objectB.Attention:
            self.objectB.Attention_time = time_now  # ピークを過ぎたところで時刻設定
        self.Attention_in_B = max(self.objectB.Attention, self.Attention_in_B*decay)
        # Aが先に動く-> AのAttention_timeの方が古い＝小さい という理屈
        link_abs = max(self.Attention_in_A, self.Attention_in_B)
        sig = np.sign(self.objectB.Attention_time - self.objectA.Attention_time)
        if sig == 0.:
            sig = 1.
        self.relation["Linkageness"] = link_abs*sig
        #print("link=%.2f" % self.relation["Linkageness"])

    def relation_evaluation(self, time_now):
        self.relation["Time"] = time_now
        filter_ratio = min(1., (time_now - self.last_time)/self.filter_time)
        self.VAfilt = (1. - filter_ratio)*self.VAfilt + filter_ratio*np.array(self.pointA.v) # フィルターをかけた速度
        self.VBfilt = (1. - filter_ratio)*self.VBfilt + filter_ratio*np.array(self.pointB.v) # フィルターをかけた速度
        self.calc_relative_motion()
        cogni_vector = np.dot(self.relation_matrix, self.state_vector())
        max_val = (max(0., np.amax(cogni_vector)))**2
        cogni_dict = self.cogni_vec2dict(cogni_vector)
        self.relation["Unity"]  = (max(0., cogni_dict["Co_motion"]))**2
        self.relation["Agency"] = (max(0., cogni_dict["Chase"]))**2
        self.relation["Affectedness"] = (max(0., cogni_dict["Chased"]))**2
        self.relation["Kinesis"] = (max(0., cogni_dict["Sideby"]))**2
        self.set_last()

        if self.Distance < max(SPACE_RESOLUTION, self.pointA.radius + self.pointB.radius):
            self.Contact = True
            self.Attention_time = time_now
        else :
            self.Contact = False
            
        # 連動性 Linkageness （フカダ造語）を評価
        self.linkage_evaluation(time_now)

        max_val = max(max_val, self.relation["Linkageness"], float(self.Contact))
        if max_val > 0.3 :
            self.Attention = max_val
            self.Attention_time = time_now
        elif self.Attention_time + self.ATTENTION_RELEASE_TIME > time_now:
            self.Attention = 0.
        self.cogni_dict = cogni_dict
        self.last_time = time_now
        return cogni_dict

        


######################################################
#import matplotlib.animation as animation
from matplotlib.animation import ArtistAnimation
# テストプログラム
class Moving_Object:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.Vx = 0.0
        self.Vy = 0.0

# 世界の初期化
Xmin = 0.0 #c4
Xmax = 10.0 #c5
Ymin = 0.0 #c6
Ymax = 6.0 #c7
Vplus = 0.5 #c12
DAMAGE_RECOVER_TIME = 3.0   # 2021Dev15追加。追いつかれてダメージを受けてからの回復
Viblation_time = 0.06        # 2021Dec15追加。ダメージを受けた場合の振動周期
Viblation_amp = 7.          # 2021Dec15追加。ダメージを受けた場合の振動振幅

# Run and chase
object_list_chase = [{"x": 2.0, "y": 2.0, "Vx": 0.18, "Vy": -0.5,  "Vmax": 1.0, "Amax": 0.7, "Rand": 0.0, "color": 'blue',   "size": 50, "Damage": 0.0},
                     {"x": 7.0, "y": 4.0, "Vx": 0.43, "Vy": -0.9,  "Vmax": 1.5, "Amax": 0.7, "Rand": 0.5, "color": 'red',    "size": 40, "Damage": 0.0},
                     {"x": 5.0, "y": 1.0, "Vx": 0.18, "Vy": 0.06,  "Vmax": 0.5, "Amax": 0.5, "Rand": 3.0, "color": 'green',  "size": 30, "Damage": 0.0},
                     {"x": 4.0, "y": 5.0, "Vx": 0.043, "Vy": -0.1, "Vmax": 0.5, "Amax": 0.5, "Rand": 3.0, "color": 'yellow', "size": 20, "Damage": 0.0}]

# Pikopiko
object_list_piko = [{"x": 2.0, "y": 2.0, "Vx": 0.18, "Vy": -0.5,  "Vmax": 0.4, "Amax": 0.4, "Rand": .50, "color": 'blue',   "size": 50},
                    {"x": 7.0, "y": 4.0, "Vx": 0.43, "Vy": -0.9,  "Vmax": 0.3, "Amax": 0.3, "Rand": 1.0, "color": 'red',    "size": 40},
                    {"x": 5.0, "y": 1.0, "Vx": 0.18, "Vy": 0.06,  "Vmax": 0.4, "Amax": 0.5, "Rand": .50, "color": 'green',  "size": 30},
                    {"x": 4.0, "y": 5.0, "Vx": 0.043, "Vy": -0.1, "Vmax": 0.3, "Amax": 0.4, "Rand": 1.0, "color": 'yellow', "size": 20},
                    {"x": 1.5, "y": 3.0, "Vx": 0.13, "Vy": -0.2,  "Vmax": 0.4, "Amax": 0.3, "Rand": .50, "color": 'cyan',   "size": 20},
                    {"x": 4.0, "y": 3.1, "Vx": -0.1, "Vy": 0.12,  "Vmax": 0.3, "Amax": 0.5, "Rand": 1.0, "color": 'magenta', "size": 20},
                    {"x": 8.0, "y": 2.0, "Vx": 0.15, "Vy": -0.1,  "Vmax": 0.4, "Amax": 0.4, "Rand": .50, "color": 'violet', "size": 20},
                    {"x": 7.5, "y": 5.5, "Vx": 0.12, "Vy": 0.09,  "Vmax": 0.3, "Amax": 0.3, "Rand": 1.0, "color": 'tomato', "size": 20}]

#object_list = object_list_chase
object_list = object_list_piko
SAMPLING_TIME = 0.1 #sec
class World:
    def __init__(self, name):
        self.world_name = name
        self.objects = []
        self.piko_freq = 0.2*SAMPLING_TIME # ピコピコする動きの頻度
        self.piko_duration = 3. # ピコの継続時間（回数）
        self.piko_Y = 0.15
        object_item = Moving_Object() 
        for n in range(len(object_list)):
            object_item.x = object_list[n]["x"]
            object_item.y = object_list[n]["y"]
            object_item.Vx = object_list[n]["Vx"]
            object_item.Vy = object_list[n]["Vy"]
            object_item.Size = object_list[n]["size"]/100.0
            object_item.piko = False
            object_item.piko_time = 0.0
            self.objects.append(copy.copy(object_item))
        worldstate = [{"Time": 0.0}]
        self.world_state = worldstate[0]
        world_character_list = [{"Time_interval": SAMPLING_TIME}]
        self.world_character = world_character_list[0]

    #####################################
    # Run and chase 
    def world_change_run_and_chase(self):
        # 0 がA(Run)、1 がB(chase)
        # SAMPLING_TIME #c18
        AtoBx = self.objects[1].x -self.objects[0].x
        AtoBy = self.objects[1].y -self.objects[0].y
        AtoB2 = AtoBx**2 + AtoBy**2
        # 2021Dec15追加 追いついて攻撃
        if AtoB2 < 0.1:
            hit = True
        else:
            hit = False
        
        for n in range(len(object_list)):
            rand = object_list[n]["Rand"]
            Amax = object_list[n]["Amax"]
            Vmax = object_list[n]["Vmax"]
            dVmax = Amax*SAMPLING_TIME
            # 2021Dec15追加、ダメージ回復
            object_list[n]["Damage"] *= 1. - SAMPLING_TIME/DAMAGE_RECOVER_TIME
            if n == 0 or n == 1:
                AtoBxnorm = AtoBx/math.sqrt(AtoBx**2 + AtoBy**2)
                AtoBynorm = AtoBy/math.sqrt(AtoBx**2 + AtoBy**2)
                Vxtarget = -AtoBxnorm*object_list[n]["Vmax"]
                Vytarget = -AtoBynorm*object_list[n]["Vmax"]
                rand_sx = -AtoBynorm
                rand_sy = AtoBxnorm
                if n == 0: # 中央に逃げ込もうとするバイアス
                    Vxtarget = 0.8*Vxtarget + 0.4*Vmax*((Xmax+Xmin)/2 - self.objects[n].x)/(Xmax+Xmin)
                    Vytarget = 0.8*Vytarget + 0.4*Vmax*((Ymax+Ymin)/2 - self.objects[n].y)/(Ymax+Ymin)
                    if hit :
                        object_list[0]["Damage"] = 1.
                    if object_list[0]["Damage"] > 0.4 :
                        Vxtarget *= 0.25
                        Vytarget *= 0.25
                        #Vxtarget +=  Vytarget*Viblation_amp*math.sin(self.world_state["Time"]/Viblation_time)
                        #Vytarget += -Vxtarget*Viblation_amp*math.sin(self.world_state["Time"]/Viblation_time)
                elif n == 1: # 周囲から追い立てようとするバイアス
                    Vxtarget = 0.8*Vxtarget - 0.4*Vmax*((Xmax+Xmin)/2 - self.objects[n].x)/(Xmax+Xmin)
                    Vytarget = 0.8*Vytarget - 0.4*Vmax*((Ymax+Ymin)/2 - self.objects[n].y)/(Ymax+Ymin)
                    # 2021Dec15追加。0番にダメージを与えたら一旦離脱
                    if object_list[0]["Damage"] > 0.5:
                        Vxtarget *= -0.4
                        Vytarget *= -0.4
            else:
                Vxtarget = 0.0
                Vytarget = 0.0
                AtoBxnorm = 0.0 
                AtoBynorm = 0.0 
                rand_sx = 1.0
                rand_sy = 1.0

            if ((self.objects[n].x>Xmax and self.objects[n].Vx>0) or (self.objects[n].x<Xmin and self.objects[n].Vx<0)):
                Vxnew = -self.objects[n].Vx
            else:
                Vxnew = max(-(Vmax*abs(AtoBxnorm)+Vplus),\
                           min(Vmax*abs(AtoBxnorm)+Vplus,self.objects[n].Vx \
                             + max(-dVmax,min(dVmax,Vxtarget-self.objects[n].Vx-rand*rand_sx*(random.random()-0.5)))))
            if ((self.objects[n].y>Ymax and self.objects[n].Vy>0) or (self.objects[n].y<Ymin and self.objects[n].Vy<0)):
                Vynew = -self.objects[n].Vy
            else:
                Vynew = max(-(Vmax*abs(AtoBynorm)+Vplus),\
                           min(Vmax*abs(AtoBynorm)+Vplus,self.objects[n].Vy\
                             +max(-dVmax,min(dVmax,Vytarget-self.objects[n].Vy+rand*rand_sy*(random.random()-0.5)))))
            if object_list[n]["Damage"] > 0.4 :
                Vxnew +=  Vytarget*Viblation_amp*math.sin(self.world_state["Time"]/Viblation_time)
                Vynew += -Vxtarget*Viblation_amp*math.sin(self.world_state["Time"]/Viblation_time)

            self.objects[n].Vx = Vxnew
            self.objects[n].Vy = Vynew

        for n in range(len(object_list)):
            self.objects[n].x += SAMPLING_TIME*self.objects[n].Vx
            self.objects[n].y += SAMPLING_TIME*self.objects[n].Vy

        self.world_state["Time"] += self.world_character["Time_interval"]

    #####################################
    # Piko piko motion
    # ピコ とは、Y方向に一瞬動いて戻る事
    def world_pikopiko(self):
        # 0 がA、1 がB
        AtoBx = self.objects[1].x -self.objects[0].x
        AtoBy = self.objects[1].y -self.objects[0].y
        
        for n in range(len(object_list)):
            rand = object_list[n]["Rand"]
            Amax = object_list[n]["Amax"]
            Vmax = object_list[n]["Vmax"]
            dVmax = Amax*SAMPLING_TIME
            Vxtarget = 0.0
            if self.objects[n].y > (Ymax-1.0): 
                Vytarget = -0.2
            else:
                Vytarget = 0.0
            if ((self.objects[n].x>Xmax and self.objects[n].Vx>0) or (self.objects[n].x<Xmin and self.objects[n].Vx<0)):
                Vxnew = -self.objects[n].Vx
            else:
                Vxnew = max(-Vmax,min(Vmax,self.objects[n].Vx\
                                    + max(-dVmax,min(dVmax,Vxtarget-self.objects[n].Vx-rand*(random.random()-0.5)))))
            if ((self.objects[n].y>Ymax and self.objects[n].Vy>0) or (self.objects[n].y<Ymin and self.objects[n].Vy<0)):
                Vynew = -self.objects[n].Vy
            else:
                Vynew = max(-Vmax,min(Vmax,self.objects[n].Vy\
                                    + max(-dVmax,min(dVmax,Vytarget-self.objects[n].Vy+rand*(random.random()-0.5)))))

            if n == 0: # A
                if self.objects[n].piko == False:
                    if random.random() < self.piko_freq :
                        self.objects[n].piko_time = self.world_state["Time"] + SAMPLING_TIME*self.piko_duration
                        self.objects[n].piko = True
                        self.objects[n].y += self.piko_Y
                        Vynew += self.piko_Y/SAMPLING_TIME
                elif self.objects[n].piko_time < self.world_state["Time"] :
                    self.objects[n].piko = False
                    self.objects[n].y -= self.piko_Y
                    Vynew -= self.piko_Y/SAMPLING_TIME
            elif n == 1: # B
                if self.objects[n].piko == False:
                    # A (n=0)に続いてピコする
                    if self.objects[0].piko == False \
                        and self.objects[0].piko_time > (self.world_state["Time"] - 1.5*SAMPLING_TIME) : 
                        self.objects[n].piko_time = self.world_state["Time"] + SAMPLING_TIME*2
                        self.objects[n].piko = True
                        self.objects[n].y += self.piko_Y
                        Vynew += self.piko_Y/SAMPLING_TIME
                elif self.objects[n].piko_time < self.world_state["Time"] :
                    self.objects[n].piko = False
                    self.objects[n].y -= self.piko_Y
                    Vynew -= self.piko_Y/SAMPLING_TIME
            else:
                if self.objects[n].piko == False:
                    if random.random() < self.piko_freq :
                        self.objects[n].piko_time = self.world_state["Time"] + SAMPLING_TIME*self.piko_duration
                        self.objects[n].piko = True
                        self.objects[n].y += self.piko_Y
                        Vynew += self.piko_Y/SAMPLING_TIME
                elif self.objects[n].piko_time < self.world_state["Time"] :
                    self.objects[n].piko = False
                    self.objects[n].y -= self.piko_Y
                    Vynew -= self.piko_Y/SAMPLING_TIME

            self.objects[n].Vx = Vxnew
            self.objects[n].Vy = Vynew

        for n in range(len(object_list)):
            self.objects[n].x += SAMPLING_TIME*self.objects[n].Vx
            self.objects[n].y += SAMPLING_TIME*self.objects[n].Vy

        self.world_state["Time"] += self.world_character["Time_interval"]

###################################################################################################
    def plot_world(self, fig, ax):
        anime = []
        #plt.cla()   
        #plt.plot(Xmax, Ymax)
        #plt.plot(Xmin, Ymin)
        #ax = fig.add_subplot(1,1,1)
        for n in range(len(object_list)):
            color = object_list[n]["color"]
            x = self.objects[n].x
            y = self.objects[n].y
            size = object_list[n]["size"]
            sca = plt.scatter(x,y, c=color, s=size, marker='o')
            anime.append(sca)

        #ax.scatter(Xmin,Ymin, c='white', edgecolors='white', alpha=0.1, s=1, marker='o')
        #ax.scatter(Xmax,Ymax, c='white', edgecolors='white', alpha=0.1, s=1, marker='o')
        left =   np.array([Xmin, Xmax, Xmax, Xmin, Xmin])
        height = np.array([Ymin, Ymin, Ymax, Ymax, Ymin])
        ani, = plt.plot(left, height, color = 'black')
        anime.append(ani)
        return anime

#############################################################################################################
def plot_person(detected_objs, relations, fig, ax):
    anime=[]
    #plt.cla()   
    #ax = fig.add_subplot(1,1,1)
    for item in detected_objs:
        if item.Attention > 0.5 : #and item.continuity == True:
            x = item.point.x[0]
            y = item.point.x[1]
            ID = item.point.ID
            if ID == 1:
                color = 'blue'
            elif ID == 2:
                color = 'red'
            elif ID == 3:
                color = 'green'
            elif ID == 4:
                color = 'orange'
            else:
                color = 'cyan'
            # 形でAttentionの理由を示す。
            if item.Volitionality > item.Punctuality: 
                mark ='*' # ☆
                size = 300
            elif item.Punctuality > 0.0:
                mark = ',' # 四角
                size = 150
            else: #Kinesis
                mark = 'o'
                size = 150
            sca = plt.scatter(x,y,c = 'none', edgecolors=color, alpha=1.0, s=size, marker=mark)
            anime.append(sca)
    for relation in relations :
        if relation.Attention > 0.5 :
            xa = relation.pointA.x[0]
            ya = relation.pointA.x[1]
            xb = relation.pointB.x[0]
            yb = relation.pointB.x[1]
            max_val = max(relation.relation.get("Kinesis", 0.0), relation.relation.get("Affectedness",  0.0))
            if relation.relation.get("Kinesis", 0.0) > relation.relation.get("Affectedness",  0.0):
                if relation.relation.get("Agency", 0.0) > relation.relation.get("Kinesis", 0.0): 
                    line_colour = 'pink' # "Agency"
                    x = xa
                    y = ya
                else:
                    line_colour = 'red' # "Kinesis"="Sideby"
                    x = xa
                    y = ya
            else:
                if relation.relation.get("Affectedness",  0.0) > relation.relation.get("Agency", 0.0): 
                    line_colour = 'green' # "Affectedness"
                    x = xb
                    y = yb
                else:
                    line_colour = 'pink' # "Agency"
                    x = xa
                    y = ya
            if relation.relation.get("Linkageness", 0.0) > max(max_val,0.5):
                line_colour = 'blue' # "Linkageness"
                x = xa
                y = ya
            elif relation.relation.get("Linkageness", 0.0) < -max(max_val,0.5):
                line_colour = 'blue' # "Linkageness"
                x = xb
                y = yb

            left = np.array([xa, xb])
            height = np.array([ya, yb])
            ani, = ax.plot(left, height, color = line_colour)
            anime.append(ani)
            sca = ax.scatter(x,y,c = 'none', edgecolors='pink', alpha=0.7, s=350, marker='o')
            anime.append(sca)
    return anime

################################################################################################
def main():
    plot = []
    world1 = World("Four_Objects")
    detected_objs = []
    n = 1
    for obj in world1.objects:
        point=pcd.Point_F(2)
        point.x = [obj.x,  obj.y]
        point.v = [obj.Vx, obj.Vy]
        point.ID = copy.copy(n)
        detected_objs.append(Detected_Object(point, filter_time=0.2))
        n += 1

    relations = []
    for obj1 in detected_objs:
        for obj2 in detected_objs:
            if obj1.point.ID > obj2.point.ID:
                rel = Relation(obj1, obj2, filter_time=0.2)
                relations.append(rel)
    fig, ax = plt.subplots()
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    fig.show()
    s = ''

    for n in range(200):
        plt.cla()   # ビデオを残すときはここを殺す必要
        #world1.world_change_run_and_chase()
        world1.world_pikopiko()
        n=0
        for obj in world1.objects:
            detected_objs[n].point.x = [obj.x,  obj.y]
            detected_objs[n].point.v = [obj.Vx, obj.Vy]
            n += 1

        for obj in detected_objs:
            obj.objects_kinesis(time.time())
    
        for rel in relations:
            cog = rel.relation_evaluation(time.time())
            # "Sideby"の表示
            if rel.Attention > 0.5:
                for cogkey in cog.keys():
                    if max(0., cog[cogkey])**2 == rel.Attention:
                        s = cogkey
                #print("Attention in ID %d and %d %s val=%.2f" % (rel.pointA.ID, rel.pointB.ID, s, rel.Attention))
                if s == "Sideby":
                    print("VdotVnor = %.2f, DeltaXMeanVnor = %.2f" %(rel.VdotVnor, rel.DeltaXMeanVnor))
            #else:
            #    print("Non relation")
        #print("Whole samp----------------------------")
        #save_person(person1)
        # 下記、既に限界の早さで動いてる・・・
        #time.sleep(SAMPLING_TIME)
        ani1=plot_person(detected_objs, relations, fig, ax)
        ani2=world1.plot_world(fig, ax)
        plot.append(ani1+ani2)
        fig.show()
        plt.pause(0.01)

    # 下記 plt.show() を有効にすると、終了後もグラフィックウィンドウを残せる
    #plt.show()
    # ビデオを残すときは、plt.cla() を殺すこと
    ani = ArtistAnimation(fig, plot, interval=150)
    ani.save("ling.mp4")
    #ani.save("plot2.gif",writer='imagemagick')


if __name__ == '__main__':
    main()

def sample_mp4():
    # mp4 file
    artists = []
    fig, ax = plt.subplots()
    for i in range(100):
        x = np.linspace(0, 4*np.pi)
        y = np.sin(x - i/100 * 2*np.pi)
    
        # アニメーション化する要素の準備
        my_line, = ax.plot(x, y,"blue")
        my_text = ax.text(0, y[0], " ⇐ inlet", color="darkblue", size ="large")
        my_title = ax.text( 4.5, 1.15, f"Count = {i}", size="xx-large")
    
        #  アニメーション化する要素をリスト化
        artists.append([my_line, my_text, my_title])
 
    # 4. アニメーション化
    anim = ArtistAnimation(fig, artists, interval=50)
    anim.save("hoge3.mp4")
    #plt.show()