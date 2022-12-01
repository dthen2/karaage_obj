# 意識のメモリー空間 2021Sep10
# 2021Jul2のメモ参照
# 2021Sep29 新規作成。以前は Data_Aquire_Condition というクラスを作っていたが、そんなクラス不要で、ディクショナリに変更

import numpy as np
import math
import random
import copy
import time

import id_network as ids

############################################################################### 
# メモリー空間上のデータのフォーマット
# project_IDを使わなくてもメモリ空間自体を分けることでも分離が可能だが、頑健にするために設ける
class Data_On_Memory:
    def __init__(self, projectID, appID, workID, fieldID, dataID):
        self.data_ID = dataID           # データのID。data_type = unit_data に与える。2021Sep29新設
        self.project_ID = projectID     # プロジェクトのID。プロジェクトは最上位概念で、例えば異なるクライアントの仕事をプロジェクトIDで区別する
        self.app_ID = appID             # データを生成したアプリのID
        self.work_ID = workID           # データのIDといって言い。ただしこのIDに紐付いたデータは書き換わる。一連のデータのIDとしてこれを使用
        self.fieldID = fieldID            # データが存在する空間。リアルであったり、想像であったり、推定であったりする
        self.participant_IDs = []       # 参加者=認知言語学用語(participant)のIDリスト。使い方はフレキシブルに考えても良さそうな
        self.work_origin_IDs = []       # 元データのwork_IDのリスト
        self.time_stamp = time.time()   # タイムスタンプ
        self.data = 0                   # データ本体を格納。ここは何でも良く、ダミーで0を入れる
        self.data_reliance = 1.0        # 0から1の値で確度を表す
        self.data_type = ""             # データタイプを文字列で表す。2021June27のメモ参照。使用する文字列は、id_network.py の data_type_str() 参照
        self.Attention_self = 0.0       # 自前でつける Attention 
        self.Attention_third = []       # Attention_third は、他のアプリから書き込める。逆に、自分で書いちゃダメ（上書きされる）
                                        # Attention_third に入れるデータは、{"project_ID":,"app_ID":,"work_ID":,"Attention":} の形式で、
                                        # かつ、aq_data_IDs を指定すること
        self.Attention_ID = 0           # Attentionを与えたい相手のID

############### dat_ID が ad_IDs に含まれるかの判定 ###########
# aq_IDs は整数（ID)のリストで、dat_ID は整数（ID)
def _scan_IDs(aq_IDs, dat_ID):
    if aq_IDs :
        for aq_ID in aq_IDs :
            if aq_ID == dat_ID :
                return(True)
    else:   # リストが空なら不問 = True
        return(True)
    return(False)

################## データ要求条件に一致するかどうかの判定 #######################
# work_ID は条件に含まれず、ログのためだけに値を持っている
# aq_cond は上記に示すディクショナリ
# data は Data_On_Memory クラスのデータ
def aq_condition_match(aq_cond, data):
    attention_max = data.Attention_self
    for att in data.Attention_third:
        attention_max = max(attention_max, att.get("Attention",0.0))
    # データ取得の必要条件
    if aq_cond.get("project_ID",0) == data.project_ID \
    and _scan_IDs(aq_cond.get("aq_data_IDs", []), data.data_ID) \
    and _scan_IDs(aq_cond.get("aq_app_IDs",[]), data.app_ID) \
    and _scan_IDs(aq_cond.get("aq_work_IDs",[]), data.app_ID) \
    and _scan_IDs(aq_cond.get("aq_data_types",[]), data.data_type) \
    and _scan_IDs(aq_cond.get("aq_fieldIDs",[]), data.fieldID) \
    and aq_cond.get("aq_attention_min",0.0) <= attention_max \
    and aq_cond.get("aq_time_stamp_oldest",0.0) <= data.time_stamp \
    and aq_cond.get("aq_time_stamp_newest", time.time()) >= data.time_stamp \
    and aq_cond.get("aq_data_reliance_min", 0.0) <= data.data_reliance :
        ###### data.participant_IDs のスキャン #######
        # 下準備
        if aq_cond.get("aq_group_IDs",0) :
            group_flag = False
            group_num = len(aq_cond.get("aq_group_IDs",[]))
        else:
            group_flag = True
            group_num = 0
        # 下準備
        if aq_cond.get("aq_participant_IDs",0) : # 参加者のID一致。一つでも合うのがあればよしとする
            char_flag = False
        else:   # リストが空なら不問 = True
            char_flag = True
        # data.participant_IDs のスキャン開始
        g_count = 0
        # data.participant_IDs は非常に数が多い場合があるので、スキャンの順がこうなる
        for d_char_ID in data.participant_IDs : # ここだけ for がネストしてるので遅くなりそうで気になる
            for aq_group_ID in aq_cond.get("aq_group_IDs",[]) : # グループ（全ての参加者がいれば取得）のスキャン
                if aq_group_ID == d_char_ID :
                    g_count += 1
            if g_count == group_num :
                group_flag = True
            if char_flag == False:
                char_flag = _scan_IDs(aq_cond.get("aq_participant_IDs",[]), d_char_ID) # 一つでも参加者が合えば取得、のスキャン
            if char_flag and group_flag:
                break
        # data.participant_IDs のスキャン完了

        origin_flag = False
        if aq_cond.get("aq_work_origin_IDs",0) :
            for aq_origID in aq_cond.get("aq_work_origin_IDs",[]) :
                origin_flag = _scan_IDs(data.work_origin_IDs, aq_origID)
        else:   # リストが空なら不問 = True
            origin_flag = True

        if char_flag and group_flag and origin_flag :
            return(True)
        else:
            return(False)
    else:
        return(False)


#####################################################################################
# メモリー空間本体。つまり、上記 Data_on_memoryクラスデータのリスト
# 「帳簿」と「本の束」は、librarian_scannerが呼ばれる度にリセットされることに注意
# つまり、タイマ割り込みのような事をすると危険、要注意
class Memory_space:
    def __init__(self):
        self.memory_space = []   # メモリー空間本体。リストの中身は、Data_on_memoryクラスのデータ
        self.space_size = 1000   # メモリー空間のサイズ上限（データの数）
        self.time_limit = 10.0   # 単位は秒。これ以上古いデータは消す。詳細仕様はこれから考える
        self.aq_conditions = []  # 「帳簿」アプリからの要求データ仕様のリスト。Data_Aquire_Conditions のリストになってる
        self.upload_datas = []   # 「帳簿」アプリからアップロードするデータ。Data_On_Memory のリストになってる
        self.acquired_datas = {} # 「本の束」メモリ空間から取得し、アプリに渡すデータ。例外的にディクショナリで、アプリのIDを文字列としてキーとし、
                                 #  その「値」は Data_On_Memory のリスト（「本の束」）
        # ここから、アプリとやりとりする二人目の司書が使う為の領域
        self.aq_conditions_tmp = []  # 「帳簿」の一時書き込み
        self.upload_datas_tmp = []   # 「帳簿」の一時書き込み
        self.acquired_datas_tmp = {} # 「本の束」の一時書き込み例外的にディクショナリ。
                                     # acquired_datas と acquired_datas_tmp はディクショナリで、キーは app_ID を文字列化したもので、値はリスト。
                                     # リストの中身は、メモリー空間から取得したデータ（Data_On_Memory形式）で、
                                     # ディクショナリにする理由は、二人目の司書（librarian_counter）でのスキャンを速くするため

    ###############################################################################
    # データ取得条件
    # 要するにData_On_Memoryと似た構成だが、
    # 要求する側の属性と、データ側に要求される属性に分かれる
    # データ本体が無いのと、
    # タイムスタンプの範囲を示している所、
    # アテンションとデータ確度は最小値で示され、
    # データタイプはリストになっている所が違う
    DATA_AQUIRE_CONDITION = { \
#        プロジェクトID。これは例外なく一致を求める
        "project_ID":         [ 0, "Project ID. It must be matched"], \
#        データを要求するアプリ自身の ID。これも例外なく一致を求める。この値は、 aq_condition_match ではなく、データを渡す際にチェックされる。
        "app_ID":             [ 0, "App ID of the requester"], \
#   ### ここまで、データを要求する側の属性
#   ### ここから、欲しいデータの属性
#        欲しい参加者(participant)のIDのリスト。or結合で、リスト中のどれか一つがあれば取得
        "aq_participant_IDs": [ [], "List of ID of participant in data. OR is used, i.e., if one of the participant was matched, data will be obtained"], \
#       欲しい参加者(participant)のIDのリストだが、and 結合で、リスト中の全てがあった場合のみ取得
        "aq_group_IDs":       [ [], "List of ID of participant in data. AND is used, i.e., all of list have to matched to participants in data."], \
#       データを生成したアプリのリスト。空リストなら不問
        "aq_app_IDs":         [ [], "List of app_IDs who created the data"], \
#       データのIDのリスト。空リストなら不問
        "aq_work_IDs":        [ [], "List of work_ID in data"], \
#       元データの work_ID のリスト。空リストなら不問
        "aq_work_origin_IDs": [ [], "List of work_origin_ID in data"], \
#       fieldID のリスト。空リストなら不問
        "aq_fieldIDs":        [ [], "List of fieldID"], \
#       data_ID のリスト。空リストなら不問
        "aq_data_IDs":         [ [], "List of data_ID"], \
#       取得するデータの古さ限度。デフォルトは現在時刻の10秒前までとした
        "aq_time_stamp_oldest": [ time.time() - 10.0, "Oldest limit of timestamp of data"], \
#       取得するデータの新しさ限度。現在時刻までとした
        "aq_time_stamp_newest": [ time.time(),        "Newest limit of timestamp of data"], \
#       Attention 最低値 0から1の値で注意を表す。ゼロなら不問
        "aq_attention_min":   [ 0.0, "Minimum attention value"], \
#       0から1の値で確度 data_reliance の最低値を表す。ゼロなら不問
        "aq_data_reliance_min": [ 0.0, "Minimum data_reliance"], \
#       欲しいデータのデータタイプ。文字列のリスト。空リストなら不問
        "aq_data_types":      [ [], "List of data type (strings)"]} 

    ################################################################
    # 開発者用に、 aq_condition に書き込むべき要素のリストを羅列する
    # キーが aq_conditions の要素にあるキーで、値がその説明文。
    # ここで呼んでいる init_properties() 関数は、id_network.py に定義されている。
    def data_aquired_condition_str(self):
        return(ids.init_properties(1, self.DATA_AQUIRE_CONDITION)) # num=1 は説明文

    ################# メモリー空間からのデータ削除条件。今後中身を充実 #######################
    # Attention があった場合のデータ延命などの処置の作成が今後必要
    def _delate_data(self, data_m):
        if data_m.time_stamp < (time.time() - self.time_limit):
            return(True)
        else:
            return(False)

    #########################################################################
    # メモリー空間マネージャー
    # 詳細仕様は今後追加していく。
    # space_size を超えたときの強制削除と、削除データの長期記憶への移行を今後作成する必要
    # ログ機能も必要
    def memory_space_manerger(self):
        for n in reversed(range(len(self.memory_space))):
            if self._delate_data(self.memory_space[n]) :
                del self.memory_space[n]
    #def memory_space_manerger_old(self):
    #    counter = 0
    #    del_counts = [] # 削除するデータのリスト。memory_spaceリストのインデックスで表記
    #    for data_m in self.memory_space:
    #        if self._delate_data(data_m) :
    #            del_counts.append(counter)
    #            counter += 1
    #    ### 削除作業。削除するとインデックス変わっちゃうので、逆順でスキャンする ###
    #    for count in reversed(del_counts):
    #        del self.memory_space[count]

    ########################### データ更新（アップロード）の可否判断を個々のデータに対して行う #############
    # work_ID と project_ID の一致に加え、data_ID の一致を追加 2021Sep29
    def _put_condition_much(self, data, upload_data):
        if data.project_ID == upload_data.project_ID \
        and data.work_ID == upload_data.work_ID \
        and data.data_ID == upload_data.data_ID:
            return(True)
        else:
            return(False)

    #################################################################################
    # 各アルゴリズムとメモリー空間の橋渡しをするのがこの librarian（司書）
    # 一人目の司書である librarian_scanner は、メモリ空間のスキャンを担当
    # メイン関数からタイマーで呼ばれると想定
    # aq_conditions は、各アプリから受け取ったデータ取得条件で、Data_Aquire_Conditionsのリスト
    # upload_datas は、各アプリから受け取ったデータで、Data_On_Memoryのリスト
    def librarian_scanner(self):
        # 先ず、データの取得
        # 「本の束」を刷新する。ここだけディクショナリである事に注意
        # このディクショナリは、キーは app_ID を文字列化したもので、値はリスト
        # 要するに、アプリ毎に渡すデータをまとめてリスト化している
        self.acquired_datas = {} # 一旦白紙にする
        for aq_cond in self.aq_conditions:
            for data in self.memory_space:
                if aq_condition_match(aq_cond, data): # データ取得条件が一致したら
                    # データ取得本体
                    # 要ログ生成
                    appIDstr = str(aq_cond.get("app_ID",0))
                    self.acquired_datas.setdefault(appIDstr, [])
                    self.acquired_datas[appIDstr].append(data)
        
        # 次に新データのアップロード 
        for upload_data in self.upload_datas: 
            uploaded = False
            #upload_data.time_stamp = time.time() # タイムスタンプは司書の責任で更新
            for data in self.memory_space:
                if self._put_condition_much(data, upload_data): # データ更新条件が一致したらデータ更新
                    #upload_data.time_stamp = time.time() # タイムスタンプは司書の責任で更新
                    # データの書き込み。また、Attention_third も継承される。
                    upload_data.Attention_third = copy.copy(data.Attention_third)
                    # copyをdeepcopyにしたが、変わらず
                    data = upload_data #copy.deepcopy(upload_data)
                    uploaded = True
                    # 要ログ生成
                # Attention の追記。例外的に、他のデータに Attention を付与できる。
                if data.data_ID == upload_data.Attention_ID:
                    data.Attention_third.append(upload_data)
            # 更新条件にマッチしなかった場合は、新たにメモリー空間に追加する
            if uploaded == False:
                self.memory_space.append(upload_data)
                print('New data uploaded. ID: ' + str(upload_data.data_ID))
        self.aq_conditions = [] # データ取得完了し、「帳簿」を空にする
        self.upload_datas = []  # アップロード完了し、「帳簿」を空にする

    #################################################################
    # 隠れた3人目の司書。事実上、一人目の司書の一部だが、プロセス管理の都合上、別の関数にしている。
    # _tmp のデータとスワップするだけ
    # この関数を走らせている間だけは、下記の二人目の司書は待って頂く
    # ギリギリまで帳簿を集票するために、librarian_scanner の直前に走らせる
    def librarian_swapper_before_scanner(self):
        self.aq_conditions_tmp, self.aq_conditions = self.aq_conditions, self.aq_conditions_tmp
        self.upload_datas_tmp,  self.upload_datas  = self.upload_datas,  self.upload_datas_tmp

    #################################################################
    # 隠れた4人目の司書。事実上、一人目の司書の一部だが、プロセス管理の都合上、別の関数にしている。
    # _tmp のデータとスワップするだけ
    # この関数を走らせている間だけは、下記の二人目の司書は待って頂く
    # 速くデータを渡すために、librarian_scanner の直後に走らせる
    def librarian_swapper_after_scanner(self):
        self.acquired_datas_tmp, self.acquired_datas = self.acquired_datas, self.acquired_datas_tmp

    ############################################################################################################
    # librarian（司書）は二人いて、お客様カウンターでアプリとの授受をする二人目がこれ
    # アプリから、要求データ仕様と、アップロードするデータののリスト（伝票 = aq_ondition_list, upload_data_list）を渡し、
    # メモリ空間から得たデータ（本）をアプリから渡されたリストのポインタ（aq_data_list）に書き込む
    # アプリ側からこの関数が呼び出されることを想定（アプリ単位でやりとりする）
    # ここでセキュリティ上の問題。project_IDを見てないので、アプリ側で情報リークする恐れ
    # 上記librarian_scanner と扱うデータを分けることで、同時に走らせても問題が起きないようにした
    def librarian_counter(self, appID, aq_condition_list, upload_data_list, aq_datas_list):
        # 要求データ仕様とアップロードするデータ（伝票）を帳簿に追記
        self.aq_conditions_tmp += aq_condition_list 
        self.upload_datas_tmp  += upload_data_list
        # アプリに渡すデータ（本）をリストに書き込む
        # アプリの数が膨大になるとスキャンに時間がかかるため、探索の速いディクショナリを採用した
        appIDstr = str(appID)
        aq_data  = self.acquired_datas_tmp.get(appIDstr, [])
        aq_datas_list += aq_data
        # aq_datas_list の本体はポインタなので、呼び出した側で参照できる

    #def librarian_counter_old(self, appID, aq_condition_list, upload_data_list, aq_datas_list):
    #    # 要求データ仕様とアップロードするデータ（伝票）を帳簿に追記
    #    if aq_condition_list :
    #        for aq_condition in aq_condition_list:
    #            self.aq_conditions_tmp.append(aq_condition)
    #    if upload_data_list :
    #        for upload_data in upload_data_list:
    #            self.upload_datas_tmp.append(upload_data)
    #    # アプリに渡すデータ（本）をリストに書き込む
    #    # アプリの数が膨大になるとスキャンに時間がかかるため、探索の速いディクショナリを採用した
    #    appIDstr = str(appID)
    #    aq_data = self.acquired_datas_tmp.get(appIDstr, [])
    #    aq_datas_list += aq_data
    #    # aq_datas_list の本体はポインタなので、呼び出した側で参照できる



###############################################################################################################
###############################################################################################################
# アプリ側サンプルプログラム
# librarian_counter()は、アプリ側から呼ぶ
# プロセスを分ける（プロセス間通信前提）場合のサンプル
def app_main_sample1(self):
    idns = ids.ID_NetWork()
    projectID = idns.register_ID({"data_type":"project","time_stamp": time.time()}) 
    appID = idns.register_ID({"data_type":"app","time_stamp": time.time()})
    memory_space = init_app()  # メモリースペースをもらってくる。共有の仕方は後ほど。
    
    aq_datas_list = [] # メモリ空間から取ってきたデータを入れる入れ物
    
    # 要求データ要件を data_aqcuire_condition に書き込む
    data_aqcuire_condition1 = Data_Aquire_Condition(projectID, appID, workID)
    data_aqcuire_condition2 = Data_Aquire_Condition(projectID, appID, workID)

    while(True): # 本当はタイマー機能でやり、work毎に走らせるとかやるはず。後ほど
        workID = IDnetwork(workID) # 本当は、データによって新しいworkを立ち上げたりする
        # データの要求条件作成
        aq_condition_list = []
        data_aqcuire_condition1 = hoge# データ取得条件を色々書き込む
        data_aqcuire_condition2 = hoge# データ取得条件を色々書き込む
        aq_condition_list.append(data_aqcuire_condition1)
        aq_condition_list.append(data_aqcuire_condition2)

        # 新しいデータ作成
        upload_data_list = []
        upload_data1 = app_mainbody1(aq_datas_list)  # アプリのアルゴリズム本体
        upload_data2 = app_mainbody2(aq_datas_list)  # アプリのアルゴリズム本体
        upload_data_list.append(upload_data1)
        upload_data_list.append(upload_data2)   

        # 司書にデータ以来とアップロード依頼をする
        memory_space.librarian_counter(appID, aq_condition_list, upload_data_list, aq_datas_list)


###############################################################################################################
###############################################################################################################
# アプリ側サンプルプログラム
# librarian_counter()は、アプリ側から呼ぶ
# プロセスを分けない（同じmain()中に置く）場合のサンプル
#
# このアプリ固有のデータ。work 毎に持たせるもの
# メモリ空間は、同一IDを保ったままどんどん更新していく。それを念頭に設計する。、元データ（点群）
# ここでは、元の少数点群から、個々の点の時系列データを生成するものを作る
# 元データは一つで、生成するデータは複数である事に注意
class Work_in_app2:
    def __init__(self, workID, time_org, data_orgID, data_org):
        self.workID = workID
        self.time_org = time_org        # 最新入力データのタイムスタンプ
        self.data_orgID = data_orgID    # 入力データのID
        self.fieldID = 0                # 入力データのものを継承する（例）が、初期化には入れない
        self.data_org = data_org        # 入力データ
        self.data_created = []          # 生成した手持ちデータのリスト

class Sample_apply2:
    def __init__(self, memory_space, idnet, projectID):
        self.memory_space = memory_space
        self.idns = idnet
        self.projectID = projectID
        self.appID = idnet.register_ID({"data_type":"app","time_stamp": time.time()})
        self.aq_datas_list = []     # メモリ空間から得たデータの置き場
        self.works = []
        self.data_aqcuire_conditions = [] # データ取得条件のリスト

    # データ処理の本体。work単位で処理する
    # ここで、入力データ構造として、"ID", "x" を持つディクショナリと想定した（後ほど点群フォーマットに直したい）
    # "x" は[x, y] のフォーマット
    # 生成データは"ID", "ID_org", "xs" を持つディクショナリで、"xs" は、[[x,y,t], [x,y,t]...]のフォーマットを持つ
    def app_process_sample2(self, work):
        for point in work.data_org: # 元データをスキャン。少数点群
            new_point_detected = True
            for point_created in work.data_created: # 手持ちデータから一致を探す
                if point_created["ID_org"] == point.get("ID", 0):
                    new_point_detected = False
                    xt = copy.copy(point["x"])
                    xt.append(work.time_org)
                    point_created["xs"].append(xt)
            if new_point_detected :
                new_pointID = self.idns.register_ID({"data_type":"point","time_stamp": time.time()})
                xt = copy.copy(point["x"])
                xt.append(work.time_org)
                new_point = {"ID": new_pointID, "ID_org": point.get("ID", 0), "xs": [xt]}
                work.data_created.append(new_point)

    # メモリー空間とのやりとりも含めたプロセス本体。
    def app_main_sample2(self):
        # 前回にメモリー空間からもらったデータをスキャンし、work 毎に整理する。
        for data in self.aq_datas_list:
            new_work_created = True
            for work in self.works :
                if data.data_ID == work.data_orgID: # 現有ID一致しなければ、new_work_createdがＯＮ・・・一つの例
                    new_work_created = False
                    work.data_org = data.data
                    work.time_org = data.time_stamp
            if new_work_created :
                new_workID = self.idns.register_ID({"data_type":"work", "time_stamp": time.time()}) # データによって新しいworkを立ち上げる
                new_work = Work_in_app2(new_workID, data.time_stamp, data.data_ID, data.data)
                new_work.fieldID = data.fieldID
                self.works.append(new_work)

        upload_data_list = []
        for work in self.works : # work 毎に処理を走らせる
            # データ処理本体
            self.app_process_sample2(work)

            # アップロードする新しいデータ作成
            data = Data_On_Memory(self.projectID, self.appID, work.workID, work.fieldID, work.dataID)
            data.data = work.data_created
            data.data_type = "time_series_point_cloud_large_num"
            upload_data_list.append(data) 
        
        # データの要求条件作成
#        aq_condition_list = []
#        new_data_aquire_cond = { "aq_data_types": ["point_cloud_small_num"] }
#                        "project_ID": self.projectID, \
#                        "app_ID":     self.appID, \
#                        "work_ID":    new_workID }
        self.data_aqcuire_conditions = { "aq_data_types": ["point_cloud_small_num"] } 
#        data_aqcuire_condition1 = hoge# データ取得条件を色々書き込む
#        data_aqcuire_condition2 = hoge# データ取得条件を色々書き込む
#        aq_condition_list.append(data_aqcuire_condition1)
#        aq_condition_list.append(data_aqcuire_condition2)

        # データ置き場を空にする
        self.aq_datas_list = []
        # 司書にデータ取得依頼とアップロード依頼をする
        memory_space.librarian_counter(self.appID, self.data_aqcuire_conditions, upload_data_list, self.aq_datas_list)


####################################################
# サンプルプログラム
# プロセスを分けない場合
if __name__ == "__main__":
    idnet = ids.ID_NetWork()
    memory_space = Memory_space()
    projectID = idnet.register_ID({"data_type":"project","time_stamp": time.time()}) # 本当は、それなりのプロセスを経て新規プロジェクト生成
    sample_apply2 = Sample_apply2(memory_space, idnet, projectID)
    #while(True): # 本当は、タイマー機能でやる。後ほど
    for i in range(100):
        memory_space.memory_space_manerger()
        memory_space.librarian_swapper_before_scanner()
        memory_space.librarian_scanner()

        sample_apply2.app_main_sample2()
        
        memory_space.librarian_swapper_after_scanner()
# こんな風に走らす
# 本当はタイマとかで、一定のサンプリングタイムで走らせる

