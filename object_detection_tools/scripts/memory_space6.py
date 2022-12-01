# 意識のメモリー空間 2021Sep10
# 2021Jul2のメモ参照
# 2021Sep29 新規作成。以前は Data_Aquire_Condition というクラスを作っていたが、そんなクラス不要で、ディクショナリに変更
# 2022Jan28 タイムスタンプは司書の責任で更新という仕様を削除。タイムラグを考慮
# 2022Jan28 メモリー空間の更新で、for文の使い方を間違っていたのを直した。これできちんと更新されるようになった。

import numpy as np
import math
import random
import copy
import time

import id_network2 as ids

############################################################################### 
# メモリー空間上のデータのフォーマット
# projectIDを使わなくてもメモリ空間自体を分けることでも分離が可能だが、頑健にするために設ける
class Data_On_Memory:
    def __init__(self, projectID, appID, workID, fieldID, dataID, time_now=False):
        if time_now==False:
            time_now=time.time()
        self.dataID = dataID            # データのID。data_type = unit_data に与える。2021Sep29新設
        self.projectID = projectID      # プロジェクトのID。プロジェクトは最上位概念で、例えば異なるクライアントの仕事をプロジェクトIDで区別する
        self.openProject = False        # 2022Feb11 追加。ここがTrueのデータは他プロジェクトからも参照できる
        self.appID = appID              # データを生成したアプリのID
        self.workID = workID            # データのIDといって言い。ただしこのIDに紐付いたデータは書き換わる。一連のデータのIDとしてこれを使用
        self.fieldID = fieldID            # データが存在する空間。リアルであったり、想像であったり、推定であったりする
        self.participant_IDs = []       # 参加者=認知言語学用語(participant)のIDリスト。使い方はフレキシブルに考えても良さそうな
        self.work_origin_IDs = []       # 元データのworkIDのリスト
        self.time_stamp = time_now      # タイムスタンプ
        self.data = 0                   # データ本体を格納。ここは何でも良く、ダミーで0を入れる
        self.data_reliance = 1.0        # 0から1の値で確度を表す
        self.data_type = ""             # データタイプを文字列で表す。2021June27のメモ参照。使用する文字列は、id_network.py の data_type_str() 参照
        self.Attention_self = 0.0       # 自前でつける Attention 
        self.Attention_third = []       # Attention_third は、他のアプリから書き込める。逆に、自分で書いちゃダメ（上書きされる）
                                        # Attention_third に入れるデータは、{"projectID":,"appID":,"workID":,"Attention":} の形式で、
                                        # かつ、aq_dataIDs を指定すること
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
# aq_cond は下記DATA_AQUIRE_CONDITIONに示すディクショナリ
# data は Data_On_Memory クラスのデータ
# 2022Feb10追加。自分自身の吐き出したデータも取得するかどうかの判定追加
# 2022Feb11 openProject を導入。これがTrueのデータは他プロジェクトからも参照できる
def aq_condition_match(aq_cond, data, time_now=False):
    if time_now==False:
        time_now=time.time()
    attention_max = data.Attention_self
    for att in data.Attention_third:
        attention_max = max(attention_max, att.get("Attention",0.0))
    # データ取得の必要条件
    if (aq_cond.get("projectID",0) == data.projectID or data.openProject==True) \
    and _scan_IDs(aq_cond.get("aq_dataIDs", []), data.dataID) \
    and _scan_IDs(aq_cond.get("aq_appIDs",[]), data.appID) \
    and _scan_IDs(aq_cond.get("aq_workIDs",[]), data.workID) \
    and _scan_IDs(aq_cond.get("aq_data_types",[]), data.data_type) \
    and _scan_IDs(aq_cond.get("aq_fieldIDs",[]), data.fieldID) \
    and aq_cond.get("aq_attention_min",0.0) <= attention_max \
    and aq_cond.get("aq_time_stamp_oldest",0.0) <= data.time_stamp \
    and aq_cond.get("aq_time_stamp_newest", time_now) >= data.time_stamp \
    and aq_cond.get("aq_data_reliance_min", 0.0) <= data.data_reliance \
    and (aq_cond.get("include_self_app", True) or data.appID != aq_cond.get("appID", 0)  ) :
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
# 2022Jan29 aq_conditions_tmp を廃止し、初期に書き込んで書き換えをしない限り保持する仕様に変更
# 2022Feb2 さらに aq_conditions の書き込みを司書にやらせるようにし、分散コンピューティングでクラッシュが起きないようにした。
class Memory_space:
    def __init__(self):
        self.memory_space = []   # メモリー空間本体。リストの中身は、Data_on_memoryクラスのデータ
        self.space_size = 1000   # メモリー空間のサイズ上限（データの数）
        self.time_limit = 10.0   # 単位は秒。これ以上古いデータは消す。詳細仕様はこれから考える
        self.aq_conditions = []  # 「帳簿」アプリからの要求データ仕様のリスト。Data_Aquire_Conditions のリストになってる
        self.aq_conditions_add = [] # aq_conditions に追加する要求データ仕様のリストを格納するバッファ
        self.aq_conditions_rep = [] # aq_conditions に入れ替える要求データ仕様のリストを格納するバッファ [[appID, list],[appID, list],...] の形になっているので注意
        self.upload_datas = []   # 「帳簿」アプリからアップロードするデータ。Data_On_Memory のリストになってる
        self.acquired_datas = {} # 「本の束」メモリ空間から取得し、アプリに渡すデータ。例外的にディクショナリで、アプリのIDを文字列としてキーとし、
                                 #  その「値」は Data_On_Memory のリスト（「本の束」）
        # ここから、アプリとやりとりする二人目の司書が使う為の領域
        #self.aq_conditions_tmp = []  # 「帳簿」の一時書き込み
        self.upload_datas_tmp = []   # 「帳簿」の一時書き込み
        self.acquired_datas_tmp = {} # 「本の束」の一時書き込み。例外的にディクショナリ。
                                     # acquired_datas と acquired_datas_tmp はディクショナリで、キーは appID を文字列化したもので、値はリスト。
                                     # リストの中身は、メモリー空間から取得したデータ（Data_On_Memory形式）で、
                                     # ディクショナリにする理由は、二人目の司書（librarian_counter）でのスキャンを速くするため
        self.upload_status = {}      # 2022Jan29追加。アップロード状況。キーは文字列化したアプリのID。値は各アプリのアップロード数
        self.upload_status_tmp = {}  # 上記の一時書き込み

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
        "projectID":          [ 0, "Project ID. It must be matched"], \
#        データを要求するアプリ自身の ID。これも例外なく一致を求める。この値は、 aq_condition_match ではなく、データを渡す際にチェックされる。
        "appID":              [ 0, "App ID of the requester"], \
#   ### ここまで、データを要求する側の属性
#   ### ここから、欲しいデータの属性
#        欲しい参加者(participant)のIDのリスト。or結合で、リスト中のどれか一つがあれば取得
        "aq_participant_IDs": [ [], "List of ID of participant in data. OR is used, i.e., if one of the participant was matched, data will be obtained"], \
#       欲しい参加者(participant)のIDのリストだが、and 結合で、リスト中の全てがあった場合のみ取得
        "aq_group_IDs":       [ [], "List of ID of participant in data. AND is used, i.e., all of list have to matched to participants in data."], \
#       データを生成したアプリのリスト。空リストなら不問
        "aq_appIDs":          [ [], "List of appIDs who created the data"], \
#       データのIDのリスト。空リストなら不問
        "aq_workIDs":         [ [], "List of workID in data"], \
#       元データの workID のリスト。空リストなら不問
        "aq_work_origin_IDs": [ [], "List of work_origin_ID in data"], \
#       fieldID のリスト。空リストなら不問
        "aq_fieldIDs":        [ [], "List of fieldID"], \
#       dataID のリスト。空リストなら不問
        "aq_dataIDs":         [ [], "List of dataID"], \
#       取得するデータの古さ限度。デフォルトは現在時刻の10秒前までとした
        "aq_time_stamp_oldest": [ - 10.0, "Oldest limit of timestamp of data"], \
#       取得するデータの新しさ限度。現在時刻までとした
        "aq_time_stamp_newest": [ 0.0,        "Newest limit of timestamp of data"], \
#       Attention 最低値 0から1の値で注意を表す。ゼロなら不問
        "aq_attention_min":   [ 0.0, "Minimum attention value"], \
#       0から1の値で確度 data_reliance の最低値を表す。ゼロなら不問
        "aq_data_reliance_min": [ 0.0, "Minimum data_reliance"], \
#       欲しいデータのデータタイプ。文字列のリスト。空リストなら不問
        "aq_data_types":      [ [], "List of data type (strings)"], \
#       2022Feb10追加。自分自身の吐き出したデータも取得する
        "include_self_app":   [ True, "Include data from same app"]} 

    ################################################################
    # 開発者用に、 aq_condition に書き込むべき要素のリストを羅列する
    # キーが aq_conditions の要素にあるキーで、値がその説明文。
    # ここで呼んでいる init_properties() 関数は、id_network.py に定義されている。
    def data_aquired_condition_str(self):
        return(ids.init_properties(1, self.DATA_AQUIRE_CONDITION)) # num=1 は説明文

    ################# メモリー空間からのデータ削除条件。今後中身を充実 #######################
    # Attention があった場合のデータ延命などの処置の作成が今後必要
    def _delate_data(self, data_m, time_now=False):
        if time_now==False:
            time_now=time.time()
        if data_m.time_stamp < (time_now - self.time_limit):
            return(True)
        else:
            return(False)

    #########################################################################
    # メモリー空間マネージャー
    # 詳細仕様は今後追加していく。
    # space_size を超えたときの強制削除と、削除データの長期記憶への移行を今後作成する必要
    # ログ機能も必要
    def memory_space_manerger(self, time_now=False):
        if time_now==False:
            time_now = time.time()
        for n in reversed(range(len(self.memory_space))):
            if self._delate_data(self.memory_space[n], time_now) :
                del self.memory_space[n]

    ########################### データ更新（アップロード）の可否判断を個々のデータに対して行う #############
    # workID と projectID の一致に加え、dataID の一致を追加 2021Sep29
    def _put_condition_much(self, data, upload_data):
        if data.projectID == upload_data.projectID \
        and data.workID == upload_data.workID \
        and data.dataID == upload_data.dataID:
            return(True)
        else:
            return(False)

    #################################################################################
    # 各アルゴリズムとメモリー空間の橋渡しをするのがこの librarian（司書）
    # 一人目の司書である librarian_scanner は、メモリ空間のスキャンを担当
    # メイン関数からタイマーで呼ばれると想定
    # aq_conditions は、各アプリから受け取ったデータ取得条件で、Data_Aquire_Conditionsのリスト
    # upload_datas は、各アプリから受け取ったデータで、Data_On_Memoryのリスト
    # 2022Jan29 仕様変更。アップロードを先に行い、新鮮なデータを渡せるように変更
    def librarian_scanner(self, time_now=False):
        if time_now==False:
            time_now = time.time()
        # 先ず新データのアップロード 
        for upload_data in self.upload_datas: 
            uploaded = False
            for n in range(len(self.memory_space)):
                if self._put_condition_much(self.memory_space[n], upload_data): # データ更新条件が一致したらデータ更新
                    #upload_data.time_stamp = time.time() # タイムスタンプは司書の責任で更新
                    # データの書き込み。また、Attention_third も継承される。
                    upload_data.Attention_third = copy.copy(self.memory_space[n].Attention_third)
                    self.memory_space[n] = copy.deepcopy(upload_data)
                    uploaded = True
                    # 要ログ生成
                # Attention の追記。例外的に、他のデータに Attention を付与できる。
                if self.memory_space[n].dataID == upload_data.Attention_ID:
                    self.memory_space[n].Attention_third.append(upload_data)
            # 更新条件にマッチしなかった場合は、新たにメモリー空間に追加する
            if uploaded == False:
                self.memory_space.append(copy.deepcopy(upload_data))
        self.upload_datas = []  # アップロード完了し、「帳簿」を空にする

        # 次にデータの取得
        # 「本の束」を刷新する。ここだけディクショナリである事に注意
        # このディクショナリは、キーは appID を文字列化したもので、値はリスト
        # 要するに、アプリ毎に渡すデータをまとめてリスト化している
        self.acquired_datas = {} # 一旦白紙にする
        for aq_cond in self.aq_conditions:
            for data in self.memory_space:
                if aq_condition_match(aq_cond, data, time_now): # データ取得条件が一致したら
                    # データ取得本体
                    # 要ログ生成
                    appID = aq_cond.get("appID",0)
                    self.acquired_datas.setdefault(appID, [])
                    self.acquired_datas[appID].append(data)

    #################################################################
    # 隠れた3人目の司書。事実上、一人目の司書の一部だが、プロセス管理の都合上、別の関数にしている。
    # _tmp のデータとスワップするだけ
    # この関数を走らせている間だけは、下記の二人目の司書は待って頂く
    # ギリギリまで帳簿を集票するために、librarian_scanner の直前に走らせる
    def librarian_swapper_before_scanner(self):
        self.upload_datas_tmp,  self.upload_datas  = self.upload_datas,  self.upload_datas_tmp
        self.upload_status = self.upload_status_tmp
        self.upload_status_tmp = {}
        # データ要求仕様の入れ替え
        for rep in self.aq_conditions_rep:
            for n in reversed(range(len(self.aq_conditions))):
                if self.aq_conditions[n]["appID"] == rep[0]: # rep[0]はappID
                    del self.aq_conditions[n]
            self.aq_conditions += rep[1] # rep[1] は入れ替えるaq_conditionsリスト
        self.aq_conditions_rep = []
        # データ要求仕様の初期化
        self.aq_conditions += self.aq_conditions_add
        self.aq_conditions_add = []

    #################################################################
    # 隠れた4人目の司書。事実上、一人目の司書の一部だが、プロセス管理の都合上、別の関数にしている。
    # _tmp のデータとスワップするだけ
    # この関数を走らせている間だけは、下記の二人目の司書は待って頂く
    # 速くデータを渡すために、librarian_scanner の直後に走らせる
    def librarian_swapper_after_scanner(self):
        self.acquired_datas_tmp, self.acquired_datas = self.acquired_datas, self.acquired_datas_tmp

    ############################################################################################################
    # librarian（司書）は二人いて、お客様カウンターでアプリとの授受をする二人目がこれ
    # 「二人目」といいつつ、アプリの数だけ走るので、アプリ毎に担当司書がいるイメージ
    # アプリから、要求データ仕様と、アップロードするデータののリスト（伝票 = aq_ondition_list, upload_data_list）を渡し、
    # メモリ空間から得たデータ（本）をアプリから渡されたリストのポインタ（aq_data_list）に書き込む
    # アプリ側からこの関数が呼び出されることを想定（アプリ単位でやりとりする）
    # ここでセキュリティ上の問題。projectIDを見てないので、アプリ側で情報リークする恐れ
    # 上記librarian_scanner と扱うデータを分けることで、同時に走らせても問題が起きないようにした
    def librarian_counter(self, appID, upload_data_list, aq_datas_list):
        # 要求データ仕様とアップロードするデータ（伝票）を帳簿に追記
        # 要求データ仕様のアップロードを廃止
        #self.aq_conditions_tmp += aq_condition_list
        self.upload_datas_tmp  += upload_data_list
        #appIDstr = str(appID)
        self.upload_status_tmp.setdefault(appID, 0)
        self.upload_status_tmp[appID] += len(upload_data_list)
        # アプリに渡すデータ（本）をリストに書き込む
        # アプリの数が膨大になるとスキャンに時間がかかるため、探索の速いディクショナリを採用した
        aq_data  = self.acquired_datas_tmp.get(appID, [])
        aq_datas_list += aq_data
        # aq_datas_list の本体はポインタなので、呼び出した側で参照できる

    # 2022Jan29 新設。アップロードとダウンロードの司書を別にし、早くデータを得られるようにした。
    # データが1周期早く得られる代わりに通信負荷は増える。なので、元の仕様も残す
    def librarian_counter_up(self, appID, upload_data_list):
        # アップロードするデータ（伝票）を帳簿に追記
        self.upload_datas_tmp  += upload_data_list
        #appIDstr = str(appID)
        self.upload_status_tmp.setdefault(appID, 0)
        self.upload_status_tmp[appID] += len(upload_data_list)

    # # 2022Jan29 新設。ダウンロード専用関数
    def librarian_counter_down(self, appID, aq_datas_list):
        # アプリに渡すデータ（本）をリストに書き込む
        aq_data  = self.acquired_datas_tmp.get(appID, [])
        aq_datas_list += aq_data
        # aq_datas_list の本体はポインタなので、呼び出した側で参照できる

    # 2022Jan29 新設。要求データ仕様のアップロード ---旧仕様
    # この関数は初期化時に走らせるモノ
    #def set_aq_conditions(self, aq_condition_list):
    #    self.aq_conditions += aq_condition_list 

    # 2022Jan29 新設。要求データ仕様の入れ替え。---旧仕様
    # appIDの一致する要求データ仕様を全て削除して入れ替える
    #def renew_aq_conditions(self, appID, aq_condition_list_new):
    #    for n in reversed(range(len(self.aq_conditions))):
    #        if self.aq_conditions[n]["appID"] == appID:
    #            del self.aq_conditions[n]
    #    self.aq_conditions += aq_condition_list_new

    # 2022Jan29 新設。2022Feb2 仕様変更。要求データ仕様のアップロード
    # この関数は初期化時に走らせるモノ
    def set_aq_conditions(self, aq_condition_list):
        self.aq_conditions_add += aq_condition_list 

    # 2022Jan29 新設。2022Feb2 仕様変更。要求データ仕様の入れ替え
    # appIDの一致する要求データ仕様を全て削除して入れ替える
    def renew_aq_conditions(self, appID, aq_condition_list_rep):
        self.aq_conditions_rep += [appID, aq_condition_list_rep]







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
                if data.dataID == work.data_orgID: # 現有ID一致しなければ、new_work_createdがＯＮ・・・一つの例
                    new_work_created = False
                    work.data_org = data.data
                    work.time_org = data.time_stamp
            if new_work_created :
                new_workID = self.idns.register_ID({"data_type":"work", "time_stamp": time.time()}) # データによって新しいworkを立ち上げる
                new_work = Work_in_app2(new_workID, data.time_stamp, data.dataID, data.data)
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
#                        "projectID": self.projectID, \
#                        "appID":     self.appID, \
#                        "workID":    new_workID }
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
        memory_space.librarian_swapper_after_scanner()

        sample_apply2.app_main_sample2()
        
        
# こんな風に走らす
# 本当はタイマとかで、一定のサンプリングタイムで走らせる

