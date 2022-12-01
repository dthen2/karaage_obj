import copy
import time
import id_network2 as ids

############################################################
# 2022Mar25 新設。フィールドを登録。時空スケールや分解能（精度）をここに登録
# 例えば、import field as fld
# として、取得した、または与えた ID で、下記のようにfieldsディクショナリにアクセスすれば、IDだけで他のモジュールからもアクセスできる
#        fld.fields_dic[ID]
# ID にゼロを入力すれば自動で登録してくれるが、推奨しない。property_dat は下記の内容
# property_dat = {"data_type": "field", "time_stamp": time_stamp, "projectID": projectID, "appID": appID, "workID": workID, "relation_IDs": []}
class Fields:
    def __init__(self):
        self.fields_dic = {}

class Field:
    def __init__(self, idnet, fields, ID, property_dat={} ): #time_stamp, projectID, appID, workID):
        if ID == 0:
            self.ID = idnet.register_ID(property_dat)
        else:
            self.ID = ID
        self.time_scale  = 1.0      # 単位は sec 例えばこれが10なら、10 が 1秒として処理される。
                                    # 地球史をイメージするなら、べらぼうにでかい値
        self.space_scale = 1.0      # 単位は m   例えばこれが10なら、x=10 が 1mとして処理される。
                                    # 入力がピクセルで、1ピクセルが0.01mであれば、この値は0.01
                                    # 太陽系をイメージするなら、この値は公転軌道半径(m)=でかい
        self.time_resolution = 0.1  # 入力の数値を単位として、その分解能（サンプリングタイム）。概略値とする
                                    # フィルター時定数を決めるのに使える
        self.space_resolution = 1.  # 入力の数値を単位として、その分解能。画像なら 1 = 1 pixel のままで良いが、信頼できる位置精度が 10 ピクセルなら 10 を入力
        self.max_x = 640.           # カメラのピクセル数を入れるために用意2022Apr14
        self.max_y = 480.           # カメラのピクセル数を入れるために用意2022Apr14
        self.ref_distance = 2.0     # m cogni_ling2.py の関係性にて REF_DISTANCE に入れるために用意。この距離より小さい距離での相互作用は強いとする 2022June2
        self.ref_speed = 0.3        # m/sec cogni_ling2.py の関係性にて、REF_SPEED に入れるために用意 2022June2
        fields.fields_dic[self.ID] = self

##### Test program
if __name__ == "__main__":
    idnet = ids.ID_NetWork()
    fs = Fields()
    f = Field(idnet, fs, 1)#, time.time(), 0,0,0)
    f.time_scale = 20.
    print(fs.fields_dic[f.ID].time_scale)
    print(f.__dict__)