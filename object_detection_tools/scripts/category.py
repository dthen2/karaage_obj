# カテゴリー 2022Feb2 作成開始
import json
import copy
import time
import id_network as ids
import os

### よく考えたら、プロトタイプ毎にファイルに記録するのはナンセンス
### ここで大問題。データは多くはクラスなので、テキスト化してファイルにすることができない！！
#class Category:
#    def __init__(self, idnet, file_name_prt, file_name_str, project_ID, additional_str):
#        self.ID = idnet.register_ID({"data_type":"category","time_stamp": time.time(),'additional_str':additional_str})
#        self.file_name_prt = file_name_prt
#        self.file_name_str = file_name_str
#        # self.prototype: 各プロトタイプのリスト。ディクショナリのリストで、ディクショナリのひな形は上記 PROTOPYPE_DATA
#        if os.path.isfile(file_name_prt):
#            with open(file_name_prt, mode='r') as fp:
#                self.prototypes = json.load(fp)
#        else:
#            with open(file_name_prt, mode='w') as fp:
#                self.prototypes = []
#                fp.write(json.dumps(self.prototypes))
#        # self.stereotypes: ステレオタイプのリスト。ステレオタイプは1個でいいが、複数置けるようにした
#        if os.path.isfile(file_name_str):
#            with open(file_name_str, mode='r') as fs:
#                self.stereotypes = json.load(fs)
#        else:
#            with open(file_name_str, mode='w') as fs:
#                self.stereotypes = []
#                fs.write(json.dumps(self.stereotypes))
#        self.project_ID = project_ID

PROTOTYPE_DATA = { \
            "data_type":  ["None",  "Data type. The list of types can be obtained from data_type_str() in id_network.py ."], \
            "dataID":     [0,       "Data ID of the prototype."], \
            "data":       [0,       "Data itself."], \
            "time_stamp": [0.0,     "Time stamp when the prototype was registered."], \
            "app_ID":     [0,       "ID of App that identified the prototype."], \
            "work_ID":    [0,       "ID of work that identified the prototype."], \
            "stength":    [0.0,     "Strength of matching to the category."]}

#    # プロトタイプの登録
#    def register_prototype(self, prototype_data):
#        self.prototypes.append(prototype_data)
#    
#    def save_prototypes(self):
#        with open(self.file_name_prt, mode='w') as f:
#            f.write(json.dumps(self.prototypes))
#
#    # ステレオタイプの登録
#    def register_stereotype(self, stereotype_data):
#        self.stereotypes.append(stereotype_data)
#    
#    def save_stereotypes(self):
#        with open(self.file_name_str, mode='w') as f:
#            f.write(json.dumps(self.stereotypes))

##################################################
# カテゴリーの集合体としてのクラス
NO_MATCHING = -2
CATEGORY_DATA = {\
    "ID": 0, \
    "prototypes": [],\
    "stereotypes": []
    }

class Categories:
    def __init__(self, ids, project_ID, file_name):
        self.ids = ids
        self.project_ID = project_ID
        self.file_name = file_name
        self.categories = []
        if os.path.isfile(file_name):
            with open(file_name, mode='r') as f:
                self.categories = json.load(f)
        else:
            with open(file_name, mode='w') as f:
                self.categories = []
                f.write(json.dumps(self.categories))

    def save_categories(self):
        with open(self.file_name, mode='w') as f:
            f.write(json.dumps(self.categories))

    # 新しいカテゴリーの登録
    # ここで、category_data は上記 CATEGORY_DATA のフォームに従い、stereotypes リストは上記 PROTOTYPE_DATA の形式のディクショナリのリストになっていること
    def register_category(self, category_data, additional_str):
        newID = self.ids.register_ID({"data_type":"category","time_stamp": time.time(),'additional_str':additional_str})
        newCategory = {"ID": newID, "prototypes": category_data["prototypes"], "stereotypes": category_data["stereotypes"]}
        self.categories.append(newCategory)


    def register_stereotype(self, categoryID, stereotype_data):
        num=0
        for cat in self.categories:
            if cat["ID"] == categoryID:
                break
            num += 1
        self.categories[num]["stereotypes"].append(stereotype_data)

    def register_prototype(self, categoryID, prototype_data):
        num=0
        for cat in self.categories:
            if cat["ID"] == categoryID:
                break
            num += 1
        self.categories[num]["prototypes"].append(prototype_data)    

    def find_same_type_category(self, data_types, startingNum):
        stereonum = 0
        protonum = 0
        #if startingNum < 0:# or startingNum > len(self.categories):
        #    print('startingNum = '+str(startingNum))
        #    print('hoge')
        #    #return NO_MATCHING, NO_MATCHING, NO_MATCHING
        #if startingNum > len(self.categories):
        #    print('hogehage')
        #    #return NO_MATCHING, NO_MATCHING, NO_MATCHING
        for categoryNum in range(startingNum, len(self.categories)):
            #print('startingNum = '+str(startingNum))
            #print('categoryNum = '+str(categoryNum))
            for stereotype in self.categories[categoryNum]["stereotypes"]:
                for data_type in data_types:
                    if stereotype["data_type"]==data_type: 
                        return categoryNum, stereonum, protonum
                stereonum += 1
            for prototype in self.categories[categoryNum]["prototypes"]:
                #print('protonum = '+str(protonum))
                for data_type in data_types:
                    #print(str(data_type)+' '+str(prototype["data_type"]))
                    if prototype["data_type"]==data_type:
                        #print('Match')
                        return categoryNum, NO_MATCHING, protonum
                protonum += 1
        return NO_MATCHING, NO_MATCHING, NO_MATCHING

# サンプルカテゴリー。「の」の字の多数点群
SAMPLE_CLOUD = [[13.6, 16.0],\
        [13.8, 13.0],\
        [13.0, 9.0],\
        [11.5, 5.0],\
        [8.5, 2.0],\
        [5.6, 1.9],\
        [3.9, 4.7],\
        [3.8, 9.0],\
        [5.0, 12.7],\
        [6.6, 15.9],\
        [10.0, 19.0],\
        [13.0, 20.7],\
        [17.0, 20.8],\
        [20.2, 19.2],\
        [22.3, 15.0],\
        [22.6, 11.0],\
        [22.8, 13.2],\
        [21.5, 17.5],\
        [18.8, 20.2],\
        [15.0, 21.0],\
        [11.5, 20.0],\
        [8.2, 17.6],\
        [5.6, 14.5],\
        [4.2, 11.0],\
        [3.6, 7.0],\
        [4.6, 3.0],\
        [7.0, 1.5],\
        [10.0, 3.2],\
        [12.5, 7.2],\
        [13.5, 11.0],\
        [13.7, 14.8]]

pl1  = [150., 130.]
pl2  = [130., 70.]
pl3  = [110., 30.]
pl4  = [ 60., 10.]
pl5  = [ 25., 60.]
pl6  = [ 50., 110.]
pl7  = [100., 160.]
pl8  = [180., 170.]
pl9  = [230., 120.]
pl10 = [220., 40.]
pl11 = [230., 80.]
pl12 = [210., 150.]
pl13 = [140., 170.]
pl14 = [ 75., 140.]
pl15 = [ 35., 90.]
pl16 = [ 23., 35.]
pl17 = [ 33., 12.]
pl18 = [ 85., 17.]
pl19 = [120., 53.]
pl20 = [143., 100.]
SAMPLE_CLOUD2 = [pl1,pl2,pl3,pl4,pl5,pl6,pl7,pl8,pl9,pl10,pl11,pl12,pl13,pl14,pl15,pl16,pl17,pl18,pl19,pl20]