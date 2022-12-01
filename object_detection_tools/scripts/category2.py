# カテゴリー 2022Feb2 作成開始
import json
import copy
import time
import id_network as ids
import os

### よく考えたら、プロトタイプ毎にファイルに記録するのはナンセンス
### ここで大問題。データは多くはクラスなので、テキスト化してファイルにすることができない！！
#class Category:
#    def __init__(self, idnet, file_name_prt, file_name_str, projectID, additional_str):
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
#        self.projectID = projectID

PROTOTYPE_DATA = { \
            "data_type":  ["None",  "Data type. The list of types can be obtained from data_type_str() in id_network.py ."], \
            "dataID":     [0,       "Data ID of the prototype."], \
            "data":       [0,       "Data itself."], \
            "time_stamp": [0.0,     "Time stamp when the prototype was registered."], \
            "appID":      [0,       "ID of App that identified the prototype."], \
            "workID":     [0,       "ID of work that identified the prototype."], \
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
    def __init__(self, ids, projectID, file_name):
        self.ids = ids
        self.projectID = projectID
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
    def register_category(self, category_data, additional_str, time_now=False):
        if time_now==False:
            time_now=time.time()
        newID = self.ids.register_ID({"data_type":"category","time_stamp": time_now,'additional_str':additional_str})
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
        if startingNum < 0 or startingNum > len(self.categories):
        #    print('startingNum = '+str(startingNum))
        #    print('hoge')
            return NO_MATCHING, NO_MATCHING, NO_MATCHING
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

# 「A」のパターン
SAMPLE_CLOUD_A = [[0.0,0.0],\
   [0.5,1.0	],\
   [1.0,2.0	],\
   [1.5,3.0	],\
   [2.0,4.0	],\
   [2.5,5.0	],\
   [3.0,6.0	],\
   [3.5,7.0	],\
   [4.0,8.0	],\
   [4.5,9.0	],\
   [5.0,10.0],\
   [5.5,11.0],\
   [6.0,12.0],\
   [6.5,13.0],\
   [7.0,14.0],\
   [7.5,15.0],\
   [8.0,16.0],\
   [8.5,17.0],\
   [9.0,18.0],\
   [9.5,19.0],\
   [10.0,20.0],\
   [10.5,21.0],\
   [11.0,22.0],\
   [11.5,23.0],\
   [12.0,24.0],\
   [12.5,25.0],\
   [13.0,26.0],\
   [13.5,27.0],\
   [14.0,28.0],\
   [14.5,29.0],\
   [15.0,30.0],\
   [15.5,29.0],\
   [16.0,28.0],\
   [16.5,27.0],\
   [17.0,26.0],\
   [17.5,25.0],\
   [18.0,24.0],\
   [18.5,23.0],\
   [19.0,22.0],\
   [19.5,21.0],\
   [20.0,20.0],\
   [20.5,19.0],\
   [21.0,18.0],\
   [21.5,17.0],\
   [22.0,16.0],\
   [22.5,15.0],\
   [23.0,14.0],\
   [23.5,13.0],\
   [24.0,12.0],\
   [24.5,11.0],\
   [25.0,10.0],\
   [25.5,9.0],\
   [26.0,8.0],\
   [26.5,7.0],\
   [27.0,6.0],\
   [27.5,5.0],\
   [28.0,4.0],\
   [28.5,3.0],\
   [29.0,2.0],\
   [29.5,1.0],\
   [30.0,0.0],\
   [5.0,10.0],\
   [6.0,10.0],\
   [7.0,10.0],\
   [8.0,10.0],\
   [9.0,10.0],\
   [10.0,10.0],\
   [11.0,10.0],\
   [12.0,10.0],\
   [13.0,10.0],\
   [14.0,10.0],\
   [15.0,10.0],\
   [16.0,10.0],\
   [17.0,10.0],\
   [18.0,10.0],\
   [19.0,10.0],\
   [20.0,10.0],\
   [21.0,10.0],\
   [22.0,10.0],\
   [23.0,10.0],\
   [24.0,10.0],\
   [25.0,10.0]]

# 「B」のパターン
SAMPLE_CLOUD_B = [   [	0.0	,	0.0	],\
   [	0.0	,	1.0	],\
   [	0.0	,	2.0	],\
   [	0.0	,	3.0	],\
   [	0.0	,	4.0	],\
   [	0.0	,	5.0	],\
   [	0.0	,	6.0	],\
   [	0.0	,	7.0	],\
   [	0.0	,	8.0	],\
   [	0.0	,	9.0	],\
   [	0.0	,	10.0	],\
   [	0.0	,	11.0	],\
   [	0.0	,	12.0	],\
   [	0.0	,	13.0	],\
   [	0.0	,	14.0	],\
   [	0.0	,	15.0	],\
   [	0.0	,	16.0	],\
   [	0.0	,	17.0	],\
   [	0.0	,	18.0	],\
   [	0.0	,	19.0	],\
   [	0.0	,	20.0	],\
   [	0.0	,	21.0	],\
   [	0.0	,	22.0	],\
   [	0.0	,	23.0	],\
   [	0.0	,	24.0	],\
   [	0.0	,	25.0	],\
   [	0.0	,	26.0	],\
   [	0.0	,	27.0	],\
   [	0.0	,	28.0	],\
   [	0.0	,	29.0	],\
   [	0.0	,	30.0	],\
   [	1.0	,	30.0	],\
   [	2.0	,	30.0	],\
   [	3.0	,	30.0	],\
   [	4.0	,	30.0	],\
   [	5.0	,	30.0	],\
   [	6.0	,	30.0	],\
   [	7.0	,	30.0	],\
   [	8.0	,	30.0	],\
   [	9.0	,	30.0	],\
   [	10.0	,	30.0	],\
   [	11.0	,	29.9	],\
   [	12.0	,	29.8	],\
   [	13.0	,	29.6	],\
   [	14.0	,	29.3	],\
   [	15.0	,	29.0	],\
   [	16.0	,	28.6	],\
   [	17.0	,	28.0	],\
   [	18.2	,	27.0	],\
   [	19.1	,	26.0	],\
   [	19.7	,	25.0	],\
   [	20.0	,	24.0	],\
   [	20.0	,	23.0	],\
   [	20.0	,	22.0	],\
   [	19.8	,	21.0	],\
   [	19.5	,	20.0	],\
   [	19.0	,	19.3	],\
   [	18.0	,	18.5	],\
   [	17.0	,	17.9	],\
   [	16.0	,	17.3	],\
   [	15.0	,	16.7	],\
   [	14.0	,	16.3	],\
   [	13.0	,	15.8	],\
   [	12.0	,	15.4	],\
   [	11.0	,	15.0	],\
   [	9.0 	,	15.0	],\
   [	10.0	,	15.0	],\
   [	12.0	,	14.6	],\
   [	13.0	,	14.2	],\
   [	14.0	,	13.7	],\
   [	15.0	,	13.3	],\
   [	16.0	,	12.7	],\
   [	17.0	,	12.1	],\
   [	18.0	,	11.5	],\
   [	19.0	,	10.7	],\
   [	19.5	,	10.0	],\
   [	19.8	,	9.0	],\
   [	20.0	,	8.0	],\
   [	20.0	,	7.0	],\
   [	20.0	,	6.0	],\
   [	19.7	,	5.2	],\
   [	19.2	,	4.5	],\
   [	18.5	,	3.8	],\
   [	17.8	,	3.3	],\
   [	17.0	,	2.7	],\
   [	16.0	,	2.0	],\
   [	15.0	,	1.4	],\
   [	14.0	,	0.9	],\
   [	13.0	,	0.5	],\
   [	12.0	,	0.2	],\
   [	11.0	,	0.0	],\
   [	10.0	,	0.0	],\
   [	9.0	,	0.0	],\
   [	8.0	,	0.0	],\
   [	7.0	,	0.0	],\
   [	6.0	,	0.0	],\
   [	5.0	,	0.0	],\
   [	4.0	,	0.0	],\
   [	3.0	,	0.0	],\
   [	2.0	,	0.0	],\
   [	1.0	,	0.0	],\
   [	0.0	,	15.0	],\
   [	1.0	,	15.0	],\
   [	2.0	,	15.0	],\
   [	3.0	,	15.0	],\
   [	4.0	,	15.0	],\
   [	5.0	,	15.0	],\
   [	6.0	,	15.0	],\
   [	7.0	,	15.0	],\
   [	8.0	,	15.0	]]


   # 「C」のパターン
SAMPLE_CLOUD_C = [[	26.71553615	,	7.510273536	],\
   [	26.27962084	,	6.816161153	],\
   [	25.82059332	,	6.149062758	],\
   [	25.33939417	,	5.510345251	],\
   [	24.83700935	,	4.901317379	],\
   [	24.31446828	,	4.323227053	],\
   [	23.77284165	,	3.777258795	],\
   [	23.21323927	,	3.264531306	],\
   [	22.63680777	,	2.786095176	],\
   [	22.04472828	,	2.342930731	],\
   [	21.43821397	,	1.935946027	],\
   [	20.81850762	,	1.565974983	],\
   [	20.18687901	,	1.233775679	],\
   [	19.54462237	,	0.940028799	],\
   [	18.8930537	,	0.685336237	],\
   [	18.23350807	,	0.470219864	],\
   [	17.5673369	,	0.295120458	],\
   [	16.8959052	,	0.160396802	],\
   [	16.22058875	,	0.066324949	],\
   [	15.54277129	,	0.013097652	],\
   [	14.86384167	,	0.000823977	],\
   [	14.18519105	,	0.029529071	],\
   [	13.50820999	,	0.099154119	],\
   [	12.83428564	,	0.209556456	],\
   [	12.16479889	,	0.360509866	],\
   [	11.50112153	,	0.551705042	],\
   [	10.84461345	,	0.782750219	],\
   [	10.19661986	,	1.053171983	],\
   [	9.558468499	,	1.362416231	],\
   [	8.931466963	,	1.709849316	],\
   [	8.316899991	,	2.094759339	],\
   [	7.716026844	,	2.51635761	],\
   [	7.130078725	,	2.973780266	],\
   [	6.560256255	,	3.466090036	],\
   [	6.007727013	,	3.992278167	],\
   [	5.473623145	,	4.551266487	],\
   [	4.959039042	,	5.141909617	],\
   [	4.465029097	,	5.762997314	],\
   [	3.99260555	,	6.413256956	],\
   [	3.542736407	,	7.091356146	],\
   [	3.116343459	,	7.795905444	],\
   [	2.714300396	,	8.525461211	],\
   [	2.337431013	,	9.278528571	],\
   [	1.986507524	,	10.05356447	],\
   [	1.66224898	,	10.84898085	],\
   [	1.365319794	,	11.66314787	],\
   [	1.09632838	,	12.4943973	],\
   [	0.855825908	,	13.34102587	],\
   [	0.644305173	,	14.20129884	],\
   [	0.462199585	,	15.07345348	],\
   [	0.309882285	,	15.95570272	],\
   [	0.187665372	,	16.84623883	],\
   [	0.095799272	,	17.74323706	],\
   [	0.034472222	,	18.64485945	],\
   [	0.003809881	,	19.54925856	],\
   [	0.003875078	,	20.45458124	],\
   [	0.034667678	,	21.35897248	],\
   [	0.096124587	,	22.26057915	],\
   [	0.188119879	,	23.15755385	],\
   [	0.310465053	,	24.04805864	],\
   [	0.462909421	,	24.93026887	],\
   [	0.645140621	,	25.80237686	],\
   [	0.856785257	,	26.66259565	],\
   [	1.097409664	,	27.50916263	],\
   [	1.366520797	,	28.34034317	],\
   [	1.663567242	,	29.15443415	],\
   [	1.987940344	,	29.94976748	],\
   [	2.338975454	,	30.72471351	],\
   [	2.715953294	,	31.47768436	],\
   [	3.118101427	,	32.20713717	],\
   [	3.544595842	,	32.91157727	],\
   [	3.994562644	,	33.58956126	],\
   [	4.467079839	,	34.23969993	],\
   [	4.961179229	,	34.86066112	],\
   [	5.475848393	,	35.45117248	],\
   [	6.010032762	,	36.01002403	],\
   [	6.56263778	,	36.53607067	],\
   [	7.132531146	,	37.02823453	],\
   [	7.718545137	,	37.48550715	],\
   [	8.319478996	,	37.90695155	],\
   [	8.934101395	,	38.29170421	],\
   [	9.56115296	,	38.63897674	],\
   [	10.19934885	,	38.94805758	],\
   [	10.84738138	,	39.21831342	],\
   [	11.50392272	,	39.4491905	],\
   [	12.1676276	,	39.64021573	],\
   [	12.83713609	,	39.79099771	],\
   [	13.51107632	,	39.90122749	],\
   [	14.1880674	,	39.97067919	],\
   [	14.86672215	,	39.99921052	],\
   [	15.54564998	,	39.986763	],\
   [	16.22345977	,	39.93336215	],\
   [	16.89876266	,	39.83911738	],\
   [	17.57017494	,	39.70422181	],\
   [	18.23632087	,	39.52895183	],\
   [	18.89583551	,	39.31366658	],\
   [	19.54736749	,	39.05880719	],\
   [	20.18958181	,	38.76489586	],\
   [	20.82116256	,	38.43253483	],\
   [	21.44081561	,	38.06240512	],\
   [	22.04727128	,	37.65526512	],\
   [	22.63928694	,	37.21194908	],\
   [	23.21564952	,	36.73336537	],\
   [	23.77517804	,	36.2204946	],\
   [	24.31672603	,	35.67438767	],\
   [	24.83918383	,	35.09616355	],\
   [	25.34148091	,	34.48700706	],\
   [	25.82258807	,	33.84816635	],\
   [	26.28151949	,	33.18095043	],\
   [	26.71733482	,	32.48672644	]]
