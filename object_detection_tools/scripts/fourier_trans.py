
import math
import time
import copy

# 簡易フーリエ変換。要するにラプラス変換 
# 減衰時定数 decay_time がある事が特徴
# 出力は、ラプラス変換に対して減衰時定数で割った値になってる。要するに入力と同次元になってる。
# 定常値に対してゼロにならない「残渣」（AIノート2021Sep21)があるので、参照値として二乗値 Ref_sqr と絶対値 Ref_abs の変換も算出。
# time を使うので、周波数の単位は Hz、時間の単位は秒
# ただし、time の値を入力する事で、時間ではない単位（距離とか）でもって評価もできる
# フーリエ変換したい周波数のリストを持ち、周波数毎の減衰時定数のリストも与える
# 減衰時定数のリストは短くても良く、リストの末尾の値が次の周波数成分に受け継がれる
# 減衰時定数は周波数の逆数より十分大きい事が好ましく、上記のリストの性質上、低周波から順に格納するのが良い。
# データ本体は、components で、その値はデョクショなりで "Real"(cos成分)、"Imagin"(sin成分)、"Ref_sqr"、"Ref_abs" を格納。
# 「残渣」の推定値は、Residue*Ref_sqr または、Residue*Ref_abs**2 になる。前者の方がピーク値の影響で大きい値になる。
# 突発的なピーク値に反応したくない時は Ref_sqr を、反応したいときは Ref_abs を残渣として採用すると良い。
class Fourier:
    def __init__(self, frequency_list, decay_time_list, time_new=False):
        if time_new==False:
            time_new=time.time()
        self.components = []
        n = 0
        for freq in frequency_list:
            n = min(n, len(decay_time_list)-1)
            # 残渣評価用の値。無次元化した演算なので、2021Sep21のノートと式が違う。
            tomega = 2*math.pi*freq*decay_time_list[n]
            #so = -1.0/tomega/(1.0 - 1.0/tomega**2)
            #co = -so/tomega
            #residue = so**2 + co**2
            # DC 成分の影響が除去しきれず、DC成分取りを追加 2022Jan26
            # residue は廃止 2022Jan27
            self.components.append({"frequency": freq, \
                                    "decay_time": decay_time_list[n], \
                                    "INIT": False, \
                                    "DC_compo": 0.0, \
                                    "Real": 0.0, \
                                    "Imagin": 0.0, \
                                    "Ref_sqr": 0.0,\
                                    "Ref_abs": 0.0,\
                                    "Real_sqr_peak": 0.0, \
                                    "Imagin_sqr_peak": 0.0, \
                                    "Ref_sqr_peak": 0.0,\
                                    "Ref_abs_peak": 0.0 })
            n += 1
        self.time_last = time_new # 時刻の前回値
        self.time_ref  = time_new # 基準時刻（位相の出発点）

    ###################################
    # 改めて初期化したいときに使う
    def init_fourier(self, time_new=False):
        if time_new==False:
            time_new=time.time()
        self.time_last = time_new
        self.time_ref  = time_new
        for compo in self.components:
            compo["INIT"] = False
            compo["DC_compo"] = 0.0
            compo["Real"] = 0.0
            compo["Imagin"] = 0.0
            compo["Ref_sqr"] = 0.0
            compo["Ref_abs"] = 0.0
            compo["Real_sqr_peak"] = 0.0
            compo["Imagin_sqr_peak"] = 0.0
            compo["Ref_sqr_peak"] = 0.0
            compo["Ref_abs_peak"] = 0.0

    ###################################
    # 位相だけリセットしたいときに使う
    def reset_phase(self, time_new=False):
        if time_new==False:
            time_new=time.time()
        self.time_ref  = time_new

    ####################################
    # フーリエ変換本体。
    # time_now は省略可
    def fourier_trans(self, imput, time_now=False):
        if time_now==False:
            time_now=time.time()
        for compo in self.components:
            phase = 2.0*math.pi*(time_now - self.time_ref)*compo["frequency"]
            # 1超え防止追加 2022Jan26
            decay = max(0.,min(1., (time_now - self.time_last)/compo["decay_time"]))
            if compo["INIT"]==False:
                compo["DC_compo"] = imput
            else:   # 減衰時間3倍でDC成分を抽出
                compo["DC_compo"] = (1.0 - decay/3.)*compo["DC_compo"] + imput*decay/3.
            compo["INIT"] = True
            imput_AC = imput - compo["DC_compo"]
            compo["Real"]    = (1.0 - decay)*compo["Real"]   + math.cos(phase)*imput_AC*decay
            compo["Imagin"]  = (1.0 - decay)*compo["Imagin"] + math.sin(phase)*imput_AC*decay
            compo["Ref_sqr"] = (1.0 - decay)*compo["Ref_sqr"] + imput_AC**2*decay
            compo["Ref_abs"] = (1.0 - decay)*compo["Ref_abs"] + abs(imput_AC)*decay
            compo["Real_sqr_peak"]   = max(compo["Real_sqr_peak"],   compo["Real"]**2)
            compo["Imagin_sqr_peak"] = max(compo["Imagin_sqr_peak"], compo["Imagin"]**2)
            compo["Ref_sqr_peak"]    = max(compo["Ref_sqr_peak"],    compo["Ref_sqr"])
            compo["Ref_abs_peak"]    = max(compo["Ref_abs_peak"],    compo["Ref_abs"])
        self.time_last = copy.copy(time_now)

####################################################################################
# 鋭角的な変化を検出するには、フーリエ変換は不向き。
# フィルターを２つ使い、短い時定数のフィルター値から長い時定数のフィルター値を差し引いてインパルスを評価する
# 短い時定数と長い時定数の比は、filt_ratio で指定。
# 2022Mar4 仕様大幅見直し。絶対値に着目する。ガサガサ動いている振幅の変化を捉えるイメージ
# filt_ratio のデフォルト値を５から２に変更。 比が大きすぎると位相差が出てダメ
# 様々な時定数のリストを与え、柔らかい変化とシャープな変化をそれぞれ検出できるように仕掛けた。
# パルス入力に対しては、"Diff"はあるフィルター時定数で最大値になり、
# ステップ応答に対しての"Diff"は、ステップのシャープさがフィルター時定数より短くなると信号が出る
# 検出範囲は非常にブロードなので、時定数は大きく変えるべきで、一桁刻みでも良いぐらい。
class Impulse:
    def __init__(self, time_const_list, filt_ratio=2.0, time_new=False):
        if time_new==False:
            time_new=time.time()
        self.filt_ratio = filt_ratio
        self.components = []
        for time_c in time_const_list:
            self.components.append({"time_const": time_c, \
                                    "Filted_long": 0.0, \
                                    "Filted": 0.0, \
                                    "Absolute": 0.0, \
#                                    "Windowed": 0.0,\
                                    "Absolute_long": 0.0, \
                                    "Absolute_diff": 0.0, \
#                                    "Windowed_peak_n": 0.0, \
                                    "Diff": 0.0, \
                                    "Diff_peak_p": 0.0,\
                                    "Diff_peak_n": 0.0 })
#                                        })
        self.time_last = time_new # 時刻の前回値

    ######################
    # インパルス検出本体    
    def impulse(self, imput, time_now=False):
        if time_now==False:
            time_now=time.time()
        for compo in self.components:
            decay_short = min(1., (time_now - self.time_last)/compo["time_const"])
            decay_long  = decay_short/self.filt_ratio
            compo['Filted_long']   = (1.0 - decay_long )*compo["Filted_long"]   + imput*decay_long
            compo['Filted']        = (1.0 - decay_short)*compo["Filted"]        + imput*decay_short
            compo['Absolute_long'] = (1.0 - decay_long )*compo["Absolute_long"] + abs(imput)*decay_long
            compo['Absolute']      = (1.0 - decay_short)*compo["Absolute"]      + abs(imput)*decay_short
            #abs_val = abs(compo['Filted'])
            #abs_residue = abs(compo['Filted_long'])
            #compo['Absolute'] = max(0.0, abs_val - abs_residue)
            #if abs_residue > abs_val:
            #    compo['Windowed'] = 0.0
            #else:
            #    compo['Windowed'] = compo['Filted'] - abs(abs_residue)*(compo['Filted'] > 0.0)
            compo["Diff"]            = compo['Filted']   - compo['Filted_long']
            compo["Absolute_diff"]   = compo["Absolute"] - compo['Absolute_long']
            #compo["Windowed_peak_p"] = max(compo["Windowed_peak_p"], compo['Windowed'])
            #compo["Windowed_peak_n"] = min(compo["Windowed_peak_n"], compo['Windowed'])
            compo["Diff_peak_p"] = max(compo["Diff_peak_p"], compo['Absolute_diff'])
            compo["Diff_peak_n"] = min(compo["Diff_peak_n"], compo['Absolute_diff'])
        self.time_last = copy.copy(time_now)

    # インパルス検出をリセットする。入力値の変化に反応するので、フィルタ値をimputにセットする
    def init_impulse(self, imput=0.0, time_now=False):
        if time_now==False:
            time_now=time.time()
        self.time_last = time_now
        for compo in self.components:
            compo['Filted_long'] = imput
            compo['Filted'] = imput
            compo['Absolute'] = abs(imput)
            #compo['Windowed'] = 0.0
            compo["Diff"] = 0.0
            compo["Absolute_long"] = abs(imput)
            compo["Absolute_diff"] = 0.0
            #compo["Windowed_peak_n"] = 0.0
            compo["Diff_peak_p"] = 0.0
            compo["Diff_peak_n"] = 0.0
        
    # 最大最小値のみをリセットする。
    def reset_max(self):
        #self.time_last = time_now
        for compo in self.components:
            compo["Diff_peak_p"] = 0.0
            compo["Diff_peak_n"] = 0.0


######### test program ########
# decay_time と周波数の間隔にはちょうど良い値がある（decay_timeが大きいと、周波数の刻みを細かくする必要）。
# 下記の設定(m=0.5)ぐらいでちょうどいい線。

def main1():
    k = 360.
    frequency_list =  [2./k,   3./k,   4./k,   5./k,   6./k,  8./k,  10./k, 12.5/k, 16.0/k, 20./k, 25./k, 30./k,  40./k,  50./k, 60./k]
    m = 0.5
    decay_time_list = [250.*m, 167.*m, 125.*m, 100.*m, 83.*m, 62.*m, 50.*m, 40.*m,  31.*m,  25.*m, 20.*m, 16.7*m, 12.5*m, 10.*m, 8.3*m]
    fl = Fourier(frequency_list, decay_time_list, time_new=0.)
    n = 9.
    for x in range(360):
        x_ = float(x)
        y = math.sin(n*x_/360*2*math.pi)
        fl.fourier_trans(imput=y, time_now=x_)
        
    for compo in fl.components:
        print("Frequency = ",compo["frequency"]*360.)
        print("Real    = ",compo["Real"])
        print("Imagin  = ",compo["Imagin"])
        #print("Residue = ",compo["Residue"])
        print("Ref_sqr = ",compo["Ref_sqr"])
        print("Ref_abs = ",compo["Ref_abs"])
        #print("Residue_removed = ", (compo["Real"]**2 + compo["Imagin"]**2)/compo["Ref_sqr"] - compo["Residue"])
        print("next")


def func1(x, pulse_length): 
    y = math.exp(-((x -200.)/pulse_length)**2) - math.exp(-((x -600.)/pulse_length)**2) + 2.
    return y

def func2(x, pulse_length):
    y = 1./(1. + math.exp(-((x -200.)/pulse_length))) + 1./(1. + math.exp(((x -600.)/pulse_length))) + 2.
    return y

def main2():
    filt_time_list =  [1., 3., 10., 30., 100., 300., 1000., 3000., 10000.]
    pl = Impulse(filt_time_list, filt_ratio=2.0, time_new=0.)
    pulse_length = 1.
    for n in range(6):
        y = func2(-1., pulse_length)
        pl.init_impulse(imput=y, time_now=-1.0)
        for x in range(1000):
            x_ = float(x)
            y = func2(x_, pulse_length)
            if x == 50:
                pl.reset_max()
            pl.impulse(imput=y, time_now=x_)
        print('Pulse_length = ', pulse_length)
        for compo in pl.components:
            print("time_const = ",compo["time_const"])
            #print("Absolute_peak    = ",compo["Absolute_peak"])
            print("Diff_peak_p  = ",compo["Diff_peak_p"])
            print("Diff_peak_n  = ",compo["Diff_peak_n"])
            print("next")
        print("\n")
        pulse_length *= 3.


if __name__ == "__main__":
    main2()