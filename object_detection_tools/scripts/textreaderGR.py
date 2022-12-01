import os
import re
x_fullscale = 1280.
y_fullscale = 720.

# GRファクトリーデータに特化したテキストファイル読み込み
# frame_000123.txt の形のファイル専用
def reader(name):
    if os.path.isfile(name) == False:
        return [], False
    # readlineメソッドを使ってテキストファイルから1行ずつ内容を読み込む（返り値は文字列）
    list = []
    with open(name) as f:
        line = f.readline()
        while line != '':
            nums = line.split(' ')  # スペースで区切る
            classnum = int(nums[0])
            x = float(nums[1])*x_fullscale
            y = float(nums[2])*y_fullscale
            dx = float(nums[3])*x_fullscale
            dy = float(nums[4])*y_fullscale
            ret = {'class': classnum, 'x': x, 'y': y, 'size_x': dx, 'size_y': dy}
            list.append(ret)
            #print('class=%d, x=%.2f, y=%.2f, dx=%.2f, dy=%.2f' % (classnum, x, y, dx, dy))
            line = f.readline()
        #print('endread %s' % name)
    return list, True 

# AI_mindGR.py の出力を読み込む
# 比較的汎用に読める
# データはスペースで区切られ、「=」または「:」で、「ID:3」「time=0.1」のように表されている事
# ひとつのデータセットの中にスペースがあってはならない。例えば「class:cell phone」は{'class':cell, phone':0}に分割されてしまう
# また、「:」の後にスペースを置いてもいけない
def readerUni(name):
    if os.path.isfile(name) == False:
        return [], False
    # readlineメソッドを使ってテキストファイルから1行ずつ内容を読み込む（返り値は文字列）
    datalist = []
    with open(name) as f:
        line = f.readline()
        #print(line)
        while line != '':
            dictionary = {}
            items = line.split(' ')  # スペースで区切る
            for item in items:
                #print(item)
                itemtexts = re.split('[=:,]', item)
                #print(itemtexts)
                if len(itemtexts) < 2:
                    value = 0
                else:
                    value = itemtexts[1]
                dictionary[itemtexts[0]] = value
            datalist.append(dictionary)
            line = f.readline()
    return datalist, True 

def testmain1():
    # ファイル名の文字列
    FILENAME_TOP = 'frame_'
    RES_10 = '00000'
    RES_100 = '0000'
    RES_1000 = '000'
    RES_10000 = '00'
    RES_100000 = '0'
    RES_1000000 = ''
    top = FILENAME_TOP
    for num in range(10):
        numstr = str(num)
        if num < 10:
            res = RES_10
        elif num < 100:
            res = RES_100
        elif num < 1000:
            res = RES_1000
        elif num < 10000:
            res = RES_10000
        elif num < 100000:
            res = RES_100000
        else:
            res = RES_1000000
        filename = top + res + numstr + '.txt'
        objlist, flag = reader(filename)
        if  flag == False:
            print('no file')
            break
        else:
            print(objlist)

def testmain2():
    datalist, isok = readerUni('record7__.txt')
    print(datalist)

if __name__ == '__main__':
    testmain2()
    
