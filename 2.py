#-*-coding:gb2312-*-
import os
import random

trainval_percent = 0.8  # 表示训练集和验证集(交叉验证集)所占总图片的比例
train_percent = 0.75  # 训练集所占交叉验证集的比例
xmlfilepath = "D:/nwpu vhr-10/Annotations"
txtsavepath = "D:/nwpu vhr-10/ImageSets/Main"
total_xml = os.listdir(xmlfilepath)

num = 650  # 有目标的图片数
list = range(num)
tv = int(num * trainval_percent)  # xml文件中的交叉验证集数
tr = int(tv * train_percent)  # xml文件中的训练集数，注意，我们在前面定义的是训练集占交叉验证集的比例
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open("D:/nwpu vhr-10/ImageSets/Main/trainval.txt", 'w')
ftest = open('D:/nwpu vhr-10/ImageSets/Main/test.txt', 'w')
ftrain = open("D:/nwpu vhr-10/ImageSets/Main/train.txt", 'w')
fval = open('D:/nwpu vhr-10/ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

for i in range(150):
    num = str(651 + i).zfill(6) + '\n'
    ftest.write(num)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

print("done!")

