#-*-coding:gb2312-*-

import os
import shutil


def imag_rename(old_path, new_path,start_number = 0):
    filelist = os.listdir(old_path)  # 该文件夹下所有的文件（包括文件夹）
    if os.path.exists(new_path) == False:
        os.mkdir(new_path)
    for file in filelist:  # 遍历所有文件
        Olddir = os.path.join(old_path, file)  # 原来的文件路径
        if os.path.isdir(Olddir):  # 如果是文件夹则跳过
            continue
        filename = os.path.splitext(file)[0]  # 文件名
        filetype = os.path.splitext(file)[1]  # 文件扩展名
        if filetype == '.jpg':
            Newdir = os.path.join(new_path, str(int(filename) + start_number).zfill(6) + filetype)
            # 用字符串函数zfill 以0补全所需位数
            shutil.copyfile(Olddir, Newdir)


if __name__ == "__main__":
    # 解决positive image set文件夹中的重命名问题，start_number = 0
    old_path = "C:/Users/Yezizhuang/Desktop/WEIXING/positive image set/"
    new_path = "D:/nwpu vhr-10/images"
    imag_rename(old_path, new_path)
    # 解决negative image set文件夹中的重命名问题，start_number = 650
    old_path = "/Users/Yezizhuang/Desktop/WEIXING/negative image set/"
    new_path = "D:/nwpu vhr-10/images"
    imag_rename(old_path,new_path,start_number = 650)
    print("done!")

