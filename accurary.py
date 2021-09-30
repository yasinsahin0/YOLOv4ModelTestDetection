import YoloV4TinyAccrary as yolo
import cv2
import os
import YoloV4TinyAccrary as yolodet
img = []
txt = []
red_tomato_count = 0
green_tomato_count = 0
dirs = os.listdir("test_image")

for file in dirs:
    if file.rfind(".txt") > 0:
        txt.append(file)
    elif file.rfind(".jpg") or file.rfind(".JPG") or file.rfind(".png") or file.rfind(".PNG") > 0:
        img.append(file)


def txt_Acc():
    red_data=[]
    green_data=[]
    if len(img) == len(txt):
        for i in range(0,len(img)):
            with open("test_image/"+txt[i],"r") as f:
                for colm in f.readlines():
                    s = colm.split()
                    if s[0] =="0":
                        green_data.append("0")
                    elif s[0] == "1":
                        red_data.append("1")
    test_red_c = len(red_data)
    test_green_c = len(green_data)
    return test_green_c,test_red_c

t_green,t_red = txt_Acc()
green = 0
red = 0
for i in img:
    img1 = cv2.imread("test_image/"+i)
    green_c ,red_c = yolodet.detech(img1)
    green += green_c
    red += red_c
top1 = t_green+t_red
top2 = green +red
print(top1)
print(top2)
a = top2*100/top1

print(a)