# convert darknet cfg/weights to pytorch model
from models import *;
from os import listdir
from os.path import isfile, join


# onlyfiles = [f for f in listdir('../Aerial Yolov3/cfg')]
# for f in onlyfiles:
#     try:
#         convert(f, '../Aerial PreTrained/yolov3-aerial.weights')
#         print(" ------------------------- can be converted to ", f)
#     except:
#         continue
# print("done")


convert('../Aerial Yolov3/cfg/yolov3-aerial.cfg', '../Aerial PreTrained/yolov3-aerial.weights') #vice versa