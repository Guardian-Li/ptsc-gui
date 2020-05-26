from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.point_line import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
###额外导入部分
from objecttracker.KalmanFilterTracker import Tracker  # 加载卡尔曼滤波函数
import colorsys



def changeBGR2RGB(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img


def changeRGB2BGR(img):
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r

    return img


def getXKey(x):
    return x[0]


def getYKey(x):
    return x[1]


#####################################额外导入的轨迹函数部分START#########################################################
def calc_center(out_boxes, out_classes, out_scores, score_limit=0.5):  ###添加一个种类参数
    outboxes_filter = []
    # print(float(out_scores))
    # print("---------------")
    # print(len(out_classes))
    # print("-----------------")
    for x, y, z in zip(out_boxes, out_classes, out_scores):

        if z > score_limit:
            ####存储需要追踪的物体一个为后续统计做基础

            if y == 1:
                outboxes_filter.append(x)

    centers = []
    number = len(outboxes_filter)
    # print(number)
    # print("length")
    for box in outboxes_filter:
        x1, y1, x2, y2 = box
        # top, left, bottom, right = box
        # print(float(x1))
        # print(x1)

        center = np.array([[(x1 + x2) // 2], [(y1 + y2) // 2]])
        centers.append(center)
    return centers, number


def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    # colors = [(255,99,71) if c==(255,0,0) else c for c in colors ]  # 单独修正颜色，可去除
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


def trackerDetection(tracker, image, centers, count, max_point_distance=30, max_colors=20, track_id_size=0.8):
    '''
        - max_point_distance为两个点之间的欧式距离不能超过30
            - 有多条轨迹,tracker.tracks;
            - 每条轨迹有多个点,tracker.tracks[i].trace
        - max_colors,最大颜色数量
        - track_id_size,每个
    '''
    # track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    #            (0, 255, 255), (255, 0, 255), (255, 127, 255),
    #            (127, 0, 255), (127, 0, 127)]
    track_colors = get_colors_for_classes(max_colors)

    result = np.asarray(image)

    # print("*****************************")
    # print("np,as")
    # print(result)
    # print("***************************")
    font = cv2.FONT_HERSHEY_SIMPLEX

    # cv2.putText(image, str(number), (20, 40), font, 1, (0, 0, 255), 5)  # 左上角，人数计数
    ##################################红绿灯线的位置START#############################
    RoadLine = Segment(Point(300,250), Point(800,250))
    ##################################红绿灯线的位置END###############################

    if (len(centers) > 0):
        # Track object using Kalman Filter
        tracker.Update(centers)
        # print(centers)
        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        road = 0
        for i in range(len(tracker.tracks)):
            # print(i)
            # print(outbox[i])
            # print(outbox)
            # print("------------------")

            # 多个轨迹
            if (len(tracker.tracks[i].trace) > 1):
                x0, y0 = tracker.tracks[i].trace[-1][0][0], tracker.tracks[i].trace[-1][1][0]
                ##铺设轨迹点
                # cv2.putText(result, str(tracker.tracks[i].track_id), (int(x0), int(y0)), font, track_id_size,
                #             (255, 255, 255), 4)

                # (image,text,(x,y),font,size,color,粗细)

                #############################绘制轨迹START############################

                for j in range(len(tracker.tracks[i].trace) - 1):
                    # 每条轨迹的每个点
                    # Draw trace line
                    x1 = tracker.tracks[i].trace[j][0][0]
                    y1 = tracker.tracks[i].trace[j][1][0]
                    x2 = tracker.tracks[i].trace[j + 1][0][0]
                    y2 = tracker.tracks[i].trace[j + 1][1][0]
                    clr = tracker.tracks[i].track_id % 9
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    ######################两次画面移动距离START###############
                    diverdistance = ((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    road = diverdistance
                    ######################两次画面移动距离END###############

                    ######################中心点的线段START#################
                    MidLine = Segment(Point(tracker.tracks[i].trace[0][0][0],tracker.tracks[i].trace[0][1][0]), Point(tracker.tracks[i].trace[j][0][0],tracker.tracks[i].trace[j][1][0]))
                    if segmentsIntersect(RoadLine,MidLine) and tracker.tracks[i].flag[i]==False:
                        tracker.tracks[i].flag[i] = True
                        count += 1
                    ######################中心点的线段START#################

                    # print(distance)
                    # print(x1,y1)
                    # print(x2,y2)
                    # print("----------------------------")
                    ############################添加显示速度
                    if distance < max_point_distance:
                        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 4)
                        # print(j)
                        # print("fenge---------------")
                        ############################类积分添加预测点START#####################
                        # cv2.line(result, (int(x2), int(y2)), (int(x1+x2), int(y1+y2)),
                        #          track_colors[clr], 4)
                        #############################类积分添加预测点END######################

                ################################绘制轨迹END#############################s

    return tracker, image, road,count


#############################################额外导入的轨迹函数部分END###################################################


class Vedio():
    def __init__(self,
                 car_weight_path="weights/car_num.pth",
                 vedio_file="data/vedio/video-01.mp4",
                 car_model_def="config/plate.cfg",
                 car_class_path="config/plate.names",
                 model_def="config/ptsc.cfg",
                 weights_path="weights/ptsc-new-20-epoch.pth",
                 class_path="config/ptsc.names",
                 conf_thres=0.8,
                 n_cpu=8,
                 nms_thres=0.1,
                 img_size=416,
                 batch_size=32):
        self.n_cpu = n_cpu
        self.car_class_path = car_class_path
        self.car_model_def = car_model_def
        self.batch_size=32
        
        self.vedio_file = vedio_file
        self.model_def = model_def
        self.weights_path = weights_path
        self.class_path = class_path
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.img_size = img_size
        self.plate_classes=car_class_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Darknet(model_def, img_size=img_size).to(device)
        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            print(self.weights_path)
            model.load_state_dict(torch.load(self.weights_path))
        model.eval()
        self.model = model
        model_plate = Darknet(self.car_model_def, img_size=img_size).to(device)
        self.model_plate=model_plate
        #if car_weight_path.endswith(".weights"):
            # Load darknet weights
            #print("fuck")
            #model_plate.load_darknet_weights(self.weights_path)
        #else:
            # Load checkpoint weights
            #print(self.weights_path)
        model_plate.load_state_dict(torch.load(car_weight_path))
        model_plate.eval()
        # model_plate.eval()
        # self.model_plate=model_plate


    def play_vedio(self):
        tracker = Tracker(100, 8, 15, 100)
        classes = load_classes(self.class_path)
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # if opt.vedio_file.endswith(".mp4"):
        cap = cv2.VideoCapture(self.vedio_file)
        colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
        a = []
        time_begin = time.time()
        NUM = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # NUM=0
        ####################初始化人流总数##################
        count = 0
        ####################初始化人流总数##################
        while cap.isOpened():
            ret, img = cap.read()
            if ret is False:
                break
            img = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_CUBIC)
            # PILimg = np.array(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
            # imgTensor = transforms.ToTensor()(PILimg)
            RGBimg = changeBGR2RGB(img)
            imgTensor = transforms.ToTensor()(RGBimg)
            imgTensor, _ = pad_to_square(imgTensor, 0)
            imgTensor = resize(imgTensor, self.img_size)
            imgTensor = imgTensor.unsqueeze(0)
            imgTensor = Variable(imgTensor.type(Tensor))
            with torch.no_grad():
                detections = self.model(imgTensor)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            a.clear()
            if detections is not None:
                a.extend(detections)
            b = len(a)
            if len(a):
                for detections in a:
                    #############################图片的对角坐标保存，轨迹的变量START#########################
                    out_boxs = []
                    out_classes = []
                    out_scores = []
                    ###############################图片的对角坐标保存，轨迹的变量END######################
                if detections is not None:
                    #print(detections)
                    detections = rescale_boxes(detections, self.img_size, RGBimg.shape[:2])
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        #print(x1, " ", y1)
                        #print(x2, " ", y2)
                        #############################图片的对角坐标保存，轨迹的变量START##########################################
                        out_boxs.append((x1, y1, x2, y2))
                        out_classes.append(int(cls_pred))
                        out_scores.append(cls_conf.item())
                        ############################图片的对角坐标保存，轨迹的变量END#############################################
                        box_w = x2 - x1
                        box_h = y2 - y1
                        color = [int(c) for c in colors[int(cls_pred)]]
                        # print(cls_conf)
                        img = cv2.rectangle(img, (x1, y1 + box_h), (x2, y1), color, 2)
                        cv2.putText(img, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(img, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color, 2)
                        # 识别车牌
                        if classes[int(cls_pred)] == "car":

                            car_classes = load_classes(self.car_class_path)
                            car_img = img[int(y1):int(y1) + int(box_h), int(x1):int(x1 + box_w)].copy()
                            car_img = changeBGR2RGB(car_img)
                            imgCarTensor = transforms.ToTensor()(car_img)
                            imgCarTensor, _ = pad_to_square(imgCarTensor, 0)
                            imgCarTensor = resize(imgCarTensor, self.img_size)
                            imgCarTensor = imgCarTensor.unsqueeze(0)
                            imgCarTensor = Variable(imgCarTensor.type(Tensor))
                            # if car_img is not None:
                            # cv2.imshow("fuck",car_img)
                            with torch.no_grad():
                                plate_detections = self.model_plate(imgCarTensor)
                                plate_detections = non_max_suppression(plate_detections, self.conf_thres, self.nms_thres)
                            res=[]
                            if plate_detections[0] is not None:

                                plate_detections = rescale_boxes(plate_detections[0], self.img_size, car_img.shape[:2])
                                for x1, y1, x2, y2, conf, cls_conf, cls_pred in plate_detections:
                                    box_w = x2 - x1
                                    box_h = y2 - y1
                                    #print(car_classes[int(cls_pred)])
                                    if cls_pred != 0:
                                        car_num_single = (float(x1), float(y1) + box_h, car_classes[int(cls_pred)])
                                        res.append(car_num_single)
                                res.sort(key=getYKey)
                                res.sort(key=getXKey)
                                plate_pre=""
                                for x in res:
                                    plate_pre = plate_pre + x[2]
                                print(plate_pre)



                        # end识别车牌
                        ################################步骤二载入数据START#################################################
                        centers, number = calc_center(out_boxs, out_classes, out_scores, score_limit=0.6)
                        tracker, result, road,count = trackerDetection(tracker, img, centers, count, max_point_distance=20)
                        yield number
                        ################################步骤二载入数据END####################################################################


                        #################################绘制车辆得车速检测START#######################################
                        #########由于数据得不准确，所以采用初始值优化速度###############################################
                        bestroad = 2.5
                        if road != 0 and road <= 7:
                            bestroad = road
                        #########由于数据得不准确，所以采用初始值优化速度###############################################
                        cv2.putText(result, str(round(bestroad * 20, 2)) + "km/h", (int(x2), int(y2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        #################################绘制车辆得车速检测END#########################################

                        #cv2.putText(result, str(round(road * 20, 2)) + "km/h", (int(x2), int(y2)),
                         #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            cv2.imshow('frame', changeRGB2BGR(RGBimg))
            # cv2.waitKey(0)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        time_end = time.time()
        time_total = time_end - time_begin
        print(NUM // time_total)
        ####################输出人流总数##################
        print("人流总人数："+str(int(count/11*3600)))
        yield "人流总人数："+str(int(count/11*3600))
        ####################输出人流总数##################

        cap.release()
        cv2.destroyAllWindows()

    def detect(self,img_folder):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs("output", exist_ok=True)
        model=self.model_plate


        dataloader = DataLoader(
            ImageFolder(img_folder, img_size=self.img_size),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_cpu,
        )

        classes = load_classes(self.plate_classes)  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        print("\nPerforming object detection:")
        prev_time = time.time()
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = self.model_plate(input_imgs)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            print(detections)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            # Save image and detections
            imgs.extend(img_paths)
            img_detections.extend(detections)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            print("(%d) Image: '%s'" % (img_i, path))

            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, self.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                    print(x1, y1)
                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = path.split("/")[-1].split(".")[0]
            plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)

            plt.close()

if __name__ == "__main__":
    v = Vedio()
    v.detect(img_folder="data/samples")

