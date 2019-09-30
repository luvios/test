#! /usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import queue
import threading
import subprocess as sp
import numpy as np
import io

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
import global_value as gl
import asyncio
from threading import Thread

import copy
import matplotlib.pyplot as plt
import requests
import subprocess as sp

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from tools.mysql import Mysql
from yolo import YOLO

import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

print(sys.path)


VIDEO_HEIGHT                = 640
VIDEO_WEIGHT                = 360

WEAR_FLAG                   = 1             # 佩戴标志
NOT_WEAR_FLAG               = 0             # 未佩戴标志
WEAR_BELT_IOU_THRESH        = 0.75          # 安全带佩戴iou阈值

CLIMB_FLAG                  = 0.0
MAX_COUTN                   = 10
EVNENT_NOTE_COUNT_THRESH    = 10

DEBUG=True
#DEBUG=False
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cmap = plt.get_cmap('RdYlBu')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 80)]

# 显示颜色
#color_red=(255, 0, 0)
color_blue=(255, 0, 0)
#color_blue=(0, 0, 255)
color_red=(0, 0, 255)
color_white=(255, 255, 255)
color_green=(0, 255, 0)

# 异常检测状态记录
ABNORMAL_DETECT_STATUS_DELTA = {}

# 画了检测框的视频帧推rtmp
# rtmpUrl = 'rtmp://127.0.0.1:1935/mylive/stream1'
# sizeStr = "{}x{}".format(VIDEO_HEIGHT, VIDEO_WEIGHT)
# rtmp_fps = 11
# command = ['ffmpeg',
#              '-y',
#              '-f', 'rawvideo',
#              '-vcodec', 'rawvideo',
#              '-pix_fmt', 'bgr24',
#              '-s', sizeStr,
#              '-r', str(rtmp_fps),
#              '-i', '-',
#              '-c:v', 'libx264',
#              '-pix_fmt', 'yuv420p',
#              '-preset', 'ultrafast',
#              '-f', 'flv',
#              rtmpUrl]
#
# pipe = sp.Popen(command, stdin=sp.PIPE)  #管道配置



def note_event(image,post_data,image_path):
    '''
    记录事件

    :param image:
    :param post_data:
    :param image_path:
    :return:
    '''
    # cv2.imwrite(image_path, image)
    url_request = "http://www.kangni.com/resource/addEvent"
    # files = {'file': ('pic.jpg', open(image_path, 'rb'), 'image/jpeg')}
    files = {'file': ('pic.jpg', io.BytesIO(cv2.imencode('.jpg', image)[1].tostring()), 'image/jpeg')}
    # files = {'file': ('pic.jpg', image, 'image/jpeg')}
    r = requests.post(url_request, files=files, data=post_data)
    print(r.text)


async def add_event(image,post_data,image_path):
    '''
    异步添加事件
    :param image:
    :param post_data:
    :param image_path:
    :return:
    '''
    print('bbbbbbbb')

    await my_post(image,post_data,image_path)

    print('eeeeeeee')


async def my_post(image,post_data,image_path):

    cv2.imwrite(image_path, image)
    url_request = "http://www.kangni.com/resource/addEvent"
    files = {'file': ('pic.jpg', open(image_path, 'rb'), 'image/jpeg')}
    r = requests.post(url_request, files=files, data=post_data)
    print(r.text)


def cal_box_area(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return area


def cal_my_iou(box1, box2):
    # 计算并集
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)

    # 计算交集
    box1_area = cal_box_area(box1)
    # box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # union_area = box1_area + box2_area - inter_area

    my_iou = inter_area / box1_area

    return box1_area,my_iou


def person_status_match(single_person_box, other_boxs, match_dict, max_boxes_to_draw=20):
    '''
    工程师与安全帽/安全带关联
    :param boxes:
    :return:
    '''

    # 是否佩戴安全帽/安全带
    val_list = list(match_dict.values())
    have_matched = []
    for i in range(len(val_list)):
        have_matched.extend(val_list[i])
    have_matched = [have_matched[i][-1] for i in range(len(have_matched))]

    match_ok = False
    have_hat = False
    have_belt= False
    hat_list=[]
    belt_list=[]
    for i in range(len(other_boxs)):
        if i in have_matched:
            continue
        # 安全帽佩戴
        wear_hat_status = (other_boxs[i][-1]=='hat' and other_boxs[i][2] >= single_person_box[0] and other_boxs[i][0] <=
                single_person_box[2] and other_boxs[i][3] <= single_person_box[1] + (
                        single_person_box[3] - single_person_box[1]) * 0.3 and
                other_boxs[i][1] >= single_person_box[1] - (single_person_box[3] - single_person_box[1]) * 0.3)
        # 安全带佩戴
        wear_belt_status =(other_boxs[i][-1]=='belt' and other_boxs[i][0]>=0.75*single_person_box[0] and other_boxs[i][0]<= (single_person_box[0]+single_person_box[2])/2 and other_boxs[i][1] >single_person_box[1]
                           and other_boxs[i][2]>=(single_person_box[0]+single_person_box[0])/2 and other_boxs[i][3]<=single_person_box[3]+0*single_person_box[1])
                           #and other_boxs[i][2]>=(single_person_box[0]+single_person_box[2])/2 and other_boxs[i][3]<=0.75*single_person_box[3]+0.25*single_person_box[1])
        obj_area,iou=cal_my_iou(other_boxs[i],single_person_box)
        wear_belt_status = wear_belt_status and (iou>WEAR_BELT_IOU_THRESH)

        if wear_hat_status:
            if have_hat is False:
                have_hat = True
                if single_person_box[4] not in match_dict.keys():
                    match_dict[single_person_box[4]] = [(other_boxs[i][-1], i)]
                else:
                    match_dict[single_person_box[4]].append((other_boxs[i][-1], i))
            match_ok = True

        if  wear_belt_status:
            if have_belt is False:
                have_belt = True
                belt_list.append(obj_area)
                if single_person_box[4] not in match_dict.keys():
                    match_dict[single_person_box[4]] = [(other_boxs[i][-1], i)]
                else:
                    match_dict[single_person_box[4]].append((other_boxs[i][-1], i))
                # print('匹配成功')
            else:
                if obj_area> max(belt_list):
                   match_dict[single_person_box[4]] = [(other_boxs[i][-1], i)]
                   belt_list.append(obj_area)
            match_ok = True

    if match_ok is False:
        match_dict[single_person_box[4]] = []

    # 佩戴状态
    current_wear = match_dict[single_person_box[4]]
    current_have_list = [current_wear[i][0] for i in range(len(current_wear))]
    current_state = fix_status(current_have_list)

    return match_ok,current_state


def fix_status(wear_list):
    '''
    状态确定
    :param wear_list:
    :return:
    '''

    # 标准佩戴：安全带、安全帽
    message = []
    is_save = False
    standard_status = gl.DETECT_TYPE
    state = [NOT_WEAR_FLAG] * len(standard_status)
    for i in range(len(standard_status)):
        if standard_status[i] in wear_list:
            state[i] = WEAR_FLAG

    return state


def status_change_note1(befor_status, current_status):
    '''
    判断摄像头下的目标是否发生状态变化
    :param befor_status:
    :param current_status:
    :return:
    '''

    state = [-1, -1]
    for id, wear in current_status.items():
        have_list = [wear[i][0] for i in range(len(wear))]

        # 该工程师id未变（一直在视野内）
        if id in befor_status.keys():
            if have_list.sort() != befor_status[id].sort():
                state = fix_status(have_list)
        else:
            # 新出现的工程师
            state = fix_status(have_list)

    # 老工程师离开视野
    for id, have_list in befor_status.items():
        if id not in current_status.keys():
            state = fix_status(have_list)

    return state


def status_change_note(before_status, current_status, id):
    '''
    判断摄像头下的目标是否发生状态变化
    :param befor_status:
    :param current_status:
    :return:
    '''

    add_flag = False

    current_wear = current_status[id]
    current_have_list = [current_wear[i][0] for i in range(len(current_wear))]
    current_state = fix_status(current_have_list)

    unwear_list=[]
    event_id_list = []
    if id in before_status.keys():
        if NOT_WEAR_FLAG in current_state:
            before__wear = before_status[id]
            before_have_list = [before__wear[i][0] for i in range(len(before__wear))]
            before_state = fix_status(before_have_list)
            print('detect item:',current_state,' ',before_state)
            delta = [current_state[i] - before_state[i] for i in range(len(current_state))]
         
            if -1 in delta:
                unwear_list=[i for i in range(len(delta)) if delta[i]==-1]
                unwear_list = [gl.DETECT_TYPE[i] for i in unwear_list]
                # event_id_list = [gl.NOTE_EVENT_ID[i] for i in unwear_list]
                add_flag = True
    else:
        if NOT_WEAR_FLAG in current_state:
            unwear_list = [i for i in range(len(current_state)) if current_state[i] == NOT_WEAR_FLAG]
            unwear_list=[gl.DETECT_TYPE[i] for i in unwear_list]
            # event_id_list = [gl.NOTE_EVENT_ID[i] for i in unwear_list]
            add_flag = True

    return add_flag, current_state, unwear_list, event_id_list


def engineer_status(current_status):
    '''
    判断摄像头下的目标是否发生状态变化
    :param befor_status:
    :param current_status:
    :return:
    '''

    state = [-1, -1]
    for id, wear in current_status.items():
        have_list = [wear[i][0] for i in range(len(wear))]
        state = fix_status(have_list)

    return state

def note_violation_behavior(worker_id,status):
    global ABNORMAL_DETECT_STATUS_DELTA

    if worker_id in ABNORMAL_DETECT_STATUS_DELTA.keys():
        if [status[i]-ABNORMAL_DETECT_STATUS_DELTA[worker_id]['status'][i] for i in range(len(status))]!=[NOT_WEAR_FLAG]*len(gl.DETECT_TYPE):
            ABNORMAL_DETECT_STATUS_DELTA.pop(worker_id)
        else:
            ABNORMAL_DETECT_STATUS_DELTA[worker_id]['count']+=1
    else:
        ABNORMAL_DETECT_STATUS_DELTA.update({worker_id:{'status':status,'count':1}})










# video_capture = cv2.VideoCapture("/Users/apple/Desktop/Git/Track/deep_sort_yolov3_lastest/model_data/VID_20190627_170344.mp4")
# if video_capture.isOpened() != True:
#     video_capture.open("model_data/VID_20190627_170344.mp4")

class Live(object):

    def __init__(self,camera_id,detect_items,detect_items_index,camera_rtcp,gpu_mem_rate):
        self.frame_queue = queue.Queue()
        self.post_queue = queue.Queue()
        self.command = ""

        self.rtmpUrl = 'rtmp://127.0.0.1:1935/mylive/stream1'
        #self.camera_path = "D:/20190719-185520.mp4"
        # self.camera_path = "rtsp://admin:snd123456@192.168.0.64:554/h264/main/av_stream?tcp"
        self.camera_path = "/Users/apple/Desktop/Git/Track/deep_sort_yolov3_lastest/model_data/VID_20190627_170344.mp4"

        self.camera_id = camera_id
        self.camera_rtcp = camera_rtcp
        self.gpu_mem_rate = gpu_mem_rate
        # 检测项以及检测编号
        self.detect_items = detect_items
        self.detect_items_index = detect_items_index


    def read_frame(self):
        print("开启推流")
        # Get video information
        #fps = int(cap.get(cv.CAP_PROP_FPS))
        #width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        #height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        sizeStr = "{}x{}".format(640, 360)
        rtmp_fps = 20

        # ffmpeg command
        self.command = ['ffmpeg',
                        '-y',
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-s', sizeStr,
                        '-r', str(rtmp_fps),
                        '-i', '-',
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', 'ultrafast',
                        '-f', 'flv',
                        self.rtmpUrl]

        # read webcamera
        video_capture = cv2.VideoCapture(self.camera_path)
        if video_capture.isOpened() != True:
            video_capture.open(self.camera_path)

        while True:
            ret, frame = video_capture.read()
            # if ret != True:
            #     print(':---------------------------reset---------------------------------------')
            #     video_capture.release()
            #     video_capture = cv2.VideoCapture(self.camera_path)
            #     if video_capture.isOpened() != True:
            #         video_capture.open(self.camera_path)
            #
            #     continue
            if ret:
                #frame = np.rot90(frame, -1)
                frame = cv2.resize(frame, dsize=(VIDEO_HEIGHT, VIDEO_WEIGHT), interpolation=cv2.INTER_CUBIC)
                # cv2.imshow('frame', frame)
                # put frame into queue
                print('Receive inqueue')
                self.frame_queue.put(frame)
                time.sleep(0.001)
            else:
                print('read empty')
                video_capture.release()
                video_capture = cv2.VideoCapture(self.camera_path)
                if video_capture.isOpened() != True:
                    video_capture.open(self.camera_path)

    def post_event(self):
        while True:
            print('self.post_queue.empty:', self.post_queue.empty())

            if self.post_queue.empty()!= True:
                (frame,post_data) = self.post_queue.get()
                print('post event')
                # url_request = "http://www.kangni.com/resource/addEvent"
                # files = {'file': ('pic.jpg', io.BytesIO(cv2.imencode('.jpg', frame)[1].tostring()), 'image/jpeg')}
                # r = requests.post(url_request, files=files, data=post_data)

                # print(r.text)
            time.sleep(0.1)

    def push_frame(self):
        print('push frame')
        global ABNORMAL_DETECT_STATUS_DELTA

        # # 防止多线程时 command 未被设置
        # while True:
        #     if len(self.command) > 0:
        #
        #         print('set pipe info')
        #
        #         # 管道配置
        #         p = sp.Popen(self.command, stdin=sp.PIPE)
        #         break

        K.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        yolo = YOLO()

        gl.clear_detect_items()
        print('clear after:', gl.DETECT_TYPE)

        # for camera_event_type_item in self.detect_items:
        #     gl.set_detect_items(camera_event_type_item)
        #
        # for camera_event_type_id_item in self.detect_items_index:
        #     gl.set_note_event_id(camera_event_type_id_item)

        gl.set_detect_items('安全帽检测')
        gl.set_detect_items('安全带检测')
        print(gl.DETECT_TYPE)
        # print(gl.NOTE_EVENT_ID)

        # deep_sort
        model_filename = '/Users/apple/Desktop/Git/Track/deep_sort_yolov3_lastest/model_data/mars-small128.pb'
        # model_filename = 'model_data/mars-small128.pb'

        encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        # 非极大值抑制
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.3, None)
        tracker = Tracker(metric)

        cnt = 0
        befor_worker_status={}

        while True:
            try:
                if self.frame_queue.empty() != True:
                    #print('-----------------queue size: ', self.frame_queue.qsize())
                    if self.frame_queue.qsize() > 5:
                        frame = self.frame_queue.get()
                        # process fr
                        t3 = time.time()
                        #frame = cv2.resize(frame, dsize=(VIDEO_HEIGHT,VIDEO_WEIGHT), interpolation=cv2.INTER_CUBIC)
                        print('shape:',frame.shape)
                        #frame = np.rot90(frame, -2)
                        print('resize cost:',time.time()-t3)
                        if frame.shape[0] < frame.shape[1]:
                          frame = np.rot90(frame, -1)

                        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
                        t4=time.time()
                        print('image convert cost:',t4-t3)

                        # 人框，所有目标框
                        boxs, other_boxs = yolo.detect_image(image)
                        t5=time.time()

                        print(cnt,' box:',other_boxs,boxs,'detect cost:',t5-t4)
                        frame1=copy.deepcopy(frame)
                        features = encoder(frame, boxs)
                        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

                        # Run non-maxima suppression.
                        boxes = np.array([d.tlwh for d in detections])
                        scores = np.array([d.confidence for d in detections])
                        indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
                        detections = [detections[i] for i in indices]

                        tracker.predict()
                        tracker.update(detections)
                        t6=time.time()
                        print('track decode cost:',t6-t5)

                        update_flag = False
                        current_worker_status = {}
                        for track in tracker.tracks:
                            if not track.is_confirmed() or track.time_since_update > 1:
                                continue

                            if track.corresponding_detect == -1:
                                continue
                            b = detections[track.corresponding_detect].tlwh
                            bbox = [int(b[0]), int(b[1]), int(b[0] + b[2]), int(b[1] + b[3])]
                            bbox[0] = min(max(0,bbox[0]),frame.shape[1])
                            bbox[1] = min(max(0,bbox[1]),frame.shape[0])
                            bbox[2] = min(max(0,bbox[2]),frame.shape[1])
                            bbox[3] = min(max(0,bbox[3]),frame.shape[0])
                            # bbox = track.to_tlbr()
                            frame = frame.astype(np.uint8)
                            frame = frame.copy()
                            cv2.putText(frame, str(track.track_id),(int((bbox[0]+bbox[2])/2), int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0),2)

                            # 安全带/安全帽匹配
                            t7=time.time()
                            single_person_box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), track.track_id]
                            match_ok,current_wear_list = person_status_match(single_person_box, other_boxs, current_worker_status)

                            add_flag = False
                            note_violation_behavior(track.track_id,current_wear_list)
                            if track.track_id in ABNORMAL_DETECT_STATUS_DELTA.keys() and ABNORMAL_DETECT_STATUS_DELTA[track.track_id]['count']==EVNENT_NOTE_COUNT_THRESH:
                                add_flag, current_wear_list, violation_item, event_id_list = status_change_note(befor_worker_status, current_worker_status, track.track_id)
                                update_flag = True
                            print(ABNORMAL_DETECT_STATUS_DELTA)

                            if float(int(frame.shape[0]) - bbox[3]) / float(frame.shape[0]) >= CLIMB_FLAG:
                                # 未带安全帽
                                t10 = time.time()
                                if add_flag or 0 in current_wear_list:
                                    frame1 = frame1.copy()
                                    cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_red,2)

                                    cv2.putText(frame1, str(track.track_id),(int((bbox[0] + bbox[2]) / 2), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,color_green,2)
                                    wear_list = current_worker_status[track.track_id]

                                    for i in range(len(wear_list)):
                                        wear_box = other_boxs[wear_list[i][1]]
                                        cv2.rectangle(frame1, (int(wear_box[0]), int(wear_box[1])),(int(wear_box[2]), int(wear_box[3])), color_white, 2)

                                    if add_flag:
                                        post_data = {
                                            # "employId": str(track.track_id),
                                            "cameraId": str(0),
                                            "eventTypeId": str(0),
                                            "photo": "pic.jpg",
                                            "photoHeight": str(VIDEO_WEIGHT),
                                            "photoWeight": str(VIDEO_HEIGHT)
                                        }
                                        self.post_queue.put((frame, post_data))

                                        print('##添加事件   ',str(track.track_id)+'未佩戴：',violation_item)

                                else:
                                    frame1 = frame1.copy()
                                    cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_blue,2)
                                    cv2.putText(frame1, str(track.track_id),(int((bbox[0] + bbox[2]) / 2), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,color_green,2)

                                    wear_list = current_worker_status[track.track_id]
                                    for i in range(len(wear_list)):
                                        wear_box = other_boxs[wear_list[i][1]]
                                        cv2.rectangle(frame1, (int(wear_box[0]), int(wear_box[1])),(int(wear_box[2]), int(wear_box[3])), color_white, 2)

                                    t12 = time.time()
                                    print(track.track_id,' not add cost:',t12-t10)
                                    continue

                            else:
                                if DEBUG:
                                    frame1=frame1.copy()
                                    # color = colors[int(track.track_id) % len(colors)]
                                    # color = [i * 255 for i in color]

                                    cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_blue, 2)

                                    cv2.putText(frame1, str(track.track_id),
                                                (int((bbox[0] + bbox[2]) / 2), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green,
                                                2)

                                    wear_list=current_worker_status[track.track_id]
                                    for i in range(len(wear_list)):
                                        wear_box=other_boxs[wear_list[i][1]]
                                        cv2.rectangle(frame1, (int(wear_box[0]), int(wear_box[1])), (int(wear_box[2]), int(wear_box[3])), color_blue, 2)

                        if update_flag:
                            befor_worker_status = current_worker_status

                        t9=time.time()
                        print('track cost:',t9-t6)

                        frame1 = frame1.copy()
                        cv2.line(frame1,(0,int((1-CLIMB_FLAG)*frame1.shape[0])),(frame1.shape[1],int((1-CLIMB_FLAG)*frame1.shape[0])),(0,255,0),3)
                        image_tmp = Image.fromarray(frame1)
                        image_tmp.show(str(cnt))
                        print('fps:',1.0/(t9-t3))
                        print('')
                        print('')
                        print('')
                        # write to pipe
                        # p.stdin.write(frame1.tostring())
            except:
                pass
                # write to pipe
                # p.stdin.write(frame.tostring())

    def run(self):
        threads = [threading.Thread(target=Live.read_frame, args=(self,)), threading.Thread(target=Live.push_frame, args=(self,))]
        for thread in threads:
            thread.start()

def my_run(camera_id, camera_event_name_list, camera_event_type_id_list, camera_rtcp, gpu_mem_rate):

    live = Live(camera_id, camera_event_name_list, camera_event_type_id_list, camera_rtcp, gpu_mem_rate)
    threads = [threading.Thread(target=live.read_frame),threading.Thread(target=live.push_frame),threading.Thread(target=live.post_event)]
    # threads = [threading.Thread(target=live.read_frame),threading.Thread(target=live.push_frame)]
    # threads = [threading.Thread(target=live.post_event),threading.Thread(target=live.read_frame),threading.Thread(target=live.push_frame)]

    #[thread.setDaemon(True) for thread in threads]
    for thread in threads:
        thread.start()
    #for thread in threads:
    #    thread.join()

if __name__ == '__main__':
    my_run(0,0,0,"/Users/apple/Desktop/Git/Track/deep_sort_yolov3_lastest/model_data/VID_20190627_170344.mp4",1)

    # live = Live(0,0)
    # #live.run()
    #
    #
    # threads = [threading.Thread(target=live.read_frame),
    #            threading.Thread(target=live.push_frame)]
    #
    # #[thread.setDaemon(True) for thread in threads]
    # for thread in threads:
    #     thread.start()
    #
    







