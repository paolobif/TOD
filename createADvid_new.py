import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
import random
import numpy as np
import time
import fnmatch
#from skimage.morphology import skeletonize, medial_axis
#from matplotlib import pyplot as plt
#from re import sub




if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    SCOR_DIR = sys.argv[1]
    VID_DIR = sys.argv[2]
    OUT_PATH = sys.argv[3]
    MORPH_PATH = "D:/Adef/adef.csv"
    circl = pd.read_csv(MORPH_PATH,names=("expID","fBS","x","y","r")).replace(".avi", "", regex=True)    
    circl = circl.iloc[1: , :]
    print(circl)
    vid_list = fnmatch.filter(os.listdir(VID_DIR),"*.avi")
 
    for vid_name in vid_list:
        videoPath = VID_DIR+vid_name
        vid = cv2.VideoCapture(videoPath)
        csv_path = f"{SCOR_DIR}/{os.path.basename(vid_name).strip('.avi')}.csv"
        total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        out_video_path = f"{OUT_PATH}/{os.path.basename(vid_name).strip('.avi')}_paralysis.avi"
        if os.path.isfile(out_video_path) == False and os.path.isfile(csv_path)==True:
            df = pd.read_csv(csv_path,names=('age','x1','y1','x2','y2','expID','BS'))
            df = df.iloc[1: , :]
            df = np.asarray(df)
            expname = os.path.basename(vid_name).strip('.avi')
            filtval = circl['expID'] == expname
            interim = circl[filtval] 
            interim2 = np.asarray(interim)
            if interim2.size > 0:
                for roe in interim2:
                    expIDcirc,framecirc,center_x,center_y,radius,*_ = roe
                    center_x=int(center_x)
                    center_y=int(center_y)
                    radius2=int(radius)
            print(center_x,center_y,radius2)        
            while (1):
                ret, frame = vid.read()
                frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
                
                if frame_count == 1:
                    height, width, channels = frame.shape
                    #print(height, width)
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(out_video_path, fourcc, 10, (width, height), True)
                    
                
                    #print("drawing circle")
                cv2.circle(frame, (center_x,center_y), radius2, (255,0,255), 2)
                   
                for rows in df:
                    frameNA, x1, y1, x2, y2,expID,bs, *_ = rows  
                    frameNA = int(float(frameNA))
                    if frame_count > frameNA:
                        x1=int(float(x1))
                        x2=int(float(x2))
                        y1=int(float(y1))
                        y2=int(float(y2)) 
                        if (abs(x1-center_x)^2)+(abs(y1-center_y)^2) < (radius2^2) and (abs(x2-center_x)^2)+(abs(y2-center_y)^2) < (radius2^2):
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
                            #cv2.line(frame, (center_x,center_y), (x1,y1), (255,0,0), 2)
                        else:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                            #cv2.line(frame, (center_x,center_y), (x1,y1), (0,0,255), 2)
                            
                writer.write(frame)
                if frame_count == total_frame_count:
                    break
            writer.release() 
            print(out_video_path)

        
    #if (abs(x1-center_x)^2)+(abs(y1-center_y)^2) < (radius2^2)
        