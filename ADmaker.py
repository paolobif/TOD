import os
import sys
import cv2
import csv
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
import time


def read_df(path):
    tod_calls = pd.read_csv(path, usecols=[0, 1, 2, 3, 4, 4, 5])
    return tod_calls


def save_video(path: str, vid_path: str, save_path: str):
    """Makes video from scored csv file.

    Args:
        path (str): death csv path.
        vid_path (str): raw video path.
        save_path (str): video save name.
    """
    vid = cv2.VideoCapture(vid_path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = vid.read()
    print(total_frames)
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(save_path, fourcc, 10, (width, height), True)

    df = read_df(path)
    df[df["frame"] == 'False'] = 0
    df = df.astype(int)

    for _ in tqdm(range(total_frames)):
        ret, frame = vid.read()
        frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
        if not ret:
            continue

        worms_to_draw = df[df["frame"] <= frame_count]
        worms_to_draw = worms_to_draw.to_numpy()
        for worm in worms_to_draw:
            x1, y1, x2, y2 = worm[1:5]  # Worm bounding boxes.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        writer.write(frame)

    writer.release()
    print("Video Done!")


if __name__ == "__main__":
    SCORED = "./exp/results/"
    VIDEO = "./exp/vids/"
    SAVE = "./exp/results/"

    # df = read_df(SCORED)

    for i in range(1045, 1049):
        scored = os.path.join(SCORED, f"{i}_auto.csv")
        video = os.path.join(VIDEO, f"{i}.avi")
        save = os.path.join(SAVE, f"{i}.avi")

        save_video(scored, video, save)

