from itertools import count
from typing import List
import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import *


# Loads csv
class CSV_Reader():

    def __init__(self, csv, vid):
        """ Reads the csv and video and provides useful functions for determining
        time of death"""
        self.csv = csv
        self.vid = vid

        video = cv2.VideoCapture(vid)
        self.video = video
        self.frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.df = pd.read_csv(csv,
                              usecols=[0, 1, 2, 3, 4, 5],
                              names=["frame", "x", "y", "w", "h", "class"])

    def get_frame(self, frame_id):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = self.video.read()
        return ret, frame

    def get_worms_from_frame(self, frame_id):
        """ Gets the frame image from a frame id,
        and then the bounding boxes associated with that image"""
        ret, frame = self.get_frame(frame_id)
        bbs = self.df[self.df["frame"] == frame_id]
        bbs = bbs.to_numpy()
        bbs = bbs[:, 1:5]
        return frame, bbs

    def get_worms_from_end(self, first: int, count: int, nms: float = 0.3):
        """Cycles through framse in forward from first to last, and fetches the
        bounding boxes.

        Args:
            first ([int]): [latest frame you go from]
            count ([int]): [how many frames forward to track]
            nms ([float]): [thresh of overlap for nms]

        Returns:
            list of tracked bounding boxes and all the bounding boxes
            in the selected frames.
        """
        last = first + count
        if (last > self.frame_count - 1):
            print("Please pick an earlier frame")
            last = self.frame_count - 1

        all_bbs = np.empty((0, 4), int)
        for i in range(first, last):
            _, bbs = self.get_worms_from_frame(i)
            all_bbs = np.append(all_bbs, bbs, axis=0)

        tracked = non_max_suppression_post(all_bbs, nms)
        self.tracked = tracked
        return tracked, all_bbs


class WormViewer(CSV_Reader):
    """Uses the fixed location of worms and
    locates them over the series of frames saving information
    from the videos"""

    nms = 0.3  # NMS threshold.
    count = 20  # How many frames used to locate fixed bbs.
    scan = 1500  # Numer of frames in reverse to examine.

    def __init__(self, csv: str, vid: str, first: int=2400):
        super().__init__(csv, vid)
        # Get tracked bbs of interest.
        self.tracked, _ = self.get_worms_from_end(first, self.count, self.nms)
        self.first = first

    def fetch_worms(self, worm_ids: list, frame_id: int):
        ret, frame = self.get_frame(frame_id)
        if not ret:
            print(f"Frame {frame_id} not found.")
            pass
        # Get worm image for each frame
        worm_imgs = []
        for worm in worm_ids:
            if worm >= len(self.tracked):
                pass

            x, y, w, h = self.tracked[worm].astype(int)
            worm_img = frame[y:y+h, x:x+w]
            worm_imgs.append(worm_img)

        return worm_imgs

    def transform_all_worms(self, worms):
        new_worms = []
        for worm in worms:
            new_worm = self.image_transformation(worm)
            new_worms.append(new_worm)

        return new_worms

    def compute_score(self, count=15, gap=5):
        """Goes in reverse analyzing the worm locations to determine
        time of death.

        Args:
            count (int, optional): How many frames to get average from. Defaults to 15.
            gap (int, optional): How many frames to skip before getting frame averages. Defaults to 5.
        """
        stop = self.first - self.scan  # where to stop checking in reverse.
        start = self.first  # where to start checking in reverse.
        assert(start > stop), "Invalid scan and first params."

        # Init object.
        worm_ids = np.arange(0, len(self.tracked))
        difs: dict[int, list] = {}
        for i in worm_ids:
            difs[i] = []

        # Loop through every frame in reverse.
        for i in range(start, stop, -1):
            current_worms = self.fetch_worms(worm_ids, i)
            current_worms = self.transform_all_worms(current_worms)

            spread = count + gap
            # Sets frame range for getting worm averages.
            high = min(start, i + spread)  # Upper bounds.
            low = min(start, i + gap)  # Lower bounds.

            # Lopp through frames of interest.
            worms_to_inspect = []
            for n in range(low, high):
                # wti = worm to inspect
                wti = self.fetch_worms(worm_ids, n)
                wti = self.transform_all_worms(wti)
                worms_to_inspect.append(wti)

            worms_to_inspect = np.array(worms_to_inspect, dtype=object)

            for worm_id in worm_ids:
                # iw = inspected worms
                totals = []
                older_worms = worms_to_inspect[:, worm_id]
                current_worm = current_worms[worm_id]
                xshape, yshape = current_worm.shape

                for worm in older_worms:
                    difference = self.calculate_difference(worm, current_worm)
                    totals.append(difference.sum(axis=None))
            # print(worms_to_inspect.shape)



    def create_worm_video(self, worm_id: int, duration: int):
        """Goes from first frame in reverse for 'duration' number of frames.
        creates a video for location chosen by the worm_id

        Args:
            worm_id (int): worm id to track in video
            duration (int): how many frames to cover in the video
        """
        x, y, w, h = self.tracked[worm_id].astype(int)
        save_name = f"./results/{worm_id}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(save_name, fourcc, 3, (w, h), True)

        last = self.first - duration
        for i in range(self.first, last, -1):
            img = self.fetch_worms([worm_id], i)[0]
            writer.write(img)

        writer.release()

    @staticmethod
    def image_transformation(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        new_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,22)
        return new_img

    @staticmethod
    def calculate_difference(wormA, wormB):
        wormA = cv2.blur(wormA, (2,2))
        wormB = cv2.blur(wormB, (2,2))
        diff = cv2.absdiff(wormA, wormB)
        return diff


if __name__ == "__main__":
    csv_path = "./data/1046.csv"
    vid_path = "./data/1046.avi"

    processer = CSV_Reader(csv_path, vid_path)
    frame, bbs = processer.get_worms_from_frame(206)

    # print(bbs)
    a = processer.get_worms_from_end(2400, 20)
    b = non_max_suppression_post(a, 0.3)
    print(a.shape, b.shape)
