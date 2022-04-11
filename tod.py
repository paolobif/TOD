import cv2
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

from utils import *

from perceptron_analysis import *


# Loads csv
class CSV_Reader():
    frame_interval = 100   # Frame intervals to check for stagnant worms
    step = 1  # Frame step within each interval
    prevalence = 0.3  # Fraction of unchanged bbs needed to be considered "stagnant"
    padding = 100  # How many frames to pad by to be safe

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

        self.exp_end = self.determine_exp_end() + self.padding

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

    def determine_exp_end(self, nms=0.6):
        # Get threshold for count needed to be considered "stagnant"
        interval = self.frame_interval
        thresh = int((interval / self.step) * self.prevalence)
        # Step is 1 for now due to df.between() further down.
        print("Determining Experiment End")

        stagnant = []
        # Loop through the frame intervals
        for i in tqdm(range(0, self.frame_count, interval)):
            bbs = self.df[self.df["frame"].between(i, i + interval)]
            bbs = bbs.to_numpy()
            bbs = bbs[:, 1:5]

            # Yolo was not run on entire video.
            # if len(bbs) == 0:
            #     print(f"No bbs found in interval {i} - {i + interval}")
            #     stagnant.append(0)
            #     continue

            final_bbs, counts = non_max_suppression_post(bbs, nms, counts=True)
            counts = np.array(counts)
            # Determine what worms are stagnant.
            stagnant_worms = counts[counts > thresh]
            stagnant.append(len(stagnant_worms))

        end_idx = np.argmax(stagnant)
        exp_end = (end_idx + 1) * interval

        # Prevent too large exp ends
        # if exp_end > 2400:
        #     exp_end = 2400

        print(f"Experiment Done @ Frame {exp_end}")
        return exp_end


class WormViewer(CSV_Reader):
    """Uses the fixed location of worms and
    locates them over the series of frames saving information
    from the videos"""

    nms = 0.3  # NMS threshold.
    count = 20  # How many frames used to locate fixed bbs.
    scan = 2000  # Numer of frames in reverse to examine.

    def __init__(self, csv: str, vid: str, thresh: int = 35, first=False, train_path="weights/weights_test.csv"):
        super().__init__(csv, vid)
        # Make sure doesn't exceed video frame capcacity.
        if self.exp_end + self.count > self.frame_count:
            self.exp_end = self.frame_count - self.count

        self.first = first if first else self.exp_end
        # Make sure first is not the very end of the experiment,
        if self.first > self.frame_count - 21:
            self.first = self.frame_count - 21

        print(f"Processing in reverse from {self.first}")

        # Get tracked bbs of interest.
        self.tracked, _ = self.get_worms_from_end(self.first, self.count, self.nms)

        self.thresh = thresh

        worm_ids = np.arange(0, len(self.tracked))
        worm_state: dict[int, int] = {}
        for i in worm_ids:
            worm_state[i] = False
        self.worm_state = worm_state
        self.perceptron = BinaryPerceptron(6, [0, 0, 0, 0, 0, 0], alpha=0.01, save_path=train_path)
        self.perceptron.load()

    def fetch_worms(self, worm_ids: list, frame_id: int, pad=0, offset=(0,0), auto=False):
        """Fetches worms in self.tracked by worm id on a given frame_id.
        Allows for padding and auto padding for worms that are skinny.

        Args:
            worm_ids (list): List of worm ids to be fetched.
            frame_id (int): Frame from which to fetch worms.
            pad (int, tuple): Padding in x and y direction. Tuple or Int.
            offset (tuple): Offset in x and y direction. Tuple.
            auto (tuple, optional): Auto pads for skinny worms.

        Returns:
            _type_: _description_
        """
        # Get pad ammount.
        if type(pad) == int:
            padX = pad
            padY = pad
        else:
            padX, padY = pad

        # Get the bbs for the frame.
        ret, frame = self.get_frame(frame_id)
        height, width = frame.shape[:2]

        if not ret:
            print(f"Frame {frame_id} not found.")
            pass
        # Get worm image for each frame
        worm_imgs = []
        for worm in worm_ids:
            if worm >= len(self.tracked):
                pass

            x, y, w, h = self.tracked[worm].astype(int)
            x += offset[0]
            y += offset[1]
            x, y, w, h = x - padX, y - padY, w + 2*padX, h + 2*padY

            if auto:
                x, y, w, h = auto_pad(x, y, w, h)

            # Set x y lower and upper to account for padding / offset.
            y_l, y_u = max(0, y), min(height, y + h)
            x_l, x_u = max(0, x), min(width, x + w)

            worm_img = frame[y_l:y_u, x_l:x_u]
            worm_imgs.append(worm_img)

        return worm_imgs

    def transform_all_worms(self, worms):
        new_worms = []
        for worm in worms:
            new_worm = self.image_transformation(worm)
            new_worms.append(new_worm)

        return new_worms

    def compute_score(self, skip=20, count=5, gap=50):
        """Goes in reverse analyzing the worm locations to determine
        time of death.

        Args:
            skip (int, optional): How many frames to skip.
            count (int, optional): How many frames to get average from.
                                   Defaults to 15.
            gap (int, optional): How many frames to jump before starting to
                                 skip for frame averages. Defaults to 5.
        """
        stop = max(self.first - self.scan, 1)  # where to stop checking in reverse.
        start = self.first  # where to start checking in reverse.
        assert(start > stop), "Invalid scan and first params."

        # Init object.
        worm_ids = np.arange(0, len(self.tracked))
        difs: dict[int, list] = {}
        for i in worm_ids:
            difs[i] = []

        # Loop through every {skip} frames in reverse.
        for i in tqdm(range(start, stop, -skip)):
            # Update the tracked between skipped intervals.
            update_interval = 5  # How many frames to update
            for j in range(skip, 0, -update_interval):
                self.update_tracked(i + j, thresh=0.8, count=5)
                # Updates the location every 5 frames.

            # Begin score computation
            current_worms = self.fetch_worms(worm_ids, i)
            current_worms = self.transform_all_worms(current_worms)

            # Sets frame range for getting worm averages.
            high = min(start, i + gap + skip * count)  # Upper bounds.
            low = min(start, i + gap)  # Lower bounds.

            # Lopp through frames of interest.
            worms_to_inspect = []
            for n in range(low, high, skip):
                # wti = worm to inspect
                wti = self.fetch_worms(worm_ids, n)
                wti = self.transform_all_worms(wti)
                worms_to_inspect.append(wti)

            worms_to_inspect = np.array(worms_to_inspect, dtype=object)

            # Ignore beginning where there are no worms to compare.
            if worms_to_inspect.shape == (0,):
                print("skipping empty")
                continue

            for worm_id in worm_ids:
                older_worms = worms_to_inspect[:, worm_id]
                # older_worms = self.transform_all_worms(older_worms)
                # self.older = worms_to_inspect
                current_worm = current_worms[worm_id]
                xshape, yshape = current_worm.shape

                # totals = []
                totals2 = []
                for worm in older_worms:
                    # difference = self.calculate_difference(worm, current_worm)
                    # totals.append(difference.sum(axis=None))

                    cur_data = TrainingData(current_worm, worm)
                    totals2.append(self.perceptron.classify(cur_data.getVector()))

                # pixel_count = xshape * yshape
                # avg = np.average(totals)
                # avg = avg / pixel_count  # Normalize difference by pixel count.

                # difs[worm_id].append(avg)

                total2_avg = np.nanmean(totals2)

                # if not self.worm_state[worm_id] and avg > self.thresh:
                if not self.worm_state[worm_id] and total2_avg > 0:

                    self.worm_state[worm_id] = i + skip
                    # included - gap to account for the fact that when the worm
                    # has moved it is already alive, so go back to last time it
                    # was known to be dead.
        return difs

    def update_tracked(self, frame_id: int, thresh=0.8, count: int = 5):
        """Matches the current list of bounding boxes to the list of worms
        found at frame number frame_id. Then updates the list of tracked worms
        only if the iou is greater than the threshold value.

        Args:
            frame_id (int): frame you want to fetch the future wormms from.
            thresh (float, optional): nms and iou thresh for worms. Defaults to 0.8.
            count (int, optional): How many frames to average from. Defaults to 5.
        """
        upper_bound = frame_id + count
        all_futures = self.df[self.df["frame"].between(frame_id, upper_bound)]
        all_futures = all_futures.to_numpy()[:, 1:5]
        futures = non_max_suppression_post(all_futures, thresh)  # Possible Boxes

        tracked = self.tracked
        updated = []

        for i, track in enumerate(tracked):
            # Make sure there are bounding boxes on the next frame.
            if len(futures) == 0:
                continue
            track = np.tile(track, (len(futures), 1))
            track2 = xywh_to_xyxy(track)

            futures2 = xywh_to_xyxy(futures)

            # Find overlaped box.
            xx1 = np.maximum(track2[0], futures2[0])
            yy1 = np.maximum(track2[1], futures2[1])
            xx2 = np.minimum(track2[2], futures2[2])
            yy2 = np.minimum(track2[3], futures2[3])

            # Compute area for each section
            area_track = track[:, 2] * track[:, 3]
            area_futures = futures[:, 2] * futures[:, 3]

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            area = w * h
            # Calculate intersection / union
            iou = area / (area_track + area_futures - area)
            assert(iou.all() >= 0)
            assert(iou.all() <= 1)

            match_idx = np.argmax(iou)
            iou_val = iou[match_idx]

            if iou_val >= thresh:
                # print(f"updated {i} by {futures[match_idx] - tracked[i]}")
                # Keep w and height the same.
                x1, y1, nw, nh = futures[match_idx]
                tracked[i] = [x1, y1, tracked[i][2], tracked[i][3]]
                updated.append(i)

        self.tracked = tracked
        return updated

    def save_scored_data(self, exp_id, path="./"):
        rows = []
        for i, bb in enumerate(self.tracked):
            x1, y1, w, h = bb
            x2 = x1 + w
            y2 = y1 + h
            death = self.worm_state[i]
            row = [death, x1, y1, x2, y2, exp_id]
            rows.append(row)

        df = pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "expID"])
        save_name = os.path.join(path, f"{exp_id}_auto.csv")
        df.to_csv(save_name, index=None)

    def create_worm_video(self, worm_id: int, duration: int, pad: int = 3):
        """Goes from first frame in reverse for 'duration' number of frames.
        creates a video for location chosen by the worm_id

        Args:
            worm_id (int): worm id to track in video
            duration (int): how many frames to cover in the video
            pad (int): padding on the image around the bounding box
        """
        x, y, w, h = self.tracked[worm_id].astype(int)
        # x, y, w, h = x - pad, y - pad, w + pad, h + pad  # Apply pad to image.
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
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
        new_img = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 22)
        return new_img

    @staticmethod
    def calculate_difference(wormA, wormB):
        wormA = cv2.blur(wormA, (2, 2))
        wormB = cv2.blur(wormB, (2, 2))
        diff = cv2.absdiff(wormA, wormB)
        return diff


# Functions for processing the videos

def match_csv_video(csvs, videos):
    csvs = [os.path.splitext(csv)[0] for csv in csvs]
    videos = [os.path.splitext(video)[0] for video in videos]

    matches = []
    for csv in csvs:
        if csv in videos:
            matches.append(csv)
        else:
            pass

    return matches


def batch_process(csv_dir: str, video_dir: str, save_dir: str = "./", first=False):
    """Takes directory of csvs with yolo outputs and then directory
    with videos. Using the video and bounding boxes, creates and saves
    time of death csv for each experiment.

    Args:
        csv_dir (str): folder with only csvs
        video_dir (str): folder with only videos
        save_dir (str): folder where to save videos
        first (int): frame from where to start
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    csvs = os.listdir(csv_dir)
    videos = os.listdir(video_dir)

    exp_ids = match_csv_video(csvs, videos)

    for exp_id in exp_ids:
        print(f"Processing {exp_id}")
        csv_path = os.path.join(csv_dir, f"{exp_id}.csv")
        vid_path = os.path.join(video_dir, f"{exp_id}.avi")

        viewer = WormViewer(csv_path, vid_path, first=first, thresh=30)

        # Thresh is the score in frame difference to call death.
        viewer.compute_score()
        viewer.save_scored_data(exp_id, path=save_dir)

        print(f"Done processing {exp_id}")


if __name__ == "__main__":
    CSVS = "/mnt/sdb1/videos/resveratrol_data/csvs"
    VIDS = "/mnt/sdb1/videos/resveratrol_data/vids"
    SAVE = "/mnt/sdb1/videos/resveratrol_data/results"

    # CSVS is path to directory with all the YOLO output files
    # VIDS is the path to a directory with the matching raw videos
    # SAVVE is the directory where the video outputs are saved.


    # Make sure the names for the csvs match the names for the videos.

    batch_process(CSVS, VIDS, SAVE)


    # IGNORE  ------------------- |
    #                             V

    # csv_path = "./data/1046.csv"
    # vid_path = "./data/1046.avi"

    # processer = CSV_Reader(csv_path, vid_path)
    # frame, bbs = processer.get_worms_from_frame(206)

    # # print(bbs)
    # a = processer.get_worms_from_end(2400, 20)
    # b = non_max_suppression_post(a, 0.3)
    # print(a.shape, b.shape)
