from itertools import count
import cv2
from nbformat import read
import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def non_max_suppression_post(outputs: np.ndarray, overlapThresh, counts=False):
    # if there are no boxes, return an empty list
    boxes = outputs.astype(float)
    # for out in outputs:
    #     x1, y1, x2, y2, = out
    #     fullOutputs.append([x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist(),
    #                         conf.tolist(), cls_conf.tolist()])
    # t = time.time()
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x2 = x1 + w
    y2 = y1 + h
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    cs = []
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

        # get the counts for each bounding box.
        cs.append(len(np.where(overlap > overlapThresh)[0]))

    if counts:
        return boxes[pick].astype(float), cs

    else:
        return boxes[pick].astype(float)


def csv_reader(csv_path: str):
    """Loads CSV from csv path"""
    df = pd.read_csv(csv_path,
                     usecols=[0, 1, 2, 3, 4, 5],
                     names=["frame", "x", "y", "w", "h", "class"])
    return df


def worms_from_frame_range(df, lower: int, upper: int):
    """Given CSV_Reader will retrieve all the bounding boxes detected
    within a paticular lower to upper frame range

    Args:
        reader (CSV_Reader): CSV_Reader object.
        lower (int): lower frame bound.
        upper (int): upper frame bound.
    """
    all_bbs = df[df["frame"].between(lower, upper)]
    all_bbs = all_bbs.to_numpy()[:, 1:5]  # Get only important columns.
    return all_bbs


def process_experiment(df, interval: int, exp_id: int):
    """Itterates through frames in the experiment
    and runs NMS on the detections over the spec. interval.

    Args:
        df (_type_): df with yolo data.
        interval (int): how far to check for running average.
        exp_id (int): exp id for saving the csv.

    """
    skip = 5 # Skip so that the running total isn't done every single frame...

    frame_count = int(df["frame"].max())
    print(frame_count)

    frame_count = 2480
    counts = []
    for i in tqdm(range(frame_count, 0, -skip)):
        bbs = worms_from_frame_range(df, i, i + interval)
        nms = non_max_suppression_post(bbs, 0.8)
        counts.append([exp_id, i, len(nms)])
    return counts



if __name__ == "__main__":
    INTERVAL = 100
    SAVE_PATH = "/mnt/sdb1/videos/4_data/results/tod.csv"

    csv_path = "/mnt/sdb1/videos/4_data/csvs"
    # meta_path = "exp/meta/ExpMatch.csv"  # Not in use rn.

    csv_names = os.listdir(csv_path)

    print(f"""
           CSVs from: {csv_path}
           SAVE path: {SAVE_PATH}
           Interval: {INTERVAL}
           Processing {len(csv_names)} experiments!
           """)

    # dfs = []
    for i, csv in enumerate(csv_names):
        exp_id, ext = csv.split(".")
        if ext != "csv":
            continue
        # Start processing.
        print(f"--- Processing {exp_id} ---")
        full_path = os.path.join(csv_path, csv)
        df = csv_reader(full_path)
        counts = process_experiment(df, INTERVAL, exp_id=exp_id)
        count_df = pd.DataFrame(counts, columns=["exp_id", "frame", "count"])
        count_df.to_csv(SAVE_PATH, index=None, mode='a', header=None)
        print(f"{i} / {len(csv_names)}")
        # dfs.append(count_df)

    # all_df = pd.concat(dfs, ignore_index=True)
    # all_df.to_csv(SAVE_PATH, index=None)
