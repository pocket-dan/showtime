import json
import os
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import requests
from tqdm import tqdm

import utils

RAW_DIR = Path("dataset/raw")
PROCESSED_DIR = Path("dataset/processed")
DATASETS = ["k", "m", "n2", "o", "t", "others"]


def extract_frames():
    """
    Convert .mov to .jpg files
    """

    OUTPUT_FORMAT = "frame_%03d.jpg"

    for d in DATASETS:
        for m in (RAW_DIR / d).glob("*.mov"):
            _input = RAW_DIR / d / m.name

            _m = str(m.name).replace(m.suffix, "")
            (PROCESSED_DIR / d / _m).mkdir(parents=True, exist_ok=True)
            _output = PROCESSED_DIR / d / _m / OUTPUT_FORMAT
            print(_input, _output)

            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(_input),
                    "-r",
                    "10",
                    "-vcodec",
                    "mjpeg",
                    str(_output),
                ]
            )


def count_samples():
    """
    Count num of files for each directory
    """
    dirs = [PROCESSED_DIR / d for d in DATASETS]
    for d in dirs:
        print(d.name)
        for _d in utils.subdirs(d):
            n_samples = len(list(_d.glob("frame_*.jpg")))
            print(f"  {_d.name:<20}:{n_samples}")


def annotate_pose():
    """
    get body keypoints using openpose
    """
    endpoint = os.environ.get("SERVER_ENDPOINT")

    # check server status
    req = urllib.request.Request(endpoint)

    with urllib.request.urlopen(req) as res:
        resbody = res.read()
        result = json.loads(resbody)
        print(f"{endpoint}: '{result}'")

    # get pose annotation using ml server
    url = urllib.parse.urljoin(endpoint, "infer")

    videos = [v for d in DATASETS for v in utils.subdirs(PROCESSED_DIR / d)]
    for video in tqdm(videos):
        for imgpath in tqdm(video.glob("*.jpg")):
            f = open(imgpath, "rb")
            reqbody = f.read()
            f.close()

            # send request
            req = urllib.request.Request(
                url,
                reqbody,
                method="POST",
                headers={"Content-Type": "application/octet-stream"},
            )
            try:
                with urllib.request.urlopen(req) as res:
                    resbody = res.read().decode("utf-8")

                    # save to json file
                    f = open(imgpath.with_suffix(".json"), "w")
                    f.write(resbody)
                    f.close()
            except urllib.error.HTTPError as err:
                print("failed to get annotation:", str(imgpath))
                print(err.code)
            except urllib.error.URLError as err:
                print(err.reason)


def draw_pose_annotation():
    """
    Visualize pose annotation
    """
    images = [
        i
        for d in DATASETS
        for v in utils.subdirs(PROCESSED_DIR / d)
        for i in v.glob("*.jpg")
    ]
    images = sorted(images, key=lambda x: str(x))

    score_threshold = 0.25

    connections = [
        ["neck", "lshoulder"],
        ["neck", "rshoulder"],
        ["lshoulder", "lelbow"],
        ["rshoulder", "relbow"],
        ["relbow", "rwrist"],
        ["lelbow", "lwrist"],
    ]
    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        # [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        # [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        # [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]
    for path in tqdm(images):
        # read image
        img = cv2.imread(str(path))
        height, width, channel = img.shape

        # read annotation
        anno_file = path.with_suffix(".json")
        if not anno_file.exists():
            continue
        f = open(anno_file)
        annotation = json.load(f)
        parts = annotation["parts"]
        f.close()

        # draw poses
        for pair, color in zip(connections, colors):
            if pair[0] not in parts or pair[1] not in parts:
                continue
            part1, part2 = parts[pair[0]], parts[pair[1]]
            if part1["score"] <= score_threshold or part2["score"] <= score_threshold:
                continue

            p1 = (int(width * part1["x"]), int(height * part1["y"]))
            p2 = (int(width * part2["x"]), int(height * part2["y"]))
            cv2.line(img, p1, p2, color, 3)

        # save image
        out = str(path).replace("frame", "anno")
        cv2.imwrite(out, img)


def remove_missing_data():
    images = [
        i
        for d in DATASETS
        for v in utils.subdirs(PROCESSED_DIR / d)
        for i in v.glob("frame_*.jpg")
    ]
    images = sorted(images, key=lambda x: str(x))

    parts = ["lwrist", "lelbow", "lshoulder", "neck", "rshoulder", "relbow", "rwrist"]

    for img_path in tqdm(images):
        anno_path = img_path.with_suffix(".json")

        # if the annotation file doesn't exist, remove the image file
        if not anno_path.exists():
            os.remove(img_path)
            continue

        f = open(anno_path)
        anno = json.load(f)
        f.close()

        for part in parts:
            if not part in anno["parts"]:
                os.remove(img_path)
                os.remove(anno_path)
                break


if __name__ == "__main__":
    # print("extract frames...")
    # extract_frames()
    # print("annotate pose...")
    # annotate_pose()
    # print("draw pose annotation...")
    # draw_pose_annotation()
    print("remove missing data...")
    remove_missing_data()
    print("count samples...")
    count_samples()
