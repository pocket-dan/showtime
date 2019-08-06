import argparse
import sys
import time
from typing import Dict
from urllib.parse import urljoin

import cv2
import numpy as np
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--endpoint", default="http://localhost:5000")
args = parser.parse_args()


def draw_body_parts(image: np.ndarray, parts: Dict) -> np.ndarray:
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
    score_threshold = 0.25

    height, width, channel = image.shape

    for pair, color in zip(connections, colors):
        if pair[0] not in parts or pair[1] not in parts:
            continue
        part1, part2 = parts[pair[0]], parts[pair[1]]
        if part1["score"] <= score_threshold or part2["score"] <= score_threshold:
            continue

        p1 = (int(width * part1["x"]), int(height * part1["y"]))
        p2 = (int(width * part2["x"]), int(height * part2["y"]))
        cv2.line(image, p1, p2, color, 3)

    return image


def post_image_to_server(image: np.ndarray) -> Dict:
    # convert image data to jpeg binaries
    _, jpegbytes = cv2.imencode(".jpg", image)
    reqbody = jpegbytes.tobytes()

    # send the request
    headers = {"content-type": "application/octet-stream"}
    url = urljoin(args.endpoint, "infer")
    res = requests.post(url, data=reqbody, headers=headers)

    return res.json()


def main():
    cap = cv2.VideoCapture(args.camera)

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    start_time = time.time()
    while cap.isOpened():
        # read camera image
        _, image = cap.read()

        # get pose result from ml server
        try:
            result = post_image_to_server(image)
        except:
            continue

        # draw body parts to the image
        image = draw_body_parts(image, result["parts"])

        pose = result["pose_class"]
        if "missing" in result["pose_class"]:
            pose = "None"

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(
            image,
            f"fps: {fps:.2f} class: {pose}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3,
        )
        cv2.imshow("pose estimation demo", image)

        start_time = time.time()
        if cv2.waitKey(1) == 27:
            # stop with pressing ESC
            break


if __name__ == "__main__":
    main()
