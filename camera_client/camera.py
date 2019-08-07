import argparse
import json
import os
import subprocess
import time
from urllib.parse import urljoin

import cv2
import osascript
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--endpoint", default="http://localhost:5000")
parser.add_argument("--pose-config", default="pose_action.json")
parser.add_argument("--save-video", default=None)
args = parser.parse_args()


relationPose = {
    "hands-on-head": 1,
    "victory": 2,
    "cheer-up": 3,
    "go-next": 4,
    "go-back": 5,
    "ultraman": 6,
}


def draw_body_parts(image, parts):
    connections = [
        ["neck", "lshoulder"],
        ["neck", "rshoulder"],
        ["lshoulder", "lelbow"],
        ["rshoulder", "relbow"],
        ["relbow", "rwrist"],
        ["lelbow", "lwrist"],
        ["neck", "nose"],
        # ["nose", "reye"],
        # ["nose", "leye"],
        # ["leye", "lear"],
        # ["reye", "rear"],
        ["neck", "rhip"],
        ["neck", "lhip"],
    ]
    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        # [0, 255, 85],
        # [0, 255, 170],
        # [0, 255, 255],
        # [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        # [85, 0, 255],
        # [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]
    score_threshold = 0.25

    height, width, _ = image.shape

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


def draw_face_bbox(image, bbox):
    if bbox is None:
        return image

    height, width, _ = image.shape
    w, h = bbox["w"], bbox["h"]
    tx, ty = bbox["x"] - w // 2, bbox["y"] - h // 2

    color = [0, 255, 0]
    cv2.line(image, (tx, ty), (tx + w, ty), color, 3)
    cv2.line(image, (tx + w, ty), (tx + w, ty + h), color, 3)
    cv2.line(image, (tx + w, ty + h), (tx, ty + h), color, 3)
    cv2.line(image, (tx, ty + h), (tx, ty), color, 3)

    return image


def post_image_to_server(image):
    # convert image data to jpeg binaries
    _, jpegbytes = cv2.imencode(".jpg", image)
    reqbody = jpegbytes.tobytes()

    # send the request
    headers = {"content-type": "application/octet-stream"}
    url = urljoin(args.endpoint, "infer")
    res = requests.post(url, data=reqbody, headers=headers)

    return res.json()


def operate_powerpoint(action):
    if action == "move-next":
        osascript.run(
            """
            tell application "Microsoft PowerPoint"
                activate
                tell application "System Events"
                    keystroke (ASCII character 29)
                end tell
            end tell
            """
        )
    elif action == "move-prev":
        osascript.run(
            """
            tell application "Microsoft PowerPoint"
                activate
                tell application "System Events"
                    keystroke (ASCII character 28)
                end tell
            end tell
            """
        )


def play_music(filename):
    filename = "./sounds/" + filename
    cmd = ["afplay", filename]
    subprocess.call(cmd, shell=False)


def find(iterable, default=False, pred=None):
    return next(filter(pred, iterable), default)


def execute_action(_type, action):
    if _type == "slide":
        operate_powerpoint(action)
    elif _type == "sound":
        filename = action + ".mp3"
        play_music(filename)


def load_config():
    # load pose to action relation config
    f = open(args.pose_config)
    pose_action = json.load(f)["data"]
    return pose_action


def main():
    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    while cap.isOpened():
        start_time = time.time()  # to calculate fps

        # read camera image
        _, image = cap.read()

        # perform pose classification with half size image
        height, width, _ = image.shape
        image = cv2.resize(image, (width // 2, height // 2))
        result = post_image_to_server(image)

        # when no human detected
        if "pose_class" not in result:
            continue
        pose = result["pose_class"]

        # draw detected body parts
        parts = result["parts"]
        image = draw_body_parts(image, parts)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(
            image,
            f"class: {pose} fps: {fps:.2f}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )
        cv2.imshow("pose result", image)

        if "missing_body_part" not in pose and pose != "others":
            # execute action corresponding detected pose
            pose_action = load_config()
            pose_id = relationPose[pose]
            relation = find(pose_action, None, lambda x: x["poseId"] == pose_id)
            execute_action(relation["actionType"], relation["name"])

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
