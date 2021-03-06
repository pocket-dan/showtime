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
parser.add_argument("--output", default=None)
args = parser.parse_args()


relationPose = {
    "hands-on-head": 1,
    "victory": 2,
    "cheer-up": 3,
    "go-next": 4,
    "go-back": 5,
    "ultraman": 6,
}

action_interval = 0.8  # sec
recognition_count = 2


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
                    keystroke (ASCII character 28)
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
                    keystroke (ASCII character 29)
                end tell
            end tell
            """
        )


def play_music(filename):
    filename = "./sounds/" + filename
    cmd = ["afplay", filename]
    # non-blocking supbrocesss call
    subprocess.Popen(cmd)


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

    # record video if args.output is not None
    if args.output is not None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
        fps = 4
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"writer: {args.output}, fps: {fps}, size: ({height},{width})")

    count = 0  # continuous detection times
    stop_execution, stop_execution_start = False, None
    pose_prev = None
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

        pose_print = pose
        if "missing" in pose:
            pose_print = " - "

        # draw detected body parts
        parts, face_bbox = result["parts"], result["face_bbox"]
        image = draw_body_parts(image, parts)
        image = draw_face_bbox(image, face_bbox)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(
            image,
            f"class: {pose_print:^13} fps: {fps:.2f}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )
        cv2.imshow("pose result", image)
        if cv2.waitKey(1) == 27:
            break

        if args.output is not None:
            writer.write(image)

        if stop_execution:
            if (time.time() - stop_execution_start) > action_interval:
                stop_execution = False
            continue

        if "missing_body_part" in pose or pose == "others":
            continue

        if pose == pose_prev:
            count += 1
        else:
            count = 0
            pose_prev = pose

        if pose == "hands-on-head" or count >= recognition_count:
            count = 0
            stop_execution, stop_execution_start = True, time.time()

            # execute action corresponding detected pose
            pose_action = load_config()
            pose_id = relationPose[pose]
            relation = find(pose_action, None, lambda x: x["poseId"] == pose_id)
            execute_action(relation["actionType"], relation["name"])

    if args.output is not None:
        writer.release()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
