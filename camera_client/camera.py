import cv2
import json
import urllib.parse
import urllib.request
from mutagen.mp3 import MP3 as mp3
import pygame
import time
import osascript

relationPose = {
    "hands-on-head": 1,
    "victory": 2,
    "cheer-up": 3,
    "go-next": 4,
    "go-back": 5,
    "ultraman": 6
}

class PoseCaptureCamera:
    def post_frame():
        cap = cv2.VideoCapture(1)

        # flag = 0
        while True:
            _, frame = cap.read()
            _, jpgbytes = cv2.imencode(".jpg", frame)
            reqbody = jpgbytes.tobytes()

            url = "https://raahii2.serveo.net/infer"
            req = urllib.request.Request(
                url,
                reqbody,
                method="POST",
                headers={"Content-Type": "application/octet-stream"},
            )
            with urllib.request.urlopen(req) as res:
                pose = json.loads(res.read())
                cv2.putText(frame, pose["pose_class"], (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,200), 2, cv2.LINE_AA)
                cv2.imshow('Raw Frame', frame)
                pose_class = pose["pose_class"]
                if "missing_body_part" in pose_class or "others" == pose_class:
                    flag = 0
                else:
                    flag = 0
                    print(pose_class)
                    type, action = identify_action(pose_class)
                    execute_action(type, action)

            k = cv2.waitKey(1) # （ESC）キーで終了
            if k == 27 or flag == 1:
                break

        cap.release()
        cv2.destroyAllWindows()


def exec_apple_script(action):
    if action == "move-next":
        osascript.run('''
            tell application "Microsoft PowerPoint"
                activate
                tell application "System Events"
                    keystroke (ASCII character 29)
                end tell
            end tell
            ''')
    elif action == "move-prev":
        osascript.run('''
            tell application "Microsoft PowerPoint"
                activate
                tell application "System Events"
                    keystroke (ASCII character 28)
                end tell
            end tell
            ''')

def play_music(filename):
    pygame.mixer.init()
    filename = './sounds/' + filename
    pygame.mixer.music.load(filename) #音源を読み込み
    mp3_length = mp3(filename).info.length #音源の長さ取得
    pygame.mixer.music.play(1) #再生開始。1の部分を変えるとn回再生(その場合は次の行の秒数も×nすること)
    time.sleep(mp3_length + 0.25) #再生開始後、音源の長さだけ待つ(0.25待つのは誤差解消)
    pygame.mixer.music.stop() #音源の長さ待ったら再生停止

def load_relation_pose_action():
    f = open('pose_action.json')
    pose_action = json.load(f)
    return pose_action["data"]

def identify_action(pose_class):
    pose_action = load_relation_pose_action()
    if pose_class == "hands-on-head":
        for relation in pose_action:
            if relation["poseId"] == relationPose[pose_class]:
                action = relation["name"]
                type = relation["actionType"]
    elif pose_class == "victory":
        for relation in pose_action:
            if relation["poseId"] == relationPose[pose_class]:
                action = relation["name"]
                type = relation["actionType"]
    elif pose_class == "cheer-up":
        for relation in pose_action:
            if relation["poseId"] == relationPose[pose_class]:
                action = relation["name"]
                type = relation["actionType"]
    elif pose_class == "go-next":
        for relation in pose_action:
            if relation["poseId"] == relationPose[pose_class]:
                action = relation["name"]
                type = relation["actionType"]
    elif pose_class == "go-back":
        for relation in pose_action:
            if relation["poseId"] == relationPose[pose_class]:
                action = relation["name"]
                type = relation["actionType"]
    elif pose_class == "ultraman":
        for relation in pose_action:
            if relation["poseId"] == relationPose[pose_class]:
                action = relation["name"]
                type = relation["actionType"]
    return type, action

def execute_action(type, action):
    if type == "slide":
        exec_apple_script(action)
    elif type == "sound":
        filename = action + ".mp3"
        play_music(filename)

p = PoseCaptureCamera
p.post_frame()
