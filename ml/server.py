from typing import Dict

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_api import status

from classify_pose import main as _classify_pose
from estimate_pose import main as _estimate_pose

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify("Hey! I'm ML server!")


@app.route("/infer", methods=["POST"])
def infer():
    # decode request body (image bytes) to numpy array
    _bytes = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(_bytes, flags=cv2.IMREAD_COLOR)
    img_h, img_w, img_c = img.shape

    # estimate pose
    detected_humans = _estimate_pose(img)
    if len(detected_humans) == 0:
        jsonify({})

    human_max_score = max(detected_humans, key=lambda h: h.score)

    response = {}

    # parts
    parts = {}
    for p in human_max_score.body_parts.values():
        key = p.get_part_name().name.lower()
        parts[key] = {"x": p.x, "y": p.y, "score": p.score}
    response["parts"] = parts

    # face bbox
    response["face_bbox"] = human_max_score.get_face_box(img_w, img_h)

    # classify pose
    response["pose_class"] = _classify_pose(parts)

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)  # , use_reloader=True)
