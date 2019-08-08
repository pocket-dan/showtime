import pickle
from typing import Dict, List

import lightgbm
import numpy as np
import torch

from classifier.model import FCN

PARTS: List[str] = [
    "lwrist",
    "lelbow",
    "lshoulder",
    "neck",
    "rshoulder",
    "relbow",
    "rwrist",
]

# model = torch.load("classifier/results/deep-model.pth", map_location="cpu")
# params = torch.load("classifier/results/deep-params.pth", map_location="cpu")
# model.load_state_dict(params)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

f = open("classifier/results/light-gbm.pickle", mode="rb")
model = pickle.load(f)
f.close()


def main(body_parts: Dict) -> str:
    _vertices: List[List[float]] = []
    for part in PARTS:
        if part not in body_parts:
            return f"missing_body_part({part})"
        y = body_parts[part]["y"]
        x = body_parts[part]["x"]
        _vertices.append(y)
        _vertices.append(x)

    # vertices = torch.tensor([_vertices])
    # vertices = vertices.to(device)
    # class_number = model.infer(vertices)

    vertices = np.asarray(_vertices)
    class_number = model.predict(vertices)

    return model.decode_class(class_number)
