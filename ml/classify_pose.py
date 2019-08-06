from typing import Dict, List

import torch

from classifier.model import Net

PARTS: List[str] = [
    "lwrist",
    "lelbow",
    "lshoulder",
    "neck",
    "rshoulder",
    "relbow",
    "rwrist",
]

model = Net()
params = torch.load("./classifier/model.pth", map_location="cpu")
model.load_state_dict(params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def main(body_parts: Dict) -> str:
    _vertices: List[List[float]] = []
    for part in PARTS:
        if part not in body_parts:
            return f"missing_body_part({part})"
        y = body_parts[part]["y"]
        x = body_parts[part]["x"]
        _vertices.append(y)
        _vertices.append(x)
    vertices = torch.tensor([_vertices])

    vertices = vertices.to(device)
    class_number = model.infer(vertices)

    return model.decode_class(class_number)
