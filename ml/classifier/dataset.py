import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class PoseDataset(Dataset):
    def __init__(self, video_dirs: List[Path]):
        self.classes: Dict[str, int] = {
            "pose1-hands-on-head": 0,
            "pose2-victory": 1,
            "pose3-cheer-up": 2,
            "pose4-go-next": 3,
            "pose5-go-back": 4,
            "pose6-ultraman": 5,
            "pose7-ultraman": 6,
        }

        self.parts: List[str] = [
            "lwrist",
            "lelbow",
            "lshoulder",
            "neck",
            "rshoulder",
            "relbow",
            "rwrist",
        ]

        items: List[Dict] = []
        for d in video_dirs:
            for c in self.classes:
                path = d / c
                items.extend(self.read_samples(path))

        self.items = items

    def read_samples(self, path: Path) -> List[Dict]:
        samples = []
        class_name = str(path.name)
        for img_path in path.glob("frame_*.jpg"):
            anno_path = img_path.with_suffix(".json")
            samples.append(
                {
                    "img": img_path,
                    "anno": anno_path,
                    "class_name": class_name,
                    "class_number": self.classes[class_name],
                }
            )

        return samples

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        # read annotation
        f = open(item["anno"])
        anno = json.load(f)
        f.close()

        # create vertex list
        _vertices: List[List[float]] = []
        for part in self.parts:
            y = anno["parts"][part]["y"]
            x = anno["parts"][part]["x"]
            _vertices.append([y, x])
        vertices = np.asarray(_vertices, dtype=np.float32)

        # change parts offset randomly to make the model robust against the human position
        ymin, xmin = vertices.min(axis=0)
        ymax, xmax = vertices.max(axis=0)
        _yrange, _xrange = 1 - (ymax - ymin), 1 - (xmax - xmin)

        ry, rx = np.random.rand(), np.random.rand()
        vertices -= np.array([ymin, xmin])
        vertices += np.array([ry * _yrange, rx * _xrange])
        assert np.all(vertices >= 0)
        assert np.all(vertices <= 1)

        # class number
        class_number = item["class_number"]

        return torch.tensor(vertices.flatten()), class_number
