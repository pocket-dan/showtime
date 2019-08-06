import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import utils


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n_vertex = 7
        n_hidden = 128
        n_hidden = 128
        n_classes = 8

        self.fc1 = nn.Linear(n_vertex * 2, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def infer(self, x):
        y = self.forward(x)
        yi = y.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        return yi


class PoseDataset(Dataset):
    def __init__(self, video_dirs: List[Path]):
        self.classes: Dict[str, int] = {
            "pose1-hands-on-head": 0,
            "pose2-victory": 1,
            "pose3-cheer-up": 2,
            "pose4-go-next": 3,
            "pose5-go-back": 4,
            "pose6-ultraman": 5,
            "pose7-others": 6,
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


def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    return train_loss / len(train_loader.dataset)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)

    return test_loss, test_acc


def main():
    # Training settings
    batch_size = 32
    test_batch_size = 1000
    epochs = 400
    seed = 1
    patience = 30  # for early stopping
    use_cuda = torch.cuda.is_available()
    exp_name = "model3"

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    root_dir = Path("dataset/processed")
    train_data_dirs = ["t", "o", "k", "n1", "n2"]
    test_data_dirs = ["m", "m1", "m2"]

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        PoseDataset([root_dir / d for d in train_data_dirs]),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        PoseDataset([root_dir / d for d in test_data_dirs]),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)

    early_stopping = utils.EarlyStopping(patience, f"results/{exp_name}/model.pth")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer)
        test_loss, test_acc = test(model, device, test_loader)
        print(f"epoch: {epoch:>3}, train_loss: {train_loss:.4f}, ", end="")
        print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.3f}")

        early_stopping(test_loss, test_acc, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f"best acc: {early_stopping.best_acc}")

    # save result
    shutil.copyfile("main.py", f"results/{exp_name}/main.py")


if __name__ == "__main__":
    main()
