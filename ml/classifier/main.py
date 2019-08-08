import pickle
import shutil
from pathlib import Path

import numpy as np

import utils
from dataset import PoseDataset

root_dir = Path("dataset/processed")
train_data_dirs = ["t", "o", "k", "n1", "n2"]
test_data_dirs = ["m", "m1", "m2"]


def train_deep():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset
    from model import FCN

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

    # training settings
    batch_size = 32
    test_batch_size = 1000
    epochs = 400
    patience = 30  # for early stopping
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

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

    model = FCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

    early_stopping = utils.EarlyStopping(patience, Path("results"))
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer)
        test_loss, test_acc = test(model, device, test_loader)
        print(f"epoch: {epoch:>3}, train_loss: {train_loss:.4f}, ", end="")
        print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.3f}")

        early_stopping(test_loss, test_acc, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f"deep model acc: {early_stopping.best_acc}")


def aggregate_dataset(dataset):
    X, y = [], []

    for i in range(len(dataset)):
        vertices, klass = dataset.__getitem__(i)
        vertices = vertices.numpy()

        X.append(vertices)
        y.append(klass)

    return np.stack(X), np.stack(y)


def train_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    train_dataset = PoseDataset([root_dir / d for d in train_data_dirs])
    X, y = aggregate_dataset(train_dataset)
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X, y)

    test_dataset = PoseDataset([root_dir / d for d in test_data_dirs])
    X, y = aggregate_dataset(test_dataset)
    y_hat = clf.predict(X)
    acc = accuracy_score(y, y_hat)
    print("random forest acc:", acc)

    with open("./results/random-forest.pickle", mode="wb") as f:
        pickle.dump(clf, f)


def train_lightgbm():
    import lightgbm as lgb

    train_dataset = PoseDataset([root_dir / d for d in train_data_dirs])
    X_train, y_train = aggregate_dataset(train_dataset)

    test_dataset = PoseDataset([root_dir / d for d in test_data_dirs])
    X_test, y_test = aggregate_dataset(test_dataset)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "metric": {"multi_logloss"},
        "num_class": 7,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 1,
        "num_iteration": 200,
        "verbose": 0,
    }

    model = lgb.train(params, lgb_train, valid_sets=lgb_eval, verbose_eval=False)

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_max = np.argmax(y_pred, axis=1)

    accuracy = sum(y_test == y_pred_max) / len(y_test)
    print("lightgbm acc:", accuracy)

    with open("./results/light-gbm.pickle", mode="wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    # train_deep()
    # train_random_forest()
    train_lightgbm()
