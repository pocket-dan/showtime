import pickle
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

import utils
from dataset import PoseDataset

root_dir = Path("dataset/processed")
train_data_dirs = ["t", "o", "k", "n1", "n2"]
test_data_dirs = ["m", "m1", "m2"]


def check_acc():
    import torch
    import torch.nn.functional as F

    # deep model
    with open("results/deep-model.pickle", "rb") as f:
        model = pickle.load(f)
    params = torch.load("results/deep-params.pth", map_location="cpu")
    model.load_state_dict(params)

    batchsize = 1000
    test_loader = torch.utils.data.DataLoader(
        PoseDataset([root_dir / d for d in test_data_dirs], mode="test"),
        batch_size=batchsize,
        shuffle=False,
    )

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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

    print(test_acc)


def train_deep():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset
    from model import FCN
    from torch.optim import lr_scheduler

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
    epochs = 500
    patience = 30  # for early stopping
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(9)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        PoseDataset([root_dir / d for d in train_data_dirs]),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        PoseDataset([root_dir / d for d in test_data_dirs], mode="test"),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )

    model = FCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

    early_stopping = utils.EarlyStopping(patience, Path("results"))
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer)
        test_loss, test_acc = test(model, device, test_loader)
        print(f"epoch: {epoch:>3}, train_loss: {train_loss:.4f}, ", end="")
        print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.3f}")

        early_stopping(test_loss, test_acc, model)

        if early_stopping.early_stop:
            print("Early stopping activated")
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
    from sklearn.preprocessing import PolynomialFeatures

    train_dataset = PoseDataset([root_dir / d for d in train_data_dirs])
    X_train, y_train = aggregate_dataset(train_dataset)

    # poly = PolynomialFeatures(2, include_bias=True)
    # poly.fit_transform(X_train)

    test_dataset = PoseDataset([root_dir / d for d in test_data_dirs])
    X_test, y_test = aggregate_dataset(test_dataset)

    # poly.fit_transform(X_test)

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


def train_xgboost():
    import xgboost as xgb
    from sklearn.preprocessing import PolynomialFeatures

    bst = xgb.XGBClassifier(
        base_score=0.5,
        colsample_bytree=1.0,
        gamma=0,
        learning_rate=0.1,
        max_delta_step=0,
        max_depth=5,
        min_child_weight=1,
        missing=None,
        n_estimators=100,
        nthread=-1,
        objective="multi:softprob",
        seed=0,
        silent=True,
        subsample=0.95,
    )

    train_dataset = PoseDataset([root_dir / d for d in train_data_dirs])
    X_train, y_train = aggregate_dataset(train_dataset)

    poly = PolynomialFeatures(2, include_bias=True)
    poly.fit_transform(X_train)

    test_dataset = PoseDataset([root_dir / d for d in test_data_dirs])
    X_test, y_test = aggregate_dataset(test_dataset)

    poly.fit_transform(X_test)

    bst.fit(X_train, y_train)
    y_pred = bst.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("xgboost acc:", acc)


if __name__ == "__main__":
    # train_deep()
    check_acc()
    # train_random_forest()
    # train_lightgbm()
    # train_xgboost()
