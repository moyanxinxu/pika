import json

import cv2 as cv
import mindspore as ms
import pandas as pd
from mindspore.dataset import GeneratorDataset, transforms, vision

with open("../info/config.json", "r") as f:
    config = json.load(f)
    train_img_path = config["train_img_path"]
    train_csv_path = config["train_csv_path"]
    test_img_path = config["test_img_path"]
    test_csv_path = config["test_csv_path"]
    img_size = config["img_size"]
    batch_size = config["batch_size"]

train_compose = transforms.Compose(
    [
        vision.Resize(img_size),
        vision.ToTensor(),
    ]
)
test_compose = transforms.Compose(
    [
        vision.Resize(img_size),
        vision.ToTensor(),
    ]
)


class TinyDataset:
    def __init__(self, csv_path):
        super().__init__()
        self.data = pd.read_csv(csv_path).iloc[:128, :]
        self.length = len(self.data)

        self.img = self.data["img"]
        self.xmin = self.data["xmin"]
        self.ymin = self.data["ymin"]
        self.xmax = self.data["xmax"]
        self.ymax = self.data["ymax"]

        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.img[idx]

        xmin = self.xmin[idx]
        ymin = self.ymin[idx]
        xmax = self.xmax[idx]
        ymax = self.ymax[idx]

        return self.parse(img, xmin, ymin, xmax, ymax)

    def parse(self, img, xmin, ymin, xmax, ymax):
        img = cv.imread(train_img_path + img)
        img = train_compose(img)
        position = ms.tensor([xmin, ymin, xmax, ymax], ms.float32)
        return img, position


def loader(path, flag):
    if flag == "train":
        dataset = TinyDataset(path)
    elif flag == "val":
        dataset = TinyDataset(path)
    elif flag == "test":
        dataset = TinyDataset(path)
    else:
        raise ValueError("flag must be train, val or test".title())

    columns = ["img", "position"]
    shuffle = flag != "test"
    dataset = GeneratorDataset(dataset, columns, shuffle=shuffle)
    batch_loader = dataset.batch(batch_size=batch_size)
    dataloader = batch_loader.create_tuple_iterator()
    return dataloader
