import json

import mindspore as ms
from dataset import loader
from mindspore import nn
from mindspore.train.serialization import save_checkpoint
from models import Net
from tools import get_loss, get_truth
from tqdm import trange, tqdm

try:
    ms.set_context(device_target="GPU")
except:
    ms.set_context(device_target="CPU")

print(f'Training on {ms.context.get_context("device_target")}')


with open("../info/config.json", "r") as f:
    config = json.load(f)
    train_csv_path = config["train_csv_path"]
    test_csv_path = config["test_csv_path"]
    img_size = config["img_size"]
    epoches = config["epoches"]
    device = config["device"]
    lr = config["lr"]


net = Net()
train_loader = loader(train_csv_path, flag="train")
test_loader = loader(test_csv_path, flag="test")


def forward_fn(label_pred, offset_pred, label, offset, masks):
    loss_value = get_loss(label_pred, offset_pred, label, offset, masks)
    return loss_value


optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=lr)
grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)


def train_set(label_pred, offset_pred, label, offset, masks):
    loss, grads = grad_fn(label_pred, offset_pred, label, offset, masks)
    optimizer(grads)
    return loss


def test_set(label_pred, offset_pred, label, offset, masks):
    loss = get_loss(label_pred, offset_pred, label, offset, masks)
    return loss


best = -ms.numpy.inf


for epoch in range(epoches):
    l = 0
    for img, position in train_loader:
        anchor, label_pred, offset_pred = net(img)
        label, offset, masks = get_truth(anchor, position)
        loss = train_set(label_pred, offset_pred, label, offset, masks)
        print(epoch, loss)
    for img, position in test_loader:
        anchor, label_pred, offset_pred = net(img)
        label, offset, masks = get_truth(anchor, position)
        loss = test_set(label_pred, offset_pred, label, offset, masks)
        l += loss.item()
        print(epoch, loss)
    if l > best:
        best = l
        save_checkpoint(net, "../model/anchor.ckpt")
