from mindspore import nn, ops
from tools import get_anchor


def conv(in_channels, out_channels, kernel_size, pad_mode="pad", padding=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        pad_mode=pad_mode,
        padding=padding,
    )


class FirstModel(nn.Cell):
    def __init__(self):
        super().__init__()

        self.cnn = nn.SequentialCell(
            [
                # [b,3,256,256] --> [b,16,256,256]
                conv(3, 16, 3),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # [b,16,256,256] --> [b,16,256,256]
                conv(16, 16, 3),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # [b,16,256,256] --> [b,16,128,128]
                nn.MaxPool2d(2, 2),
                # [b,16,128,128] --> [b,32,128,128]
                conv(16, 32, 3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # [b,32,128,128] --> [b,32,128,128]
                conv(32, 32, 3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # [b,32,128,128] --> [b,32,64,64]
                nn.MaxPool2d(2, 2),
                # [b,32,64,64] --> [b,64,64,64]
                conv(32, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # [b,64,64,64] --> [b,64,64,64]
                conv(64, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # [b,64,64,64] --> [b,64,32,32]
                nn.MaxPool2d(2, 2),
            ]
        )

        # [b,64,32,32] --> [b,8,32,32]
        self.label = conv(64, 8, 3)

        # num_classes * 4
        # as num_anchors = 4, num_labels = 2

        # [b,64,32,32] --> [b,2,32,32]
        self.offset = conv(64, 16, 3)

    def construct(self, img):
        features = self.cnn(img)

        anchor = get_anchor(32, 0.2, 0.272)

        # 连接多尺度的特征图
        label = self.label(features)
        label = label.permute(0, 2, 3, 1)
        label = nn.Flatten(start_dim=1)(label)

        # 同理,连接多尺度的特征图
        offset = self.offset(features)
        offset = offset.permute(0, 2, 3, 1)
        offset = nn.Flatten(start_dim=1)(offset)

        return features, anchor, label, offset


class MiddleModel(nn.Cell):
    def __init__(self, in_channel, anchor_size_small, anchor_size_big):
        super().__init__()

        self.anchor_size_small = anchor_size_small
        self.anchor_size_big = anchor_size_big

        self.cnn = nn.SequentialCell(
            [
                conv(in_channel, 128, 3),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                conv(128, 128, 3),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            ]
        )

        self.label = conv(128, 8, 3)

        self.offset = conv(128, 16, 3)

    def construct(self, img):
        features = self.cnn(img)
        anchor = get_anchor(
            features.shape[-1], self.anchor_size_small, self.anchor_size_big
        )

        label = self.label(features)
        label = label.permute(0, 2, 3, 1)
        label = nn.Flatten(start_dim=1)(label)

        offset = self.offset(features)
        offset = offset.permute(0, 2, 3, 1)
        offset = nn.Flatten(start_dim=1)(offset)

        return features, anchor, label, offset


class LastModel(nn.Cell):
    def __init__(self):
        super().__init__()

        self.cnn = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.label = conv(128, 8, 3)

        self.offset = conv(128, 16, 3)

    def construct(self, img):
        img = self.cnn(img)

        anchor = get_anchor(1, 0.88, 0.961)

        label = self.label(img)
        label = label.permute(0, 2, 3, 1)
        label = nn.Flatten(start_dim=1)(label)

        offset = self.offset(img)
        offset = offset.permute(0, 2, 3, 1)
        offset = nn.Flatten(start_dim=1)(offset)

        return img, anchor, label, offset


class Net(nn.Cell):
    def __init__(self):
        super().__init__()

        self.first = FirstModel()
        self.middle_1 = MiddleModel(
            in_channel=64,
            anchor_size_small=0.37,
            anchor_size_big=0.447,
        )

        self.middle_2 = MiddleModel(
            in_channel=128,
            anchor_size_small=0.54,
            anchor_size_big=0.619,
        )

        self.middle_3 = MiddleModel(
            in_channel=128,
            anchor_size_small=0.71,
            anchor_size_big=0.79,
        )

        self.last = LastModel()

    def construct(self, img):
        anchor = [None] * 5
        label = [None] * 5
        offset = [None] * 5

        # [2,3,256,256] --> [2,64,32,32],[4096,4],[]
        x, anchor[0], label[0], offset[0] = self.first(img)
        x, anchor[1], label[1], offset[1] = self.middle_1(x)
        x, anchor[2], label[2], offset[2] = self.middle_2(x)
        x, anchor[3], label[3], offset[3] = self.middle_3(x)
        x, anchor[4], label[4], offset[4] = self.last(x)

        anchor = ops.cat(anchor, 0)
        label = ops.cat(label, 1)

        label = label.reshape(label.shape[0], -1, 2)

        offset = ops.cat(offset, 1)

        return anchor, label, offset
