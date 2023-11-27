from itertools import product

import mindspore as ms
import numpy as np
from mindspore import nn, ops


def calculate_bbox(points, size, direction=1):
    half_size = ms.tensor(
        [
            size * (np.sqrt(direction)) / 2.0,
            size / (np.sqrt(direction)) / 2.0,
        ]
    )
    xmin_ymin = points - half_size
    xmax_ymax = points + half_size
    return xmin_ymin, xmax_ymax


def get_anchor(img_size, small_size, big_size):
    step = (ops.arange(img_size) + 0.5) / img_size

    step = step.asnumpy()

    centers = list(product(step, step))
    centers = ms.tensor(centers)

    a_xmin_ymin, a_xmax_ymax = calculate_bbox(centers, big_size)
    b_xmin_ymin, b_xmax_ymax = calculate_bbox(centers, small_size)
    c_xmin_ymin, c_xmax_ymax = calculate_bbox(centers, small_size, 2)
    d_xmin_ymin, d_xmax_ymax = calculate_bbox(centers, small_size, 2)

    anchors = ops.cat(
        [
            ops.cat([a_xmin_ymin, a_xmax_ymax], axis=1),
            ops.cat([b_xmin_ymin, b_xmax_ymax], axis=1),
            ops.cat([c_xmin_ymin, c_xmax_ymax], axis=1),
            ops.cat([d_xmin_ymin, d_xmax_ymax], axis=1),
        ],
        axis=0,
    )
    return anchors


# TODO 完成多对多的IOU计算
def get_iou(anchor, target):
    width = anchor[:, 2] - anchor[:, 0]
    height = anchor[:, 3] - anchor[:, 1]
    y_hat_area = width * height

    width = target[2] - target[0]
    height = target[3] - target[1]
    y_area = width * height

    xmin = ops.maximum(anchor[:, 0], target[0])
    ymin = ops.maximum(anchor[:, 1], target[1])
    xmax = ops.minimum(anchor[:, 2], target[2])
    ymax = ops.minimum(anchor[:, 3], target[3])

    cross_w = ops.maximum(xmax - xmin, 0)
    cross_h = ops.maximum(ymax - ymin, 0)

    cross_area = cross_w * cross_h

    s = y_hat_area + y_area - cross_area

    iou = cross_area / s
    return iou


def get_offset(anchor, target):
    anchor_width = anchor[2] - anchor[0]
    anchor_height = anchor[3] - anchor[1]
    target_width = target[2] - target[0]
    target_height = target[3] - target[1]

    anchor_x = (anchor[0] + anchor[2]) / 2
    anchor_y = (anchor[1] + anchor[3]) / 2
    target_x = (target[0] + target[2]) / 2
    target_y = (target[1] + target[3]) / 2
    target_x = (target[0] + target[1]) / 2
    target_y = (target[2] + target[3]) / 2

    offset_x = (target_x - anchor_x) / anchor_width * 10
    offset_y = (target_y - anchor_y) / anchor_height * 10
    offset_w = ops.log(1e-6 + target_width / anchor_width) * 5
    offset_h = ops.log(1e-6 + target_height / anchor_height) * 5

    return ms.tensor(
        [x.item() for x in [offset_x, offset_y, offset_w, offset_h]], dtype=ms.float32
    )


def get_active(anchor, target):
    iou = get_iou(anchor, target)
    active = [1 if item >= 0.5 else 0 for item in iou]
    active[ops.argmax(iou)] = 1
    return ms.tensor(active, dtype=ms.int32)


def get_mask(active):
    length = active.shape[0]
    mask = ops.zeros(size=(length, 4), dtype=ms.int32)
    mask[active, :] = 1
    return ms.tensor(mask)


def get_label(active):
    length = active.shape[0]
    label = ops.zeros(length, dtype=ms.int32)

    label[active] = 1
    return ms.tensor(label)


def get_active_offset(active, anchor, target):
    length = active.shape[0]
    offset = ops.zeros(size=(length, 4), dtype=ms.int32)

    for i in range(length):
        if active[i]:
            offset[i, :] = get_offset(anchor[i], target)
    return offset


def get_truth(anchor, target):
    labels = []
    offsets = []
    masks = []

    for i in range(len(target)):
        active = get_active(anchor, target[i])

        mask = get_mask(active)
        masks.append(mask.reshape(-1))

        label = get_label(active)
        labels.append(label)

        offset = get_active_offset(active, anchor, target[i])
        offsets.append(offset.reshape(-1))

    labels = ops.stack(labels)
    offsets = ops.stack(offsets)
    masks = ops.stack(masks)

    return labels, offsets, masks


get_loss_cls = nn.CrossEntropyLoss(reduction="none")
get_loss_box = nn.L1Loss(reduction="none")


def get_loss(label_pred, offset_pred, label, offset, masks):
    label_pred = label_pred.reshape(-1, 2)
    label = label.reshape(-1)
    label = ms.tensor(label, dtype=ms.int32)
    loss_cls = get_loss_cls(label_pred, label)
    loss_cls = loss_cls.reshape(32, -1)
    loss_cls = ops.mean(loss_cls, axis=1)

    offset_pred *= masks
    offset *= masks

    loss_box = get_loss_box(offset_pred, offset)
    loss_box = loss_box.mean(axis=1)

    loss = loss_box + loss_cls
    return loss


def inverse_offset(anchor, offset):
    anchor_center = ms.Tensor([0.0, 0, 0, 0], ms.float32)
    anchor_center[0] = (anchor[0] + anchor[2]) / 2
    anchor_center[1] = (anchor[1] + anchor[3]) / 2
    anchor_center[2] = anchor[2] - anchor[0]
    anchor_center[3] = anchor[3] - anchor[1]

    pred = ms.Tensor([0.0, 0, 0, 0], ms.float32)
    pred[0] = (offset[0] * anchor_center[2] * 0.1) + anchor_center[0]
    pred[1] = (offset[1] * anchor_center[3] * 0.1) + anchor_center[1]
    pred[2] = ops.exp(offset[2] / 5) * anchor_center[2]
    pred[3] = ops.exp(offset[3] / 5) * anchor_center[3]

    pred_corner = ms.Tensor([0.0, 0, 0, 0], ms.float32)
    pred_corner[0] = pred[0] - 0.5 * pred[2]
    pred_corner[1] = pred[1] - 0.5 * pred[3]
    pred_corner[2] = pred[0] + 0.5 * pred[2]
    pred_corner[3] = pred[1] + 0.5 * pred[3]

    return pred_corner
