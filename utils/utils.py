import torch
import torch.nn as nn
import torchvision.transforms as transforms


def accuracy(pred, labels):
    return torch.sum(torch.max(pred, 1)[1] == labels.data).float() / labels.size(0)


def affparam2rect(aff_param):
    assert aff_param.dim() == 2
    assert aff_param.size(1) == 3

    s, x, y = aff_param.t()
    rect = torch.stack([x-s, y-s, x+s, y+s], dim=1)
    assert rect.dim() == 2
    assert rect.size(1) == 4, rect.size()

    return rect


def rect2area(rect):
    assert rect.dim() == 2
    assert rect.size(1) == 4
    return (rect[:, 2] - rect[:, 0]) * (rect[:, 3] - rect[:, 1])


def intersect(rect1, rect2):
    lt = torch.max(rect1[:, :2], rect2[:, :2])
    rb = torch.min(rect1[:, 2:], rect2[:, 2:])
    assert lt.dim() == 2
    assert lt.size(1) == 2
    inter = (rb-lt).clamp(min=0.0)
    intersect = inter[:, 0] * inter[:, 1]
    return intersect


def iou_loss(rect1, rect2):
    area1 = rect2area(rect1)
    area2 = rect2area(rect2)
    inter = intersect(rect1, rect2)
    union = area1 + area2 - inter
    assert union.dim() == 1
    assert area1.size() == area2.size() == inter.size() == union.size()
    return inter.div(union).mean()


def draw_rect(images, aff_param, color=(0.0, 1.0, 0.0), colors=None):
    pts1, pts2 = get_corners(aff_param, images[0].size(1))
    assert pts1.size() == pts2.size() == (images.size(0), 2)

    if colors is None:
        colors = [color] * images.size(0)

    rect_images = torch.stack([
        do_draw_rect(*args)
        for args in zip(images, pts1, pts2, colors)
    ], dim=0)
    return rect_images


def do_draw_rect(img, pt1, pt2, color):
    x1, y1 = pt1
    x2, y2 = pt2

    assert img.dim() == 3

    # Expand channel if needed
    img = img.repeat(3//img.size(0), 1, 1)

    for i, co in enumerate(color):
        img[i, x1, y1:y2+1] = co
        img[i, x2, y1:y2+1] = co
        img[i, x1:x2+1, y1] = co
        img[i, x1:x2+1, y2] = co

    return img


def affparam2st(aff_param):
    assert aff_param.dim() == 2
    assert aff_param.size(1) == 3

    scale, trans = aff_param[:, :1], aff_param[:, 1:]
    return scale, trans


def st2param(scale, trans):
    assert scale.dim() == trans.dim() == 2
    assert scale.size(1) == 1
    assert trans.size(1) == 2
    return torch.cat([scale, trans], dim=1)


def st2mat(scale, trans):
    aff_mat = torch.cat([
        torch.eye(2, device=scale.device).unsqueeze(0) * scale.unsqueeze(1),
        trans.unsqueeze(2)
    ], dim=2)
    assert aff_mat.size()[1:] == (2, 3)
    return aff_mat


def affparam2mat(aff_param):
    return st2mat(*affparam2st(aff_param))



def get_corners(aff_param, sz):

    s, x, y = aff_param.t()

    tx, ty = (sz*(1-s+n)/2 for n in (x, y))
    bx, by = (sz*(1+s+n)/2 for n in (x, y))
    assert (bx-tx).sub(by -
                       ty).abs().lt(1e-3).all(), f'{bx=}\n{tx=}\n{(bx-tx).sub(by-ty).abs()=}'

    tx, ty, bx, by = map(lambda p: p.int().clamp(0, sz-1), [tx, ty, bx, by])

    return [  # I don't know why (y, x) but not (x, y) ...
        torch.stack(pts, dim=1)
        for pts in [[ty, tx], [by, bx]]
    ]


def check_is_better(logits1, logits2, labels):
    criterion_n = nn.CrossEntropyLoss(reduction='none')

    loss1 = criterion_n(logits1, labels)
    loss2 = criterion_n(logits2, labels)

    assert loss1.size() == loss2.size() == (labels.size(0),)
    return loss2 < loss1


def denormalizes(images, mean, std):
    demean, destd = zip(*[
        (-m/s, 1/s)
        for m, s in zip(mean, std)
    ])
    return torch.stack([
        transforms.Normalize(demean, destd)(img)
        for img in images
    ], dim=0)
