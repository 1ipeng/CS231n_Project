#encoding:utf-8
import torch
import numpy as np

def compute_iou(boxes_pred, boxes_true):
    '''
    Compute intersection over union of two set of boxes
    Args:
      boxes_pred: shape (num_objects, B, 4)
      boxes_true: shape (num_objects, 1, 4)
    Return:
      iou: shape (num_objects, B)

    说明:
    每个objects会有B个预测的boxes, 而实际只有一个box，所以boxes_pred的第二个dim是B，而boxes_true第二个dim是1
    每个box有四个坐标(x1, y1, x2, y2)，分别是左上角和右下角的坐标，所以最后一个dim是4

    返回：每个object的B个pred_box分别和该object的true_box计算iou，得到B个iou。
    要求：(1)vectorize, 不用for loop. (2)用torch，不用numpy
    '''

    num_objects = boxes_pred.size(0)
    B = boxes_pred.size(1)

    lt = torch.max(
        boxes_pred[:,:,:2]                      # [num_objects, B, 2]
        boxes_true.expand(num_objects, B, 2)    # [num_objects, 1, 2] -> (num_objects, B, 2]
    )

    rb = torch.min(
        boxes_pred[:,:,2:]                      # [num_objects, B, 2]
        boxes_true.expand(num_objects, B, 2)    # [num_objects, 1, 2] -> (num_objects, B, 2]
    )

    wh = rb - lt # width and height => [num_objects, B, 2]
    wh[wh<0] = 0 # if no intersection, set to zero
    inter = wh[:,:,0] * wh[:,:,1] # [num_objects, B]

    # [num_objects, B, 1] * [num_objects, B, 1] -> [num_objects, B]
    area1 = (boxes_pred[:,:,2]-boxes_pred[:,:,0]) * (boxes_pred[:,:,2]-boxes_pred[:,:,0]) 
    
    # [num_objects, 1, 1] * [num_objects, 1, 1] -> [num_objects, 1] -> [num_objects, B]
    area2 = ((boxes_true[:,:,2]-boxes_true[:,:,0]) * (boxes_true[:,:,2]-boxes_true[:,:,0])).expand(num_objects, B)

    iou = inter / (area1 + area2 - inter) # [num_objects, B]

    return iou
    


def compute_iou_reference(self, box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh<0] = 0  # clip at 0
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou