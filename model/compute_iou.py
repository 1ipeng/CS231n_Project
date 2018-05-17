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
    '''

    num_objects = boxes_pred.size(0)
    B = boxes_pred.size(1)

    lt = torch.max(
        boxes_pred[:,:,:2],                      # [num_objects, B, 2]
        boxes_true[:,:,:2].expand(num_objects, B, 2)    # [num_objects, 1, 2] -> (num_objects, B, 2]
    )


    rb = torch.min(
        boxes_pred[:,:,2:],                      # [num_objects, B, 2]
        boxes_true[:,:,2:].expand(num_objects, B, 2)    # [num_objects, 1, 2] -> (num_objects, B, 2]
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
    
def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    ### END CODE HERE ###    

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    ### END CODE HERE ###
    
    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area / union_area
    ### END CODE HERE ###
    
    return iou

# box1 = np.array([2., 1., 4., 3.])
# box2 = np.array([1., 2., 3., 4.])
box1 = np.array([2., 1., 4., 3.]).reshape(1,1,4)
box2_1 = np.array([1., 2., 3., 4.])
box2_2 = np.array([2., 1., 4., 3.])
box2_3 = np.array([0., 0., 1., 1.])
boxsum = np.array([box2_1, box2_2, box2_3]).reshape(1,3,4)
boxsum = np.array([boxsum,boxsum,boxsum,boxsum,boxsum]).reshape(5,3,4)
box1 = np.array([box1,box1,box1,box1,box1]).reshape(5,1,4)

num_objects = 5
B = 3

# boxes_pred = np.random.rand(num_objects, B, 4)
# boxes_true = np.random.rand(num_objects, 1, 4)

boxes_pred = boxsum
boxes_true = box1

result = np.zeros((num_objects, B))

for i in range(num_objects):
    for j in range(B):
        result[i, j] = iou(boxes_pred[i, j, :], boxes_true[i, 0, :])

boxes_pred = torch.from_numpy(boxes_pred)
boxes_true = torch.from_numpy(boxes_true)
result2 = compute_iou(boxes_pred, boxes_true).data.numpy()
print((result2 == result).all())
print(result, result2)


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