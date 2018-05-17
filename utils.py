import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

def draw_boxes_xy(image, boxes_xy):
	# boxes_xy: shape (N, 4), [[x1, y1, x2, y2], ...] 
	# x1, y1: coordinates of top left point
	# x2, y2: coordinates of right bottom point
	ax = plt.gca()
	plt.imshow(image)
	for i in range(boxes_xy.shape[0]):
		x1, y1, x2, y2 = boxes_xy[i]
		w = x2 - x1
		h = y2 - y1
		box = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
		ax.add_patch(box)

def draw_boxes_output(image, y):
	# y: shape (num_grid, num_grid, 5 * B)
	num_grid = y.shape[0]
	B = int(y.shape[2] / 5)

	y_pred = y.reshape(num_grid, num_grid, B, 5)
	mask = (y_pred[:, :, :, 0] > 0.5)
	indices = np.argwhere(y_pred[:, :, :, 0] > 0.5)

	boxes_cwh = y_pred[mask, 1:5]
	image_hw = image.shape[0:2]
	boxes_xy = np.zeros_like(boxes_cwh)
	
	for i in range(boxes_cwh.shape[0]):
		positon = indices[i, 0:2]
		box_cwh = denormalize_box_cwh(image_hw, num_grid, boxes_cwh[i], positon)
		boxes_xy[i] = cwh_to_xy(box_cwh)
	draw_boxes_xy(image, boxes_xy)

def xy_to_cwh(box_xy):
	# box_xy [x1, y1, x2, y2]
	# Given top left point and right bottom point coordinates
	# Compute center coordinates, height and weight
	x1, y1, x2, y2 = box_xy
	xc = (x1 + x2) / 2
	yc = (y1 + y2) / 2
	w = x2 - x1
	h = y2 - y1
	cwh = [xc, yc, w, h]
	return cwh

def cwh_to_xy(box_cwh):
	# box_cwh (xc, yc, w, h)
	# Given top left point and right bottom point coordinates
	# Compute center coordinates, height and weight
	xc, yc, w, h = box_cwh
	x1 = xc - w / 2
	x2 = xc + w / 2
	y1 = yc - h / 2
	y2 = yc + h / 2
	xy = [x1, y1, x2, y2]
	return xy

def xy_to_cwh_vectorize(boxes_xy):
	# boxes_xy: shape (N, 4), [[x1, y1, x2, y2], ...]
	# Given top left point and right bottom point coordinates
	# Compute center coordinates, height and weight
	# Return boxes_cwh: shape (N, 4) [[xc, yc, w, h], ...]
	boxes_cwh = np.zeros_like(boxes_xy)
	boxes_cwh[:, 0] = (boxes_xy[:, 0] + boxes_xy[:, 2]) / 2
	boxes_cwh[:, 1] = (boxes_xy[:, 1] + boxes_xy[:, 3]) / 2
	boxes_cwh[:, 2] = boxes_xy[:, 2] - boxes_xy[:, 0]
	boxes_cwh[:, 3] = boxes_xy[:, 3] - boxes_xy[:, 1]
	return boxes_cwh

def cwh_to_xy_vectorize(boxes_cwh):
	# boxes_cwh: shape (N, 4) [[xc, yc, w, h], ...]
	# Given top left point and right bottom point coordinates
	# Compute center coordinates, height and weight
	# boxes_xy: shape (N, 4), [[x1, y1, x2, y2], ...]
	boxes_xy = zeros_like(boxes_cwh)
	boxes_xy[:, 0] = boxes_cwh[:, 0] - boxes_cwh[:, 2] / 2
	boxes_xy[:, 1] = boxes_cwh[:, 1] - boxes_cwh[:, 3] / 2
	boxes_xy[:, 2] = boxes_cwh[:, 0] + boxes_cwh[:, 2] / 2
	boxes_xy[:, 3] = boxes_cwh[:, 1] + boxes_cwh[:, 3] / 2
	return boxes_xy

def resize_box_xy(orig_hw, resized_hw, box_xy):
	# Resize box
	# orig_h, orig_w: orginal image size
	# resized_h, resized_w: resized image size
	# x1, y1, x2, y2: orginal box coords
	orig_h, orig_w = orig_hw
	resized_h, resized_w = resized_hw
	x1, y1, x2, y2 = box_xy
	w_ratio = 1. * resized_w / orig_w
	h_ratio = 1. * resized_h / orig_h
	resized_x1 = x1 * w_ratio
	resized_x2 = x2 * w_ratio
	resized_y1 = y1 * h_ratio
	resized_y2 = y2 * h_ratio
	resized_xy = [resized_x1, resized_y1, resized_x2, resized_y2]
	return resized_xy


def normalize_box_cwh(image_hw, num_grid, box_cwh):
	# Normalize box height and weight to be 0-1
	image_h, image_w = image_hw
	xc, yc, box_w, box_h = box_cwh
	normalized_w = 1. * box_w / image_w
	normalized_h = 1. * box_h / image_h

	grid_w = 1. * image_w / num_grid 
	grid_h = 1. * image_h / num_grid 
	col = int(xc / grid_w)
	row = int(yc / grid_h)
	normalized_xc = 1. * (xc - col * grid_w) / grid_w
	normalized_yc = 1. * (yc - row * grid_h) / grid_h
	normalized_cwh = [normalized_xc, normalized_yc, normalized_w, normalized_h]
	positon = [row, col]
	return normalized_cwh, positon

def denormalize_box_cwh(image_hw, num_grid, norm_box_cwh, grid):
	image_h, image_w = image_hw 
	normalized_xc, normalized_yc, normalized_w, normalized_h = norm_box_cwh
	row, col = grid

	box_w = normalized_w * image_w
	box_h = normalized_h * image_h
	grid_w = 1. * image_w / num_grid 
	grid_h = 1. * image_h / num_grid 
	xc = normalized_xc * grid_w + col * grid_w 
	yc = normalized_yc * grid_h + row * grid_h
	cwh = [xc, yc, box_w, box_h]
	return cwh

def get_image_name(i):
	if i < 10:
		name = '0000' + str(i) + '.ppm'
	elif i < 100:
		name = '000' + str(i) + '.ppm'
	else:
		name = '00' + str(i) + '.ppm'
	assert(len(name) == 9)
	return name

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint