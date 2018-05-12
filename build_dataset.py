# Build dataset
import numpy as np
from scipy.misc import imresize
from util import *
import random
import os
from tqdm import trange

data_dir = './data/raw_GTSDB/'
output_dir = './data/GTSDB/'
image_resize = 416
num_grid = 13

train_size = 700
val_size = 100
test_size = 100

raw_data = np.loadtxt(data_dir + 'gt.txt', delimiter = ';', dtype= str)
image_names = raw_data[:, 0]
box_coords = raw_data[:, 1:5].astype(float)

X = []
Y = []

for i in trange(900):
	# Load and resize ith image
	name = get_image_name(i)
	image = plt.imread(data_dir + name)
	resized_image = imresize(image, (image_resize, image_resize))
	X.append(resized_image)

	# Load bounding boxes
	y = []
	orig_hw = image.shape[0:2]
	resized_hw = resized_image.shape[0:2]
	indices = np.argwhere(image_names == name).reshape(-1,)
	
	for index in indices:
		box_xy = box_coords[index]
		resized_box_xy = resize_box_xy(orig_hw, resized_hw, box_xy)
		y.append(resized_box_xy)
	y = np.array(y)
	Y.append(y)

X = np.array(X)
Y = np.array(Y)
permutation = list(np.random.permutation(900))
train = permutation[0:train_size]
val = permutation[train_size : train_size + val_size]
test = permutation[train_size + val_size:]

X_train = X[train]
X_val = X[val]
X_test = X[test]
Y_train = Y[train]
Y_val = Y[val]
Y_test = Y[test]

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

print('Build dataset done.')
print('Train shape:', X_train.shape, Y_train.shape)
print('Val shape:', X_val.shape, Y_val.shape)
print('Test shape:', X_test.shape, Y_test.shape)
np.save(output_dir + 'X_train', X_train)
np.save(output_dir + 'Y_train', Y_train)
np.save(output_dir + 'X_val', X_val)
np.save(output_dir + 'Y_val', Y_val)
np.save(output_dir + 'X_test', X_test)
np.save(output_dir + 'Y_test', Y_test)

'''
# Test Case
i = 100
image = plt.imread(data_dir + image_names[i])
box_xy = box_coords[i]

resized_image = imresize(image, (image_resize, image_resize))
orig_hw = image.shape[0:2]
resized_hw = resized_image.shape[0:2]
resized_box_xy = resize_box_xy(orig_hw, resized_hw, box_xy)
box_cwh = xy_to_cwh(resized_box_xy)
norm_box_cwh, grid = normalize_box_cwh(resized_hw, num_grid, box_cwh)
denorm_box_cwh = denormalize_box_cwh(resized_hw, num_grid, norm_box_cwh, grid)

denorm_w, denorm_h, denorm_xc, denorm_yc = denorm_box_cwh

plt.figure()
draw_box_cwh(resized_image, denorm_box_cwh)
# draw_box_xy(image, x1, y1, x2, y2)
plt.show()
'''
