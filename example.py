import os
import utils
import matplotlib.pyplot as plt
import numpy as np
from YOLO import YOLO

params = utils.Params('experiment/params.json')
params.device = "cpu"

images = []
for i in range(32):
	name = utils.get_image_name(i)
	image = plt.imread('./data/raw_GTSDB/' + name)
	images.append(image)
images = np.array(images)

yolo = YOLO(params)
output = yolo.predict(images)

for i in range(output.shape[0]):
	plt.subplot(4, 8, i+1)
	plt.imshow(output[i])
plt.show()

