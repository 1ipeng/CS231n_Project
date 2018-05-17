import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utils
import os
from scipy.misc import imresize

transformer = transforms.Compose([transforms.ToTensor()]) 
def predict(model, file_path, params, model_dir, restore_file):
	restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
	print("Restoring parameters from {}".format(restore_path))
	utils.load_checkpoint(restore_path, model)
	
	image = plt.imread(file_path)
	resized_image = imresize(image, (params.image_resize, params.image_resize))
	x = transformer(resized_image).unsqueeze(0)
	print(x.shape)
	
	model.eval()
	x = x.to(device=params.device, dtype=torch.float32)
	y_pred = model(x)

	y_pred = y_pred.data.numpy()
	x = x.data.numpy().transpose(0,2,3,1)

	plt.figure()
	utils.draw_boxes_output(x[0], y_pred[0])
	plt.show()


