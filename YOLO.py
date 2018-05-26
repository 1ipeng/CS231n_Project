import utils
import os
import torch
import model.data_loader as data_loader
import model.net as net
import torch.optim as optim
import train_fn
import torchvision.transforms as transforms
import numpy as np

class YOLO(object):
	def __init__(self, params):
		self.model = net.darknet(params)
		self.params = params

	def train(self, data_dir = 'data/GTSDB/', model_dir = 'experiment/', restore = None):
		print("Loading the datasets...")
		data = data_loader.fetch_dataloader(['train', 'val'], data_dir, self.params)
		train_data = data['train']
		val_data = data['val']
		print("- done.")
		optimizer = optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
		loss_fn = net.yolo_v1_loss
		print("Starting training for {} epoch(s)".format(self.params.num_epochs))
		train_fn.train_and_evaluate(self.model, train_data, val_data, optimizer, loss_fn, params, model_dir, restore)

	def predict(self, images, model_dir = 'experiment/', restore = 'best'):
		restore_path = os.path.join(model_dir, restore + '.pth.tar')
		print("Restoring parameters from {}".format(restore_path))
		utils.load_checkpoint(restore_path, self.model, self.params)
		
		transformer = transforms.Compose([transforms.ToPILImage(),
										  transforms.Resize((self.params.image_resize, self.params.image_resize)), 
			                              transforms.ToTensor()])

		x = torch.stack([transformer(image) for image in images])
		self.model.eval()
		x = x.to(device=self.params.device, dtype=torch.float32)
		y_pred = self.model(x)
		y_pred = y_pred.data.numpy()
		output = [utils.pred_to_crop(image, y, self.params) for image, y in zip(images, y_pred)]
		output = np.concatenate([o for o in output if o.shape[0] > 0])
		return output