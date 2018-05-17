import argparse
import utils
import os
import torch
import model.data_loader as data_loader
import model.net as net
import torch.optim as optim
import train
import predict

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/GTSDB/', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiment/', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="last or best")
parser.add_argument("--train", help="train model", action="store_true")
parser.add_argument("--predict", help="predict bouding box", action="store_true")
parser.add_argument("--filepath", default=None, help="predicted image filepath")

# Load the parameters from json file
args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
params = utils.Params(json_path)

# Check GPU available
if not torch.cuda.is_available():
	params.cuda = False

if params.cuda:
	params.device = "cuda"
	torch.cuda.manual_seed(231)
else:
	params.device = "cpu"

# Set the random seed for reproducible experiments
torch.manual_seed(231)
	

# Define the model
model = net.darknet(params)
if params.device == "cuda":
	model = model.cuda()

# Train mode
if args.train:
	# Create the input data pipeline
	print("Loading the datasets...")

	# load data
	data = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
	train_data = data['train']
	val_data = data['val']

	print("- done.")

	# Define optimizer
	optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

	# fetch loss function
	loss_fn = net.yolo_v1_loss

	# Train the model
	print("Starting training for {} epoch(s)".format(params.num_epochs))
	train.train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, params, args.model_dir,
	                   args.restore_file)

if args.predict:
	if args.filepath == None:
		print("Missing predicted image")
	elif args.restore_file == None:
		print("Missing weight file")
	else:
		predict.predict(model, args.filepath, params, args.model_dir, args.restore_file)