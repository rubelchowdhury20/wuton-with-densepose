# Third party imports
import torch
from torchvision import transforms

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preprocessing details
data_transforms = {
	'train': transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((224, 224)),
		# transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

data_transforms_500 = {
	'train': transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((500, 500)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((500, 500)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}


inv_normalize = transforms.Normalize(
	mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
	std=[1/0.229, 1/0.224, 1/0.255])

# geometric transformer variables
TPS_GRID_SIZE = 3


# train related values
LR = 0.0002
TPS_LR = 0.0008													
MOMENTUM = 0.2

RESUME = True

# FREEZE_EPOCHS = 1
# UNFREEZE_EPOCHS = 200

EPOCHS = 1000

PARAMS = {'batch_size': 4,
			'shuffle': True,
			'num_workers': 16}

PARAMS_VAL = {'batch_size': 4,
			'shuffle': False,
			'num_workers': 16}

EMBEDDING_SIZE = 256