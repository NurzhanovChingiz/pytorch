import torch

from torch.nn.parallel import DataParallel as DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to the correct device
model = model.to(device)
model = DDP(model, device_ids=device)

