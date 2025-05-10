import torch
DDP = torch.nn.DataParallel 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to the correct device
model = model.to(device)
model = DDP(model, device_ids=device)