class Sum_Activation(nn.Module):
    def __init__(self, activations, init_weights='ones'):
        super().__init__()
        
        self.activations = nn.ModuleList(activations)
        self.weights = nn.Parameter(torch.randn(len(activations)))
        if init_weights == 'ones':
            nn.init.ones_(self.weights)
        if init_weights == 'uniform':
            nn.init.uniform_(self.weights)
        
    def forward(self, x):
        # Compute outputs of all activation functions and stack them along a new dimension
        activations_output = torch.stack([activation(x) for activation in self.activations], dim=0)
        
        # Use broadcasting to multiply weights and sum up the activations
        out = torch.einsum('i,ij...->j...', self.weights, activations_output)
        return out
