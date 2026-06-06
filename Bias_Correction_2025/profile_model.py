import torch
import time
import sys
sys.path.insert(0, '.')
from robust.models.msua_net import MSUANet

print("Creating model...")
device = 'cuda'
model = MSUANet(
    in_channels=10,
    out_channels=1,
    base_filters=64,
    num_levels=4,
    uncertainty=True
).to(device)

# Create dummy input
B, C, H, W = 16, 10, 257, 269  # Batch size 16
x = torch.randn(B, C, H, W).to(device)

print(f"Input shape: {x.shape}")
print("Warming up...")
with torch.no_grad():
    model(x)

print("Profiling forward pass...")
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    model(x)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Forward pass time: {elapsed:.4f} seconds")

# Profile backward pass
print("Profiling backward pass...")
model.train()
optimizer = torch.optim.AdamW(model.parameters())

torch.cuda.synchronize()
start = time.time()

# Forward
output, _ = model(x)
loss = output.mean()

# Backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Training step (fwd+bwd) time: {elapsed:.4f} seconds")
