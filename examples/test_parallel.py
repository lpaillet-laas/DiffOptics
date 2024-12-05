import torch

# Example tensor A: [batch_size, nC, h, w]
batch_size, nC, h, w = 1, 2, 3, 3
A = torch.randn(batch_size, nC, h, w)

# Example tensor B: [nC, nb_rays, N, 2] (indices for h and w dimensions)
nb_rays, N = 4, 3
B = torch.randint(0, h, (nC, nb_rays, N, 2))

# We need to extract the batch and channel dimensions separately for easier indexing.
# First, let's separate the indices from B for h and w dimensions:
B_h = B[:, :, :, 0]  # Indices for h
B_w = B[:, :, :, 1]  # Indices for w

# Now we need to gather values from A at these (h, w) positions for each batch and channel.
# We will broadcast the batch and channel indices.

# Create a meshgrid of the batch and channel indices
batch_idx = torch.arange(batch_size).view(-1, 1, 1, 1)  # Shape [batch_size, 1, 1, 1]
channel_idx = torch.arange(nC).view(1, -1, 1, 1)        # Shape [1, nC, 1, 1]

# Expand B_h and B_w to match batch and channel dimensions
B_h = B_h.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, nC, nb_rays, N]
B_w = B_w.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, nC, nb_rays, N]

# Use advanced indexing to gather the values from A
output = A[batch_idx, channel_idx, B_h, B_w]  # [batch_size, nC, nb_rays, N]

# Sum over the nb_rays dimension
output = output.sum(dim=2)  # Final shape: [batch_size, nC, N]

print(B[1, :, 2, :])
print(A)
print(output.shape)  # Should print: torch.Size([batch_size, nC, N])
print(output)