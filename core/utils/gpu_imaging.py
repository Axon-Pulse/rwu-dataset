# File: core/utils/gpu_imaging.py
import torch
from torchkbnufft import KbNufftAdjoint
import matplotlib.pyplot as plt

def nufft_image_from_cube_torch(data, om, grid_size=(32, 512), device='cuda'):
    """
    Compute spatial-angular images for each cell (e.g., a range-doppler cell) using batched NUFFT on the GPU.
    
    Args:
        data (torch.Tensor): Complex tensor of shape (B, 1, nvx) representing nonuniform measurements, where B = n_range * n_doppler for example.
        om (torch.Tensor): NUFFT frequency offsets with shape (B, 2, nvx).
        grid_size (tuple): Desired output image size (H, W).
        device (str): GPU device identifier (e.g., 'cuda').
    
    Returns:
        torch.Tensor: Reconstructed images of shape (B, 1) + grid_size (complex-valued).
    """
    data = data.to(device)
    om = om.to(device)
    
    Nd = grid_size
    Kd = (grid_size[0]*2, grid_size[1]*2)
    Jd = (6, 6)
    
    nufft_adj = KbNufftAdjoint(im_size=Nd, grid_size=Kd, numpoints=Jd).to(device)
    image = nufft_adj(data, om)  # (B, 1, H, W)
    # image = image.squeeze(1) # (B, H, W) #.view(n_range, n_doppler, Nd[0], Nd[1])
    return image

def visualize_image_cell(image, cell=(0,0), extent=(0,5,0,5)):
    """
    Visualize the magnitude of one cell's image.
    """
    cell_image = image[cell[0], cell[1]]
    plt.figure(figsize=(6,5))
    plt.imshow(torch.abs(cell_image).cpu().numpy(), extent=extent, origin='lower')
    plt.title(f"Reconstructed Image for Cell {cell}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.colorbar(label="Amplitude")
    plt.show()
