import dask.array as da
from dask.diagnostics.progress import ProgressBar
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter, maximum_filter, generate_binary_structure, label, center_of_mass
import numpy as np
import os

# === Parameters ===
input_zarr_path = './datas/neun_ome.zarr/0'
output_zarr_path = './datas/neun_output/maxima.zarr'
chunk_size = (128, 128, 128)  # should match or divide input chunking
sigma = 1  # Gaussian blur std-dev

# === Step 1: Load Zarr into Dask array ===
dask_array = da.from_zarr(input_zarr_path)

# === Step 2: Define GPU processing function ===
def gpu_blur(block):
    block_gpu = cp.asarray(block.astype(cp.float32))
    block_gpu[block_gpu > 0] = 1
    
    # Apply gaussian filter
    blurred_gpu = gaussian_filter(block_gpu, sigma=sigma)
    
    filter_block = (blurred_gpu > 0.2) & (block_gpu > 0)
    
    return cp.asnumpy(filter_block.astype(cp.uint8))

def detect_maxima(block):
    block_gpu = cp.asarray(block.astype(cp.float32))
    block_gpu[block_gpu > 0] = 1
    
    # Apply gaussian filter
    blurred_gpu = gaussian_filter(block_gpu, sigma=2)
    # Apply maximum filter
    max_filtered = maximum_filter(blurred_gpu, size=5)
    
    # Find local maxima: equals max value and greater than threshold
    local_maxima = (blurred_gpu == max_filtered) & (block_gpu > 0)
    
    # Step 2: Label connected maxima blobs
    structure = generate_binary_structure(rank=3, connectivity=3)
    labeled, num_features = label(local_maxima, structure=structure)

    # Step 3: Find center of each blob (substitute for NMS)
    label_indices = cp.arange(1, num_features + 1) 
    coords = cp.array(center_of_mass(block_gpu, labeled, label_indices))
    
    # Step 4: Keep only one peak per neighborhood (you can add distance-based clustering if needed)
    final_maxima = np.zeros_like(block, dtype=np.uint8)
    for z, y, x in coords:
        final_maxima[int(z), int(y), int(x)] = 1

    return final_maxima

# === Step 3: Save to Zarr ===
if os.path.exists(output_zarr_path):
    import shutil
    shutil.rmtree(output_zarr_path)

# === Step 4: Apply GPU function blockwise ===
with ProgressBar():
    filtered_dask = da.map_blocks(
        detect_maxima,
        dask_array,
        dtype=np.uint8
    )
    
    filtered_dask.to_zarr(output_zarr_path, overwrite=True)

print(f"Filtered output saved to: {output_zarr_path}")