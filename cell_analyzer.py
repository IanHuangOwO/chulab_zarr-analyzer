import time
import os
import argparse
import numpy as np
import pyclesperanto_prototype as cle
import dask.array as da

from tqdm import tqdm
from dask.diagnostics import ProgressBar

from analyzer_count_tools import numba_unique_cell
from analyzer_report_tools import create_cell_report

# Disable OpenCL compiler logs
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"
os.environ["PYOPENCL_NO_CACHE"] = "1"
cle.select_device('GPU')

'''
python              f:\Lab\analyzer\cell_analyzer.py 
mask_path           F:\Lab\others\YA_HAN\neun_mask_ome.zarr\0 
annotation_path     F:\Lab\others\YA_HAN\annotation_ome.zarr\0 
temp_path           F:\Lab\others\YA_HAN\neun_mask_process_0.zarr
output_path         F:\Lab\others\YA_HAN\neun_output
--hemasphere_path   F:\Lab\others\YA_HAN\hemasphere_ome.zarr\0 
--chunk-size        128 128 128
'''

def check_and_load_zarr(path, component=None, chunk_size=None):
    """ 
    Check if a Zarr component exists inside the path; if yes, load it.
    If the component does not exist, try loading the full path as a Zarr store.

    Parameters:
        path (str): Directory where the Zarr store is located.
        component (str, optional): Zarr component name. Defaults to None.
        chunk_size (tuple, optional): Chunk size for loading data. Defaults to None (auto-chunking).

    Returns:
        dask.array.Array or None: Loaded Dask array if found, otherwise None.
    """
    full_path = os.path.join(path, component) if component else path

    if os.path.exists(full_path):
        print(f"✅ Found: {full_path}! Loading data...")
        
        # Load Zarr dataset with specified chunk size or auto-chunks
        return da.from_zarr(full_path, chunks=chunk_size) if chunk_size else da.from_zarr(full_path)
    
    return None

def process_filter_chunk(block, filter_size, filter_sigma):
    gpu_mask = cle.push(block.astype(np.float32))
    gpu_mask = cle.median_box(gpu_mask, radius_x=filter_size, radius_y=filter_size, radius_z=filter_size)
    gpu_mask = cle.gaussian_blur(gpu_mask, sigma_x=filter_sigma, sigma_y=filter_sigma, sigma_z=filter_sigma)
    block = cle.pull(gpu_mask).astype(block.dtype)
    
    # Determine max value based on dtype
    max_val = np.iinfo(block.dtype).max if np.issubdtype(block.dtype, np.integer) else np.finfo(block.dtype).max
    
    # Apply threshold condition
    block[block > 0.5] = max_val
    
    return block

def process_local_maxima_chunk(block):
    gpu_image = cle.push(block.astype(np.float32))
    gpu_blurred = cle.gaussian_blur(gpu_image, sigma_x=1, sigma_y=1, sigma_z=1)
    gpu_maxima = cle.detect_maxima_box(gpu_blurred, radius_x= 3, radius_y= 3, radius_z= 3)
    local_maxima = cle.pull(gpu_maxima).astype(np.uint16)
    
    local_maxima[(local_maxima > 0) & (block > 0)] = 1
    
    return local_maxima

def process_calculation_chunk(anno_chunk, hema_chunk, mask_chunk):
    """Extract unique nonzero values and their counts per block."""
    return dict(numba_unique_cell(anno_chunk, hema_chunk, mask_chunk)) # Compute unique values and counts

def process_analysis_report(full_brain_signal, left_brain_signal, right_brain_signal, voxel, output_name, output_path, structure_path='./structures.csv', target_id=None):
    """Generate Excel reports from signal dictionaries."""
    os.makedirs(output_path, exist_ok=True)
    create_cell_report(full_brain_signal, np.prod(voxel), os.path.join(output_path, f'{output_name}_full_brain_report.xlsx'), structure_path, target_id)
    create_cell_report(left_brain_signal, np.prod(voxel), os.path.join(output_path, f'{output_name}_left_brain_report.xlsx'), structure_path, target_id)
    create_cell_report(right_brain_signal, np.prod(voxel), os.path.join(output_path, f'{output_name}_right_brain_report.xlsx'), structure_path, target_id)


def main():
    parser = argparse.ArgumentParser(description="Apply median filter to a Zarr file using Dask with map_overlap.")
    parser.add_argument("mask_path", type=str, help="Zarr path to cell mask to be filtered.")
    parser.add_argument("annotation_path", type=str, help="Zarr path to annotation to be registered to.")
    parser.add_argument("temp_path", type=str, help="Temporary Zarr path for storing data.")
    parser.add_argument("output_path", type=str, help="Output path for the final report")
    parser.add_argument("--hemasphere_path", type=str, default=None, 
                        help="Zarr path to hemisphere segmentation.")
    parser.add_argument("--voxel", type=float, nargs='+', default=(0.004, 0.00182, 0.00182),
                        help="For final volume calculation. (default: 0.004, 0.00182, 0.00182)")
    parser.add_argument("--chunk-size", type=int, nargs='+', default=None,
                        help="Optional: Override chunk size for Dask processing (space-separated)")
    parser.add_argument("--filter-size", type=int, default=3,
                        help="Size of the median filter kernel (default: 3)")
    parser.add_argument("--filter-sigma", type=float, default=0.3,
                        help="Sigma of the gaussian filter (default: 0.3)")

    args = parser.parse_args()
    chunk_size = tuple(args.chunk_size) if args.chunk_size else None
    
    start_time = time.time()
    # Load Zarr arrays
    mask_data = check_and_load_zarr(args.mask_path, chunk_size=chunk_size)
    anno_data = check_and_load_zarr(args.annotation_path, chunk_size=chunk_size)
    hema_data = check_and_load_zarr(args.hemasphere_path, chunk_size=chunk_size) if args.hemasphere_path else None

    print(f"Mask shape: {mask_data.shape}")
    print(f"Annotation shape: {anno_data.shape}")
    
    # **Step 1: Apply Filtering (Skip if Exists)**
    filtered_data = check_and_load_zarr(args.temp_path, "filtered_mask", chunk_size=chunk_size)
    if filtered_data is None:
        with ProgressBar():
            print("🔄 Applying filtering...")
            filtered_data = da.map_overlap(
                process_filter_chunk,
                mask_data,
                depth=16,
                boundary='reflect',
                filter_size=args.filter_size,
                filter_sigma=args.filter_sigma,
            )
            filtered_data.to_zarr(os.path.join(args.temp_path, "filtered_mask"), overwrite=True)
            filtered_data = da.from_zarr(os.path.join(args.temp_path, "filtered_mask"))

    # **Step 2: Compute Local Maxima (Skip if Exists)**
    maxima_data = check_and_load_zarr(args.temp_path, "maxima_mask", chunk_size=chunk_size)
    if maxima_data is None:
        with ProgressBar():
            print("🔄 Finding local maxima...")
            maxima_data = da.map_overlap(
                process_local_maxima_chunk,
                filtered_data,
                depth=16,
                dtype=np.uint16,
                boundary='reflect',
            )
            maxima_data.to_zarr(os.path.join(args.temp_path, "maxima_mask"), overwrite=True)
            maxima_data = da.from_zarr(os.path.join(args.temp_path, "maxima_mask"))
                
    # **Step 3: Process Unique Values and Counts**
    full_brain_signal_dict = {}
    left_brain_signal_dict = {}
    right_brain_signal_dict = {}
    z_per_process = 128
    img_dimension = mask_data.shape
    
    print("🔄 Processing unique values and counts...")
    for i in tqdm(range(0, img_dimension[0], z_per_process)):
        start_i, end_i = i, min(i + z_per_process, img_dimension[0])
        
        if hema_data is None:
            anno_chunk, maxima_chunk,  = da.compute(
                anno_data[start_i:end_i],
                maxima_data[start_i:end_i],
            )
            hema_chunk = np.zeros_like(anno_chunk)
            
        else:
            anno_chunk, hema_chunk, maxima_chunk = da.compute(
                anno_data[start_i:end_i],
                hema_data[start_i:end_i],
                maxima_data[start_i:end_i],
            )
        
        result = process_calculation_chunk(anno_chunk, hema_chunk, maxima_chunk)
        
        for value, nums in result.items():
            if value not in full_brain_signal_dict:
                full_brain_signal_dict[value] = nums[:2]
            else:
                full_brain_signal_dict[value] += nums[:2]

            if value not in left_brain_signal_dict:
                left_brain_signal_dict[value] = nums[2:4]
            else:
                left_brain_signal_dict[value] += nums[2:4]

            if value not in right_brain_signal_dict:
                right_brain_signal_dict[value] = nums[4:6]
            else:
                right_brain_signal_dict[value] += nums[4:6]

    # **Step 4: Save Results as a CSV Report**
    print("📄 Generating final report...")
    process_analysis_report(
        full_brain_signal=full_brain_signal_dict,
        left_brain_signal=left_brain_signal_dict,
        right_brain_signal=right_brain_signal_dict,
        voxel=tuple(args.voxel),
        output_name='cell',
        output_path=args.output_path,
    )
    
    end_time = time.time()
    print(f"✅ Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()