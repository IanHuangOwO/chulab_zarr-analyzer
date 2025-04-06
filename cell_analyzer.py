"""
Usage:
python cell_analyzer.py 
    <mask_path>             neun_mask_ome.zarr
    <annotation_path>       annotation_ome.zarr
    <temp_path>             neun_mask_process.zarr
    <output_path>           neun_output
    --hemasphere_path       hemasphere_ome.zarr
    --chunk-size            128 128 128
"""

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
    if not path:
        return None

    full_path = os.path.join(path, component) if component else path
    if os.path.exists(full_path):
        print(f"✅ Found: {full_path}! Loading data...")

        # Load Zarr dataset with specified chunk size or auto-chunks
        return da.from_zarr(full_path, chunks=chunk_size) if chunk_size else da.from_zarr(full_path)

    return None

def process_filter_chunk(block, filter_size, filter_sigma):
    """
    Applies a median box filter followed by a Gaussian blur to a 3D image block on the GPU.

    Parameters:
        block (np.ndarray): The 3D image chunk to process.
        filter_size (int): The radius for the median box filter in all dimensions.
        filter_sigma (float): The sigma value for the Gaussian blur.

    Returns:
        np.ndarray: The filtered and thresholded image block.
    """
    gpu_mask = cle.push(block.astype(np.float32))
    gpu_mask = cle.median_box(
        source=gpu_mask,
        radius_x=filter_size,
        radius_y=filter_size,
        radius_z=filter_size
    )
    gpu_mask = cle.gaussian_blur(
        source=gpu_mask,
        sigma_x=filter_sigma,
        sigma_y=filter_sigma,
        sigma_z=filter_sigma
    )
    block = cle.pull(gpu_mask).astype(block.dtype)

    # Determine max value based on dtype
    if np.issubdtype(block.dtype, np.integer):
        max_val = np.iinfo(block.dtype).max
    else:
        max_val = np.finfo(block.dtype).max

    # Apply threshold condition
    block[block > 0.5] = max_val

    return block

def process_local_maxima_chunk(block):
    """
    Detects local maxima in a 3D image block using Gaussian blur and a box-based maxima filter.

    The function performs the following steps:
    1. Converts the input block to float32 and pushes it to the GPU.
    2. Applies a Gaussian blur to reduce noise.
    3. Detects local maxima using a box filter with a fixed radius.
    4. Pulls the result back to CPU and converts it to uint16.
    5. Retains only the maxima that correspond to positive values in the original block.

    Parameters:
        block (np.ndarray): The 3D image block to process.

    Returns:
        np.ndarray: A binary 3D mask (dtype=uint16) where detected local maxima are set to 1.
    """
    gpu_image = cle.push(block.astype(np.float32))
    gpu_blurred = cle.gaussian_blur(gpu_image, sigma_x=1, sigma_y=1, sigma_z=1)
    gpu_maxima = cle.detect_maxima_box(gpu_blurred, radius_x= 3, radius_y= 3, radius_z= 3)
    local_maxima = cle.pull(gpu_maxima).astype(np.uint16)

    local_maxima[(local_maxima > 0) & (block > 0)] = 1

    return local_maxima

def process_calculation_chunk(anno_chunk, hema_chunk, mask_chunk):
    """Extract unique nonzero values and their counts per block."""
    return dict(numba_unique_cell(anno_chunk, hema_chunk, mask_chunk))

def process_analysis_report(region_signals, voxel, output_name, output_path):
    """
    Generate Excel reports from signal dictionaries for multiple brain regions.

    Parameters:
        region_signals (dict): Keys are region names, values are signal dictionaries.
        voxel (tuple): Voxel size as a 3D tuple (x, y, z).
        output_name (str): Base name for the output files.
        output_path (str): Directory to save the reports.
        structure_path (str): Path to the structure CSV file.
        target_id (int, optional): ID for filtering specific brain structures.
    """
    structure_path='./structures.csv'
    target_id=None

    os.makedirs(output_path, exist_ok=True)
    voxel_volume = np.prod(voxel)

    for region_name, signal in region_signals.items():
        output_file = os.path.join(output_path, f'{output_name}_{region_name}_report.xlsx')
        create_cell_report(signal, voxel_volume, output_file, structure_path, target_id)

def main():
    """
    Main function to process a 3D cell mask and generate analysis reports.

    This function performs the following steps:
    1. Parses command-line arguments for input/output Zarr paths and processing parameters.
    2. Loads the input Zarr datasets for mask, annotation, and optionally hemisphere segmentation.
    3. Applies a median box filter followed by a Gaussian blur using GPU acceleration.
    4. Detects local maxima across the filtered image volume.
    5. Computes signal counts and distributions across full, left, and right brain hemispheres.
    6. Generates Excel reports for each region using the computed signals.

    Command-line arguments:
        mask_path (str): Path to the Zarr file containing the input cell mask.
        annotation_path (str): Path to the annotation Zarr file.
        temp_path (str): Temporary Zarr path to store intermediate filtered and maxima data.
        output_path (str): Directory where final Excel reports will be saved.
        --hemasphere_path (str, optional): Path to hemisphere segmentation Zarr file.
        --voxel (float, optional): Voxel size for volume computation.
        --chunk-size (int, optional): Override default chunk size for Dask processing.
        --filter-size (int, optional): Size of the median filter kernel.
        --filter-sigma (float, optional): Sigma value for the Gaussian blur.
    """
    parser = argparse.ArgumentParser(
        description="Apply median filter to a Zarr file using Dask with map_overlap."
    )
    parser.add_argument("mask_path", type=str, help="Zarr path to cell mask to be filtered.")
    parser.add_argument("annotation_path", type=str, help="Zarr path to annotation")
    parser.add_argument("temp_path", type=str, help="Zarr path for storing temporary data.")
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
    hema_data = check_and_load_zarr(args.hemasphere_path, chunk_size=chunk_size)

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
    full_brain_signal = {}
    left_brain_signal = {}
    right_brain_signal = {}
    z_per_process = 16
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
            if value not in full_brain_signal:
                full_brain_signal[value] = nums[:2]
            else:
                full_brain_signal[value] += nums[:2]

            if value not in left_brain_signal:
                left_brain_signal[value] = nums[2:4]
            else:
                left_brain_signal[value] += nums[2:4]

            if value not in right_brain_signal:
                right_brain_signal[value] = nums[4:6]
            else:
                right_brain_signal[value] += nums[4:6]

    # **Step 4: Save Results as a CSV Report**
    print("📄 Generating final report...")
    region_signals = {
        'full_brain': full_brain_signal,
        'left_brain': left_brain_signal,
        'right_brain': right_brain_signal
    }

    process_analysis_report(region_signals, tuple(args.voxel), 'cell', args.output_path)

    end_time = time.time()
    print(f"✅ Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
