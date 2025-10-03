"""
Usage:
python cell_analyzer.py \
    --mask_path             ./datas/neun_mask_ome.zarr/0 \
    --annotation_path       ./datas/annotation_ome.zarr/0 \
    --output_path           ./datas/neun_output \
    --chunk-size            128 128 128
"""

import time
import os
import argparse
import numpy as np
from scipy import ndimage as ndi
import dask.array as da

from tqdm import tqdm
from dask.diagnostics.progress import ProgressBar
from dask.distributed import Client, LocalCluster

from utils.analyzer_count_tools import numba_unique_cell
from utils.analyzer_report_tools import create_cell_report


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
        print(f"✅ Found: {full_path}")

        # Load Zarr dataset with specified chunk size or auto-chunks
        return da.from_zarr(full_path, chunks=chunk_size) if chunk_size else da.from_zarr(full_path)

    return None


def process_filter_chunk(block, filter_size, filter_sigma):
    """
    Applies a median filter followed by a Gaussian blur to a 3D image block on the CPU.

    Parameters:
        block (np.ndarray): The 3D image chunk to process.
        filter_size (int): The radius for the median filter in all dimensions.
        filter_sigma (float): The sigma value for the Gaussian blur.

    Returns:
        np.ndarray: The filtered and thresholded image block.
    """
    # 1. Binarize the input block to work with a mask.
    binary_block = (block > 0).astype(np.float32)

    # 2. Apply a median filter using scipy.ndimage.
    # Note: size = 2 * radius + 1. The original code used a radius.
    filter_diameter = 2 * filter_size + 1
    median_filtered = ndi.median_filter(
        input=binary_block,
        size=filter_diameter
    )

    # 3. Apply a Gaussian blur using scipy.ndimage.
    gaussian_filtered = ndi.gaussian_blur(
        input=median_filtered,
        sigma=filter_sigma
    )

    # 4. Apply the final threshold condition to get a binary mask.
    final_mask = (gaussian_filtered > 0.5).astype(np.uint8)
    
    return final_mask


def process_local_maxima_chunk(block):
    """
    Detects local maxima in a 3D image block using SciPy and NumPy.

    The function performs the following steps:
    1. Creates a binary copy of the input block to work with.
    2. Applies a Gaussian blur using scipy.ndimage to reduce noise.
    3. Detects local maxima by comparing the blurred block to its maximum-filtered version.
    4. Retains only the maxima that correspond to positive values in the original block.

    Parameters:
        block (np.ndarray): The 3D image block to process.

    Returns:
        np.ndarray: A binary 3D mask (dtype=uint8) where detected local maxima are set to 1.
    """
    # Create a binary version of the block to match the original function's logic.
    # This also prevents modifying the input array in-place.
    binary_block = (block > 0).astype(np.uint8)

    # Apply a Gaussian blur. SciPy needs a float input for this.
    blurred_block = ndi.gaussian_filter(binary_block.astype(np.float32), sigma=(1, 1, 1))

    # Detect local maxima using a maximum filter.
    # A pixel is a local maximum if it equals the maximum value in its neighborhood.
    # Note: size = 2 * radius + 1. The original code used radius (x=5, y=5, z=3).
    # We assume a (Z, Y, X) axis order for the size.
    footprint_size = (7, 11, 11) # For radius_z=3, radius_y=5, radius_x=5
    maxima_mask = (blurred_block == ndi.maximum_filter(blurred_block, size=footprint_size))

    # Ensure the final maxima are located within the original foreground region.
    final_maxima = maxima_mask & (binary_block > 0)
    
    # Return the binary mask, converting the boolean to uint8.
    return final_maxima.astype(np.uint8)

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
    structure_path='./utils/structures.csv'
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
    parser.add_argument("--mask_path", type=str, required=True, help="Zarr path to cell mask to be filtered.")
    parser.add_argument("--annotation_path", type=str, required=True, help="Zarr path to annotation")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for temporary zarr and final report")
    parser.add_argument("--hemasphere_path", type=str, default=None, 
                        help="Zarr path to hemisphere segmentation.")
    parser.add_argument("--voxel", type=float, nargs='+', default=(0.004, 0.00182, 0.00182),
                        help="For final volume calculation. (default: 0.004, 0.00182, 0.00182)")
    parser.add_argument("--chunk-size", type=int, nargs='+', default=None,
                        help="Optional: Override chunk size for Dask processing (space-separated)")
    parser.add_argument("--filter-size", type=int, default=3,
                        help="Size of the median filter kernel (default: 3)")
    parser.add_argument("--filter-sigma", type=float, default=2,
                        help="Sigma of the gaussian filter (default: 2)")

    args = parser.parse_args()
    chunk_size = tuple(args.chunk_size) if args.chunk_size else None
    cluster = LocalCluster(
        n_workers=8,               # Number of worker processes to start
        threads_per_worker=2,      # Number of threads per worker
        memory_limit='8GB'         # The memory limit for each worker
    )

    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")

    start_time = time.time()
    # Load Zarr arrays
    mask_data = check_and_load_zarr(args.mask_path, chunk_size=chunk_size)
    anno_data = check_and_load_zarr(args.annotation_path, chunk_size=chunk_size)
    hema_data = check_and_load_zarr(args.hemasphere_path, chunk_size=chunk_size)

    print(f"Mask shape: {mask_data.shape}") # type: ignore
    print(f"Annotation shape: {anno_data.shape}") # type: ignore

    # # **Step 1: Apply Filtering (Skip if Exists)**
    # filtered_data = check_and_load_zarr(args.output_path, "filtered_mask.zarr", chunk_size=chunk_size)
    # if filtered_data is None:
    #     with ProgressBar():
    #         print("🔄 Applying filtering...")
    #         filtered_data = da.map_blocks(
    #             process_filter_chunk,
    #             mask_data,
    #             dtype=np.uint8,
    #             filter_size=args.filter_size,
    #             filter_sigma=args.filter_sigma,
    #         )
    #         filtered_data.to_zarr(os.path.join(args.output_path, "filtered_mask.zarr"), overwrite=True)
    #         filtered_data = da.from_zarr(os.path.join(args.output_path, "filtered_mask.zarr"))

    # **Step 2: Compute Local Maxima (Skip if Exists)**
    maxima_data = check_and_load_zarr(args.output_path, "maxima_mask.zarr", chunk_size=chunk_size)
    if maxima_data is None:
        with ProgressBar():
            print("🔄 Finding local maxima...")
            maxima_data = da.map_overlap(
                process_local_maxima_chunk,
                mask_data,
                trim=True,
                depth=8,
                dtype=np.uint8,
            )
            maxima_data.to_zarr(os.path.join(args.output_path, "maxima_mask.zarr"), overwrite=True)
            maxima_data = da.from_zarr(os.path.join(args.output_path, "maxima_mask.zarr"))

    # **Step 3: Process Unique Values and Counts**
    full_brain_signal = {}
    left_brain_signal = {}
    right_brain_signal = {}
    z_per_process = 16
    img_dimension = mask_data.shape # type: ignore

    print("🔄 Processing unique values and counts...")
    for i in tqdm(range(0, img_dimension[0], z_per_process)):
        start_i, end_i = i, min(i + z_per_process, img_dimension[0])

        if hema_data is None:
            anno_chunk, maxima_chunk,  = da.compute(
                anno_data[start_i:end_i], # type: ignore
                maxima_data[start_i:end_i],
            )
            hema_chunk = np.zeros_like(anno_chunk)

        else:
            anno_chunk, hema_chunk, maxima_chunk = da.compute(
                anno_data[start_i:end_i], # type: ignore
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
