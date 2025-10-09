#!/usr/bin/env python3
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s"
)

def _get_volume_name(input_path: Path) -> str:
    """
    Derives a base name from an input file path, removing extensions.
    This logic mirrors the name generation in `converter.py`.
    """
    name = input_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return input_path.stem

def ensure_zarr(
    input_path_str: str,
    conversion_dir: Path,
    output_type: str,
    resize_shape: list[int] | None,
    resize_order: int,
    chunk_size: int,
    force_conversion: bool = False
) -> Path:
    """
    Ensures an input file is available as an OME-Zarr file, converting it if necessary.

    Args:
        input_path_str: The path to the source data (file or directory).
        conversion_dir: The directory where converted Zarr files should be stored.
        output_type: The desired output format ("OME-Zarr" or "Zarr").
        resize_shape: The target shape (Z, Y, X) for resizing during conversion.
        resize_order: The interpolation order for resizing (0=nearest, 1=linear, etc.).
        chunk_size: The chunk size to use for the Zarr output.
        force_conversion: If True, runs conversion even if a Zarr file already exists.

    Returns:
        The path to the ready-to-use OME-Zarr store.

    Raises:
        FileNotFoundError: If the input path does not exist.
        RuntimeError: If the conversion process fails.
    """
    input_path = Path(input_path_str).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path for conversion does not exist: {input_path}")

    # If the input is already a Zarr, no conversion is needed.
    if ".zarr" in input_path.suffixes:
        logging.info(f"Input is already a Zarr store, skipping conversion: {input_path}")
        return input_path

    volume_name = _get_volume_name(input_path)
    # The converter creates the final .ome.zarr or .zarr inside the --output_path
    if output_type == "OME-Zarr":
        zarr_suffix = ".ome.zarr"
    else:  # "Zarr"
        zarr_suffix = ".zarr"
    expected_zarr_path = conversion_dir / f"{volume_name}{zarr_suffix}"

    if expected_zarr_path.exists() and not force_conversion:
        logging.info(f"Found existing converted Zarr, skipping conversion: {expected_zarr_path}")
        return expected_zarr_path

    # --- Proceed with conversion ---
    logging.info(f"Starting conversion for: {input_path}")
    logging.info(f"Output will be at: {expected_zarr_path}")

    # Ensure the parent directory for the Zarr store exists
    conversion_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,  # Use the same python interpreter that is running this script
        "converter.py",
        "--input_path", str(input_path),
        "--output_path", str(conversion_dir), # converter.py will place the zarr inside this dir
        "--output_type", output_type,
        "--chunk-size", str(chunk_size),
    ]

    if resize_shape:
        command.extend(["--resize-shape", *(str(s) for s in resize_shape)])
        # resize-order is only relevant if resize-shape is also provided.
        command.extend(["--resize-order", str(resize_order)])

    # Pass n_workers to the converter script if it's provided via the main script
    if '--n-workers' in ' '.join(sys.argv):
        command.extend(["--n-workers", str(next(arg for i, arg in enumerate(sys.argv) if sys.argv[i-1] == '--n-workers'))])

    try:
        # Using sys.executable ensures we use the python from the correct environment (e.g., inside Docker)
        subprocess.run(command, check=True, text=True)
        logging.info(f"Successfully converted {input_path} to {expected_zarr_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to convert {input_path}. Error: {e}")
        # Consider logging e.stdout and e.stderr for more details
        raise RuntimeError(f"Conversion failed for {input_path}.")
    except FileNotFoundError:
        logging.error("Could not find 'converter.py'. Make sure it's in the same directory.")
        raise

    return expected_zarr_path


def main():
    """
    Main entry point to orchestrate data conversion and analysis.
    """
    parser = argparse.ArgumentParser(
        description="Main entry point for Zarr analysis pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Core Arguments ---
    parser.add_argument(
        "--analyzer_type",
        type=str,
        required=True,
        choices=["cell", "vessel"],
        help="The type of analysis to perform ('cell' or 'vessel')."
    )
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the primary mask file (e.g., cell or vessel mask).")
    parser.add_argument("--annotation_path", type=str, required=True, help="Path to the brain annotation file.")
    parser.add_argument("--hemasphere_path", type=str, help="Optional: Path to the hemisphere segmentation file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the final analysis reports.")

    # --- Conversion Control ---
    parser.add_argument("--conversion_output_path", type=str, default="./converted_data", help="Directory to store intermediate Zarr files.")
    parser.add_argument(
        "--conversion_output_type",
        type=str,
        default="OME-Zarr",
        choices=["OME-Zarr", "Zarr"],
        help="Output format for conversion. Defaults to OME-Zarr."
    )
    parser.add_argument("--resize-shape", type=int, nargs=3, metavar=("Z", "Y", "X"), help="Target shape for resizing during conversion.")
    parser.add_argument("--resize-order", type=int, default=0, help="Interpolation order for resizing (e.g., 0 for nearest, 1 for linear).")
    parser.add_argument("--chunk-size", type=int, default=128, help="Chunk size for creating Zarr files.")
    parser.add_argument("--force-conversion", action="store_true", help="Force reconversion even if Zarr files exist.")

    
    # --- Analysis Performance ---
    # These arguments are passed through to the selected analyzer script.
    # We define them here for a unified CLI and for --help documentation.
    perf_group = parser.add_argument_group('Analysis Performance Control')
    perf_group.add_argument("--z-per-slab", type=int, help="Number of Z-slices to process per slab. If not set, analyzer's default is used.")
    perf_group.add_argument("--n-workers", type=int, help="Number of Dask worker processes. If not set, analyzer's default is used.")
    perf_group.add_argument("--memory-limit", type=str, help="Memory limit per Dask worker (e.g., '16GB'). If not set, analyzer's default is used.")
    
    args, unknown_args = parser.parse_known_args()

    logging.info("--- Step 1: Ensuring all inputs are in Zarr format ---")

    conversion_dir = Path(args.conversion_output_path)
    
    # Convert each input path and get the path to the resulting Zarr store    
    zarr_mask_path = ensure_zarr(args.mask_path, conversion_dir, args.conversion_output_type, args.resize_shape, args.resize_order, args.chunk_size, args.force_conversion)
    zarr_anno_path = ensure_zarr(args.annotation_path, conversion_dir, args.conversion_output_type, args.resize_shape, args.resize_order, args.chunk_size, args.force_conversion)
    zarr_mask_path = ensure_zarr(args.mask_path, conversion_dir, args.conversion_output_type, args.resize_shape, args.resize_order, args.chunk_size, args.force_conversion) # type: ignore
    zarr_anno_path = ensure_zarr(args.annotation_path, conversion_dir, args.conversion_output_type, args.resize_shape, args.resize_order, args.chunk_size, args.force_conversion) # type: ignore

    zarr_hema_path = None
    if args.hemasphere_path:
        zarr_hema_path = ensure_zarr(args.hemasphere_path, conversion_dir, args.conversion_output_type, args.resize_shape, args.resize_order, args.chunk_size, args.force_conversion)
        zarr_hema_path = ensure_zarr(args.hemasphere_path, conversion_dir, args.conversion_output_type, args.resize_shape, args.resize_order, args.chunk_size, args.force_conversion) # type: ignore

    logging.info("--- All inputs are ready in Zarr format. ---")
    logging.info(f"Mask Zarr: {zarr_mask_path}")
    logging.info(f"Annotation Zarr: {zarr_anno_path}")
    if zarr_hema_path:
        logging.info(f"Hemisphere Zarr: {zarr_hema_path}")

    logging.info(f"--- Step 2: Running {args.analyzer_type} analyzer ---")

    analyzer_script = f"{args.analyzer_type}_analyzer.py"
    
    # Ensure the output directory for the analysis exists
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        analyzer_script,
        "--mask_path", str(zarr_mask_path),
        "--annotation_path", str(zarr_anno_path),
        "--output_path", str(args.output_path),
    ]

    if zarr_hema_path:
        command.extend(["--hemasphere_path", str(zarr_hema_path)])
        
    # Add performance arguments if they were provided
    if args.z_per_slab is not None:
        command.extend(["--z-per-slab", str(args.z_per_slab)])
    if args.n_workers is not None:
        command.extend(["--n-workers", str(args.n_workers)])
    if args.memory_limit is not None:
        command.extend(["--memory-limit", args.memory_limit])

    # Pass through any other arguments to the analyzer script
    if unknown_args:
        command.extend(unknown_args)

    logging.info(f"Executing command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, text=True)
        logging.info(f"Successfully ran {analyzer_script}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Analyzer script {analyzer_script} failed. Error: {e}")
        raise RuntimeError(f"Analysis failed for {analyzer_script}.")
    except FileNotFoundError:
        logging.error(f"Could not find '{analyzer_script}'. Make sure it's in the same directory.")
        raise

if __name__ == "__main__":
    main()
