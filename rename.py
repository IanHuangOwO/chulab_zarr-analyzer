import os

def rename_tiff_files(folder_path, prefix="20220804_brain3-1_Lectin_488_TH_561_NeuN_Z"):
    """
    Renames all .tiff files in the given folder with a sequential naming pattern.
    
    Parameters:
        folder_path (str): Path to the folder containing TIFF files.
        prefix (str): Prefix for the renamed files (default: "image_").
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return
    
    # Get a list of all TIFF files in the directory
    tiff_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tiff')])
    
    if not tiff_files:
        print("No TIFF files found in the specified folder.")
        return
    
    # Rename files sequentially
    for idx, filename in enumerate(tiff_files):
        old_path = os.path.join(folder_path, filename)
        new_filename = f"{prefix}{str(idx).zfill(4)}.tif"  # e.g., image_1.tiff, image_2.tiff
        new_path = os.path.join(folder_path, new_filename)
        
        # Rename file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

    print("Renaming completed successfully!")

# Example usage
folder_path = r"F:\Lab\others\YA_HAN\neun_mask"  # Change this to your directory
rename_tiff_files(folder_path)