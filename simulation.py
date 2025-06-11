# in this file, we read the data from the path and return as we are acquiring the image from the camera

import cv2
import numpy as np
import os

# Default path (will be overridden by SharedConfig)
PATH = './sample_inputs'
#PATH = './saved_data/PAT-066-2'

# Global shared_config reference
_shared_config = None

def set_shared_config(shared_config):
    """Set the global shared config reference"""
    global _shared_config
    _shared_config = shared_config

def get_simulation_path():
    """Get the simulation path from shared config or use default"""
    global _shared_config
    if _shared_config is not None:
        return _shared_config.simulation_path.value
    return PATH

def get_fov_id(path):
    """
    Scans a directory for .bmp files and extracts unique FOV IDs.

    Args:
        path (str): The directory path to scan.

    Returns:
        list[str]: A list of unique FOV IDs found in the directory,
                   potentially truncated or repeated based on internal logic.
    """
    # go though the bmp files in the path
    print(f"[get_fov_id] Scanning directory: {path}")
    
    if not os.path.exists(path):
        print(f"[get_fov_id] ERROR: Directory does not exist: {path}")
        return []
        
    files = os.listdir(path)
    print(f"[get_fov_id] Found {len(files)} files in directory")
    
    bmp_files = [f for f in files if f.endswith('.bmp')]
    print(f"[get_fov_id] Found {len(bmp_files)} .bmp files: {bmp_files[:10]}...")  # Show first 10
    
    fov_id = []
    for file in bmp_files:
        # split with last "_" and take whatever is before that
        fov_id.append(file.split('_')[0])
    # Get the unique FOV IDs
    unique_fov_ids = list(set(fov_id))
    print(f"[get_fov_id] Unique FOV IDs found: {unique_fov_ids}")
    
    # Calculate how many times we need to repeat the FOVs to reach 50
    upper_limit = 10
    repeat_count = (upper_limit + len(unique_fov_ids) - 1) // len(unique_fov_ids)
    
    # Repeat the FOV IDs to reach at least 50
    #if repeat_count > 1:
    #    extended_fov_ids = unique_fov_ids * repeat_count
    #    print(f"[get_fov_id] Repeating FOV IDs {repeat_count} times to reach upper limit")
    #else:
    #    extended_fov_ids = unique_fov_ids
    extended_fov_ids = unique_fov_ids
    
    print(f"[get_fov_id] Final FOV list: {extended_fov_ids}")
    return extended_fov_ids

# now given the list of fov, create a iterator to read the images
def get_image():
    """
    Generator function to simulate image acquisition by reading data from the PATH directory.

    Yields:
        str: The current FOV ID (e.g., '0', '1', ...).
        np.ndarray | None: Left half brightfield image. Shape (2800, 2800), dtype=uint8. None if file not found.
        np.ndarray | None: Right half brightfield image. Shape (2800, 2800), dtype=uint8. None if file not found.
        np.ndarray: Fluorescent image. Shape (2800, 2800, 3), dtype=uint8.
        np.ndarray | None: DPC image. Shape (2800, 2800), dtype=uint8 if loaded from bmp, None if file not found.
                           Single-channel image extracted from the first channel of the original DPC file.
    """

    # Use dynamic path from shared config
    simulation_path = get_simulation_path()
    # parse the path by '_Blue' and take the first part
    simulation_path = simulation_path.split('_Blue')[0]
    DPC_PATH = os.path.join(simulation_path,'Images')
    FL_PATH = os.path.join(simulation_path + '_Blue','Images')
    fov_id = get_fov_id(FL_PATH)

    print(f"[get_image] Creating new iterator")
    print(f"[get_image] Using FL_PATH path: {FL_PATH}")
    print(f"[get_image] fov_id list: {fov_id}")
    print(f"[get_image] Number of FOVs to process: {len(fov_id)}")

    j = 0

    for fov in fov_id:
        # yield left_half, right half, and floresence image sequentially
        current_fov_id_str = str(j)
        print(f"[get_image] Processing FOV {current_fov_id_str} from file prefix {fov}")
        # Output: FOV ID (str)
        yield current_fov_id_str

        j += 1


        left_half = None
        yield left_half


        right_half = None
        yield right_half

        if os.path.exists(os.path.join(FL_PATH, fov)):
            floresence = cv2.imread(os.path.join(FL_PATH, fov))
            if floresence.shape[0] > 2800 or floresence.shape[1] > 2800:
                floresence = crop_image_large(floresence)
            print(f"[get_image] fluorescent image found for {fov} of shape {floresence.shape}")

        else:
            print(f"[get_image] ERROR: No fluorescent image found for {fov}")
            # Create a dummy fluorescent image if not found
            floresence = np.zeros((2800, 2800, 3), dtype=np.uint8)
        # Output: floresence (ndarray, (2800, 2800, 3), uint8)
        yield floresence

        # now try to load DPC
        if os.path.exists(os.path.join(DPC_PATH, fov)):
            dpc_raw = cv2.imread(os.path.join(DPC_PATH, fov))
            # Input: dpc_raw (ndarray, (H, W, 3), uint8) - Size might vary initially
            # Extract the first channel
            dpc = dpc_raw[:,:,0]
            if dpc.shape[0] > 2800 or dpc.shape[1] > 2800:
                dpc = crop_image_large(dpc)
            print(f"[get_image] dpc image found for {fov} of shape {dpc.shape}")

        else:
            dpc = None
            #print(f"[get_image] No DPC image found for {fov}, path {os.path.join(simulation_path, fov + '_dpc.bmp')}")

        # Output: dpc (ndarray, (2800, 2800), uint8) - Single channel or None
        yield dpc

    print(f"[get_image] Iterator exhausted - finished processing all {j} FOVs")

# crop the image from 3000x3000 to 2800x2800
def crop_image(image):
    """
    Crops an input image from 3000x3000 to 2800x2800 by removing a 100-pixel border.

    Args:
        image (np.ndarray): Input image, expected shape (3000, 3000) or (3000, 3000, 3).

    Returns:
        np.ndarray: Cropped image, shape (2800, 2800) or (2800, 2800, 3) depending on input,
                    maintaining the input dtype.
    """
    parameters = {}
    parameters['crop_x0'] = 100
    parameters['crop_x1'] = 2900
    parameters['crop_y0'] = 100
    parameters['crop_y1'] = 2900

    # Input: image (ndarray, (3000, 3000) or (3000, 3000, 3), any dtype)
    cropped_image = image[parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]
    # Output: cropped_image (ndarray, (2800, 2800) or (2800, 2800, 3), same dtype as input)
    return cropped_image


# crop any image larger than 2800x2800 to 2800x2800
def crop_image_large(image):
    """
    Crops an input image larger than 2800x2800 to 2800x2800 by center cropping.
    """
    # get the shape of the image
    height, width = image.shape[:2]
    # get the center of the image
    center_x = width // 2
    center_y = height // 2
    # crop the image to 2800x2800
    cropped_image = image[center_y-1400:center_y+1400, center_x-1400:center_x+1400]
    if cropped_image.shape[0] != 2800 or cropped_image.shape[1] != 2800:
        print(f"[crop_image_large] ERROR: Cropped image shape is not 2800x2800: {cropped_image.shape}")
        return image
    return cropped_image