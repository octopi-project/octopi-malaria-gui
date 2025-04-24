# in this file, we read the data from the path and return as we are acquiring the image from the camera

import cv2
import numpy as np
import os

PATH = './sample_inputs'

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

    files = os.listdir(path)
    fov_id = []
    for file in files:
        if file.endswith('.bmp'):
            # split with last "_" and take whatever is before that
            fov_id.append(file.split('_')[0])
    # Get the unique FOV IDs
    unique_fov_ids = list(set(fov_id))
    
    # Calculate how many times we need to repeat the FOVs to reach 50
    upper_limit = 1000
    #repeat_count = (upper_limit + len(unique_fov_ids) - 1) // len(unique_fov_ids)
    
    # Repeat the FOV IDs to reach at least 50
    #if repeat_count > 1:
    #    extended_fov_ids = unique_fov_ids * repeat_count
    
    if len(unique_fov_ids) > upper_limit:
        unique_fov_ids = unique_fov_ids[:upper_limit]
    
    return unique_fov_ids

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

    fov_id = get_fov_id(PATH)

    print(f"fov_id: {fov_id}")

    j = 0

    for fov in fov_id:
        # yield left_half, right half, and floresence image sequentially
        current_fov_id_str = str(j)
        # Output: FOV ID (str)
        yield current_fov_id_str

        j += 1

        if os.path.exists(os.path.join(PATH, fov + '_left_half.bmp')):
            left_half = cv2.imread(os.path.join(PATH, fov + '_left_half.bmp'))[:,:,1]
            # Input: left_half (ndarray, (3000, 3000), uint8)
            # if the image is 3000x3000, crop it to 2800x2800
            if left_half.shape[0] == 3000 and left_half.shape[1] == 3000:
                left_half = crop_image(left_half)
                # After crop: left_half (ndarray, (2800, 2800), uint8)

        else:
            left_half = None
            # print(f"[DATA_LOG] get_image: No left_half found for {fov}")
        # Output: left_half (ndarray, (2800, 2800), uint8) or None
        yield left_half

        if os.path.exists(os.path.join(PATH, fov + '_right_half.bmp')):
            right_half = cv2.imread(os.path.join(PATH, fov + '_right_half.bmp'))[:,:,1]
            # Input: right_half (ndarray, (3000, 3000), uint8)
            # if the image is 3000x3000, crop it to 2800x2800
            if right_half.shape[0] == 3000 and right_half.shape[1] == 3000:
                right_half = crop_image(right_half)
                # After crop: right_half (ndarray, (2800, 2800), uint8)
        else:
            right_half = None

        # Output: right_half (ndarray, (2800, 2800), uint8) or None
        yield right_half

        floresence = cv2.imread(os.path.join(PATH, fov + '_fluorescent.bmp'))
        # Input: floresence (ndarray, (3000, 3000, 3), uint8)
        # if the image is 3000x3000, crop it to 2800x2800
        if floresence.shape[0] == 3000 and floresence.shape[1] == 3000:
            floresence = crop_image(floresence)
            # After crop: floresence (ndarray, (2800, 2800, 3), uint8)
        # Output: floresence (ndarray, (2800, 2800, 3), uint8)
        yield floresence

        # now try to load DPC
        if os.path.exists(os.path.join(PATH, fov + '_dpc.bmp')):
            dpc_raw = cv2.imread(os.path.join(PATH, fov + '_dpc.bmp'))
            # Input: dpc_raw (ndarray, (H, W, 3), uint8) - Size might vary initially
            # Extract the first channel
            dpc = dpc_raw[:,:,0]
            # Extracted: dpc (ndarray, (H, W), uint8) - Single channel
            # If needed, crop the image
            if dpc.shape[0] == 3000 and dpc.shape[1] == 3000:
                dpc = crop_image(dpc)
                # After crop: dpc (ndarray, (2800, 2800), uint8)
        else:
            dpc = None

        # Output: dpc (ndarray, (2800, 2800), uint8) - Single channel or None
        yield dpc

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

'''
def ui_process(input_queue: mp.Queue, output: mp.Queue):
    while True:
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "UI Process", "start")
            
            with final_lock:
                if fov_id in shared_memory_final and not shared_memory_final[fov_id]['displayed']:
                    # Placeholder for UI update
                    temp_dict = shared_memory_final[fov_id]
                    temp_dict['displayed'] = True
                    shared_memory_final[fov_id] = temp_dict

                    if shared_memory_final[fov_id]['saved']:
                        output.put(fov_id)
            
                    log_time(fov_id, "UI Process", "end")
        except Empty:
            continue
'''