# in this file, we read the data from the path and return as we are acquiring the image from the camera

import cv2
import numpy as np
import os

PATH = './sample_inputs'

def get_fov_id(path):
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

    fov_id = get_fov_id(PATH)

    print(f"fov_id: {fov_id}")

    j = 0

    for fov in fov_id:
        # yield left_half, right half, and floresence image sequentially
        yield str(j)

        j += 1

        if os.path.exists(os.path.join(PATH, fov + '_left_half.bmp')):
            left_half = cv2.imread(os.path.join(PATH, fov + '_left_half.bmp'))[:,:,1]
            # if the image is 3000x3000, crop it to 2800x2800
            if left_half.shape[0] == 3000 and left_half.shape[1] == 3000:
                left_half = crop_image(left_half)

        else:
            left_half = None
        yield left_half

        if os.path.exists(os.path.join(PATH, fov + '_right_half.bmp')):
            right_half = cv2.imread(os.path.join(PATH, fov + '_right_half.bmp'))[:,:,1]
            # if the image is 3000x3000, crop it to 2800x2800
            if right_half.shape[0] == 3000 and right_half.shape[1] == 3000:
                right_half = crop_image(right_half)
                print(f"right_half shape: {right_half.shape}")
        else:
            right_half = None

        yield right_half

        floresence = cv2.imread(os.path.join(PATH, fov + '_fluorescent.bmp'))
        # if the image is 3000x3000, crop it to 2800x2800
        if floresence.shape[0] == 3000 and floresence.shape[1] == 3000:
            floresence = crop_image(floresence)
            print(f"floresence shape: {floresence.shape}")
        yield floresence

        # now try to load DPC
        if os.path.exists(os.path.join(PATH, fov + '_dpc.bmp')):
            dpc = cv2.imread(os.path.join(PATH, fov + '_dpc.bmp'))
        else:
            dpc = None
    
        yield dpc

# crop the image from 3000x3000 to 2800x2800
def crop_image(image):
    parameters = {}
    parameters['crop_x0'] = 100
    parameters['crop_x1'] = 2900
    parameters['crop_y0'] = 100
    parameters['crop_y1'] = 2900

    return image[parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]




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