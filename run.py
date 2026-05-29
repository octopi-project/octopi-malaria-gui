#import multiprocessing as mp
#from multiprocessing import Lock

import torch.multiprocessing as mp
import numpy as np
from queue import Empty
import time

from simulation import get_image
import threading
from log import report

from utils import SharedConfig, numpy2png

# Import the function to set shared config in simulation
from simulation import set_shared_config

try:
    from google.cloud import storage
    from google.oauth2 import service_account
    print("Google Cloud Storage SDK is available")
    gcloud_available = True
except:
    gcloud_available = False
    pass

import os

# Existing shared memory managers
manager = mp.Manager()
shared_memory_acquisition = manager.dict()
shared_memory_dpc = manager.dict()
shared_memory_segmentation = manager.dict()
shared_memory_fluorescent = manager.dict()
shared_memory_classification = manager.dict()
shared_memory_final = manager.dict()

# for patient id queue, used by cloud server
shared_memory_patient_queue = manager.list()
patient_queue_lock = manager.Lock()

# New shared memory for timing information
shared_memory_timing = manager.dict()
shared_memory_timing['START'] = None
shared_memory_timing['END'] = None

# Existing locks
dpc_lock = manager.Lock()
segmentation_lock = manager.Lock()
fluorescent_lock = manager.Lock()
classification_lock = manager.Lock()
final_lock = manager.Lock()

# time lock
timing_lock = manager.Lock()
timeout = 0.1
shared_config = SharedConfig()
shared_config.set_path('data')

INIT_FOCUS_RANGE_START_MM = 6.3
INIT_FOCUS_RANGE_END_MM = 6.5
SCAN_FOCUS_SEARCH_RANGE_MM = 0.1

# try to load the INIT_FOCUS_RANGE from a txt
try:
    with open('config/init_focus_range.txt', 'r') as f:
        INIT_FOCUS_RANGE_START_MM, INIT_FOCUS_RANGE_END_MM, SCAN_FOCUS_SEARCH_RANGE_MM = map(float, f.readline().split())
except:
    pass

print(f"INIT_FOCUS_RANGE_START_MM: {INIT_FOCUS_RANGE_START_MM:.3f}, INIT_FOCUS_RANGE_END_MM: {INIT_FOCUS_RANGE_END_MM:.3f}, SCAN_FOCUS_SEARCH_RANGE_MM: {SCAN_FOCUS_SEARCH_RANGE_MM:.3f}")

import cv2

def log_time(fov_id: str, process_name: str, event: str):
    with timing_lock:
        #main_logger.info(f"Logging time for FOV {fov_id} in {process_name} at {event}")

        if shared_memory_timing['START'] is None:
            shared_memory_timing['START'] = time.time()
        shared_memory_timing['END'] = time.time()

        if fov_id not in shared_memory_timing:
            shared_memory_timing[fov_id] = {}
        if process_name not in shared_memory_timing[fov_id]:
            shared_memory_timing[fov_id][process_name] = {}

        temp_dict = shared_memory_timing[fov_id]
        temp_process_dict = temp_dict.get(process_name, {})
        temp_process_dict[event] = time.time()
        temp_dict[process_name] = temp_process_dict
        shared_memory_timing[fov_id] = temp_dict

from simulation import crop_image
from control.utils import generate_scan_grid,interpolate_focus

def image_acquisition_simulation(dpc_queue: mp.Queue, fluorescent_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    """
    Simulates the image acquisition process using data from simulation.py.

    Reads FOV ID, left/right halves, fluorescent, and DPC images from the get_image generator.
    Processes and stores acquired images in shared memory and puts FOV IDs into appropriate queues.

    Args:
        dpc_queue (mp.Queue): Queue to send FOV IDs for DPC processing (if DPC is not pre-loaded).
        fluorescent_queue (mp.Queue): Queue to send FOV IDs for fluorescent processing.
        shutdown_event (mp.Event): Event to signal process termination.
        start_event (mp.Event): Event to signal the start of processing.

    Shared Memory Input:
        None

    Shared Memory Output:
        shared_memory_acquisition[fov_id] (dict): Contains 'left_half' (ndarray, (2800, 2800), uint8),
                                                   'right_half' (ndarray, (2800, 2800), uint8),
                                                   'fluorescent' (ndarray, (2800, 2800, 3), uint8).
        shared_memory_dpc[fov_id] (dict): Optional. Contains 'dpc_image' (ndarray, (2800, 2800), float16)
                                             if DPC is pre-loaded in simulation data.

    Queue Output:
        dpc_queue: Puts fov_id (str) if DPC needs generation.
        segmentation_queue: Puts fov_id (str) if DPC was pre-loaded.
        fluorescent_queue: Puts fov_id (str).
    """

    print("Starting image acquisition simulation")

    while not shutdown_event.is_set():

        if not start_event.is_set():
            time.sleep(1)
            continue     
    
        print(f"[image_acquisition_simulation] Start event detected, creating new iterator")
        print(f"[image_acquisition_simulation] Current simulation_path: {shared_config.simulation_path.value}")
        image_iterator = get_image()
        print("Image iterator created")

        logger = shared_config.setup_process_logger()
        
        try:
            shared_config.set_auto_focus_indicator(True)
            logger.info("Starting simulation image processing")
            
            # Process all images from the iterator in one go
            while True:
                try:
                    fov_id = next(image_iterator)
                    log_time(fov_id, "Image Acquisition", "start")
                    # Input: fov_id (str)

                    left_half = next(image_iterator)
                    # Input: left_half (ndarray, (2800, 2800), uint8) or None
                    right_half = next(image_iterator)
                    # Input: right_half (ndarray, (2800, 2800), uint8) or None
                    fluorescent = next(image_iterator)
                    # Input: fluorescent (ndarray, (2800, 2800, 3), uint8)
                    dpc = next(image_iterator)
                    # Input: dpc (ndarray, (H, W, 3), uint8) or None

                    with final_lock:
                        if shared_config.save_fluo_images.value:
                            save_path = shared_config.get_path()
                            
                            fluorescent_filename = f"{fov_id}_fluorescent.bmp"
                            cv2.imwrite(os.path.join(save_path, fluorescent_filename), fluorescent)

                    if dpc is None and (left_half is not None) and (right_half is not None):
                        shared_memory_acquisition[fov_id] = {
                            'left_half': left_half,
                            'right_half': right_half,
                            'fluorescent': fluorescent
                        }
                        # Stored: left_half, right_half, fluorescent in shared_memory_acquisition

                        with final_lock:
                            if shared_config.save_bf_images.value:
                                save_path = shared_config.get_path()   
                                # save the bmp
                                left_filename = f"{fov_id}_left_half.bmp"
                                right_filename = f"{fov_id}_right_half.bmp"         
                                cv2.imwrite(os.path.join(save_path, left_filename), left_half)
                                cv2.imwrite(os.path.join(save_path, right_filename), right_half)
                        
                        # Queue Output: fov_id (str) to dpc_queue
                        dpc_queue.put(fov_id)
                    
                    elif dpc is not None:
                        shared_memory_acquisition[fov_id] = {
                            'left_half': left_half,
                            'right_half': right_half,
                            'fluorescent': fluorescent
                        }
                        # Stored: left_half, right_half, fluorescent in shared_memory_acquisition
                        # convert to numpy array
                        # check the dimension of dpc
                        # DPC is now single-channel from simulation.py, no need to extract first channel
                        
                        assert dpc.shape == (2800, 2800)
                        # Expected: dpc shape (2800, 2800)
                        dpc = dpc.astype(np.float16)/255
                        # Converted dpc: (2800, 2800), float16
                        log_time(fov_id, "DPC Process", "start")
                        with dpc_lock:
                            shared_memory_dpc[fov_id] = {'dpc_image': dpc}
                            # Stored: dpc_image in shared_memory_dpc
             
                        log_time(fov_id, "DPC Process", "end")

                        # Queue Output: fov_id (str) to segmentation_queue
                        segmentation_queue.put(fov_id)

                    else:
                        logger.info(f"Something wrong with the image acquisition")
                        print(f"Something wrong with the image acquisition")
                        # signal the shutdown event
                        shutdown_event.set()
                        break

                    # Queue Output: fov_id (str) to fluorescent_queue
                    fluorescent_queue.put(fov_id)
                        
                    log_time(fov_id, "Image Acquisition", "end")

                except StopIteration:
                    logger.info("No more images to process")
                    break
                    
            logger.info("Finished processing all simulation images")
            
        except Exception as e:
            logger.error(f"Error in image acquisition simulation: {e}")
            continue

        # After processing all images, wait for start_event to be unset or shutdown
        while start_event.is_set() and not shutdown_event.is_set():
            time.sleep(1)

    print("Image acquisition simulation process finished")

from microscope import Microscope
from general_acquisition import run_general_acquisition
def image_acquisition(dpc_queue: mp.Queue, fluorescent_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    """
    Controls the microscope hardware to perform actual image acquisition.

    Handles autofocus, scanning grid, Z-map interpolation, and image capture for different channels.
    Stores acquired images in shared memory and puts FOV IDs into appropriate queues.

    Args:
        dpc_queue (mp.Queue): Queue to send FOV IDs for DPC processing.
        fluorescent_queue (mp.Queue): Queue to send FOV IDs for fluorescent processing.
        shutdown_event (mp.Event): Event to signal process termination.
        start_event (mp.Event): Event to signal the start of scanning.

    Shared Memory Input:
        Reads various settings from shared_config (e.g., nx, ny, save flags).

    Shared Memory Output:
        shared_memory_acquisition[fov_id] (dict): Contains 'left_half' (ndarray, (2800, 2800), uint8),
                                                   'right_half' (ndarray, (2800, 2800), uint8),
                                                   'fluorescent' (ndarray, (2800, 2800, 3), uint8).
        Updates shared_config live view image and coordinates.

    Queue Output:
        dpc_queue: Puts fov_id (str).
        fluorescent_queue: Puts fov_id (str).
    """
    global INIT_FOCUS_RANGE_START_MM, INIT_FOCUS_RANGE_END_MM, SCAN_FOCUS_SEARCH_RANGE_MM
  
    simulation = False
    microscope = Microscope(is_simulation=simulation)
    microscope.camera.start_streaming()
    microscope.camera.set_software_triggered_acquisition()
    microscope.camera.disable_callback()
    microscope.home_xyz()
    microscope.move_z_to((INIT_FOCUS_RANGE_START_MM + INIT_FOCUS_RANGE_END_MM) / 2)
    
    live_channel_index = -1

    done = False

    while not shutdown_event.is_set():

        # check if to loading or to scanning position
        if not start_event.is_set():
            # Apply any pending per-channel setting overrides from the UI
            if shared_config.channel_overrides_dirty.value:
                with shared_config.channel_overrides_lock:
                    overrides_snapshot = {k: dict(v) for k, v in shared_config.channel_overrides.items()}
                    shared_config.channel_overrides_dirty.value = False
                for ch_name, settings in overrides_snapshot.items():
                    try:
                        cfg = microscope.configurationManager.get_configuration_by_name(ch_name)
                        if cfg is None:
                            continue
                        if 'exposure_ms' in settings:
                            cfg.exposure_time = float(settings['exposure_ms'])
                        if 'analog_gain' in settings:
                            cfg.analog_gain = float(settings['analog_gain'])
                        if 'illumination_intensity' in settings:
                            cfg.illumination_intensity = float(settings['illumination_intensity'])
                    except Exception as ov_err:
                        print(f"Channel override apply failed for {ch_name}: {ov_err}")
                # Force the live view loop to re-issue set_channel on its next pass
                if shared_config.is_live_view_active.value:
                    live_channel_index = -1

            with shared_config.position_lock:
                if shared_config.to_scanning.value and not shared_config.to_loading.value:
                    #main_logger.info("Moving to scanning position")
                    microscope.to_scanning_position()
                    shared_config.reset_to_scanning()
                    continue
                if shared_config.to_loading.value and not shared_config.to_scanning.value:
                    #main_logger.info("Moving to loading position")
                    microscope.to_loading_position()
                    shared_config.reset_to_loading()
                    continue
            
            if shared_config.is_live_view_active.value:

                if live_channel_index != shared_config.live_channel_selected.value:
                    live_channel_index = shared_config.live_channel_selected.value
                    # get the value from manager.list
                    microscope.set_channel(shared_config.live_channels_list[live_channel_index])

                image = microscope.acquire_image()                   
                if not done:
                    # save the raw image to the disk
                    cv2.imwrite(f'raw_image.png', image) 
                    done = True

                if live_channel_index != 1: # first one is the fluorescent as defined in the config file
                    image = crop_image(image)
                else:
                    image = crop_image(image)

                shared_config.set_live_view_image(image)
            
                # update the live x and y
                shared_config.live_x.value = microscope.get_x()
                shared_config.live_y.value = microscope.get_y()
                shared_config.live_z.value = microscope.get_z()

            elif shared_config.ga_active.value:
                ga_logger = shared_config.setup_process_logger()
                try:
                    run_general_acquisition(microscope, shared_config, shutdown_event, ga_logger)
                except Exception as ga_err:
                    ga_logger.error(f"General acquisition failed: {ga_err}")
                    shared_config.ga_active.value = False
                    shared_config.ga_running.value = False
                # Force live view to re-apply its channel (GA left the microscope on its last channel)
                live_channel_index = -1

            # add a option to run the calibration of the autofocus searching range
            elif shared_config.is_auto_focus_calibration.value:
                # run the calibration
                microscope.set_channel("BF LED matrix left half")
                z_focus_init, _ = microscope.run_autofocus(step_size_mm = [0.01, 0.0015], start_z_mm = 5.5, end_z_mm = 7.0, shared_config=shared_config)
                # with this z_focus_init, we know that the autofocus searching range is 0.1 mm
                INIT_FOCUS_RANGE_START_MM = z_focus_init - 0.05
                INIT_FOCUS_RANGE_END_MM = z_focus_init + 0.05
                SCAN_FOCUS_SEARCH_RANGE_MM = 0.1
                # save the range to a txt
                with open('config/init_focus_range.txt', 'w') as f:
                    f.write(f"{INIT_FOCUS_RANGE_START_MM} {INIT_FOCUS_RANGE_END_MM} {SCAN_FOCUS_SEARCH_RANGE_MM}")

                shared_config.is_auto_focus_calibration.value = False
                print(f"Auto focus calibration done, range: {INIT_FOCUS_RANGE_START_MM:.3f} - {INIT_FOCUS_RANGE_END_MM:.3f} mm, search range: {SCAN_FOCUS_SEARCH_RANGE_MM:.3f} mm")
            
            time.sleep(1/shared_config.frame_rate.value)
            
            continue
        
        logger = shared_config.setup_process_logger()

        try:
            if gcloud_available:
                # add patient id to the queue
                patient_queue_lock.acquire()
                shared_memory_patient_queue.append(shared_config.patient_id.value)
                patient_queue_lock.release()

            logger.info("Running autofocus")
            microscope.set_channel("BF LED matrix left half")
            #z_focus_init, best_focus_init = microscope.run_autofocus(step_size_mm = [0.01, 0.0015], start_z_mm = INIT_FOCUS_RANGE_START_MM, end_z_mm = INIT_FOCUS_RANGE_END_MM)
            #logger.info(f"Initial focus: z = {z_focus_init:.3f} mm, focus measure = {best_focus_init:.3f}")
            z_focus_init = (INIT_FOCUS_RANGE_START_MM + INIT_FOCUS_RANGE_END_MM) / 2
            microscope.move_z_to(z_focus_init)
            # generate the focus map
            # scan settings
            dx_mm = 0.9
            dy_mm = 0.9
            Nx = shared_config.nx.value
            Ny = shared_config.ny.value
            # calculate the number of focus points such that for each grid, the edge cannot exceed 10 units
            Nx_focus = int(np.ceil(Nx/10)+1)
            Ny_focus = int(np.ceil(Ny/10)+1)
            offset_x_mm = microscope.get_x()
            offset_y_mm = microscope.get_y()
            offset_z_mm = microscope.get_z()

            print(f"Nx_focus: {Nx_focus}, Ny_focus: {Ny_focus}")

            # generate scan grid
            scan_grid = generate_scan_grid(dx_mm, dy_mm, Nx, Ny, offset_x_mm, offset_y_mm, S_scan=True)

            # generate focus map
            x = offset_x_mm + np.linspace(0, (Nx - 1) * dx_mm, Nx_focus)
            y = offset_y_mm + np.linspace(0, (Ny - 1) * dy_mm, Ny_focus)
            print(f"x: {x}, y: {y}")
            focus_map = []
            microscope.set_channel("BF LED matrix left half")
            for i, yi in enumerate(y):
                microscope.move_y_to(yi)
                x_range = x if i % 2 == 0 else x[::-1]
                for xi in x_range:
                    microscope.move_x_to(xi)
                    z_focus,best_focus = microscope.run_autofocus(step_size_mm = [0.01, 0.001], start_z_mm = offset_z_mm - SCAN_FOCUS_SEARCH_RANGE_MM/2, end_z_mm = offset_z_mm + SCAN_FOCUS_SEARCH_RANGE_MM/2)
                    logger.info(f"At x: {xi:.3f}, y: {yi:.3f}, z: {z_focus:.3f}, best focus: {best_focus:.3f}")
                    focus_map.append((xi, yi, z_focus))
                    offset_z_mm = z_focus

            z_map = interpolate_focus(scan_grid, focus_map)

            shared_config.set_auto_focus_indicator(True)
            time.sleep(0.5)
            logger.info("Autofocus done")

            # scan using focus map
            prev_x, prev_y = None, None
            for i, ((x, y), z) in enumerate(zip(scan_grid, z_map)):
                if x != prev_x:
                    microscope.move_x_to(x)
                    prev_x = x
                if y != prev_y:
                    microscope.move_y_to(y)
                    prev_y = y
                microscope.move_z_to(z)
                fov_id = f"{i+1}"
                log_time(fov_id, "Image Acquisition", "start")
                channels = ["BF LED matrix left half","BF LED matrix right half","Fluorescence 405 nm Ex"]

                microscope.set_channel(channels[0])
                left_half = microscope.acquire_image()
                microscope.set_channel(channels[1])
                right_half = microscope.acquire_image()
                microscope.set_channel(channels[2])
                fluorescent = microscope.acquire_image()

                shared_config.set_live_view_image(crop_image(left_half))
                
                if not simulation:
                    # go from 3k x 3k x 3 to 3k x 3k, take the middle channel
                    left_half = left_half[:,:,1]
                    right_half = right_half[:,:,1]
                    # for fluorescent (3k x 3k x 3), we reverse the channel order
                    fluorescent = fluorescent[:, :, ::-1]
                else:
                    fluorescent = np.stack([fluorescent,fluorescent,fluorescent],axis=2)

                left_half = crop_image(left_half)
                right_half = crop_image(right_half)
                fluorescent = crop_image(fluorescent)

                shared_memory_acquisition[fov_id] = {
                    'left_half': left_half,
                    'right_half': right_half,
                    'fluorescent': fluorescent
                }

                with final_lock:
                    if shared_config.save_bf_images.value:
                        save_path = shared_config.get_path()
                        if shared_config.SAVE_NPY.value:
                            left_filename = f"{fov_id}_left_half.npy"
                            right_filename = f"{fov_id}_right_half.npy"
                            np.save(os.path.join(save_path, left_filename), left_half)
                            np.save(os.path.join(save_path, right_filename), right_half)
                            
                        else:
                        # save the bmp
                            left_filename = f"{fov_id}_left_half.bmp"
                            right_filename = f"{fov_id}_right_half.bmp"         
                            cv2.imwrite(os.path.join(save_path, left_filename), left_half)
                            cv2.imwrite(os.path.join(save_path, right_filename), right_half)
                            

                with final_lock:
                    if shared_config.save_fluo_images.value:
                        save_path = shared_config.get_path()
                        if shared_config.SAVE_NPY.value:
                            fluorescent_filename = f"{fov_id}_fluorescent.npy"
                            np.save(os.path.join(save_path, fluorescent_filename), fluorescent)
                        else:
                            fluorescent_filename = f"{fov_id}_fluorescent.bmp"
                            cv2.imwrite(os.path.join(save_path, fluorescent_filename), fluorescent)
    
                dpc_queue.put(fov_id)
                fluorescent_queue.put(fov_id)

                log_time(fov_id, "Image Acquisition", "end")
        except Exception as e:
            logger.error(f"Error in image acquisition: {e}")
            continue

        while start_event.is_set() and not shutdown_event.is_set():
            # change the channel to whatever is in the config file
            microscope.set_channel(shared_config.live_channels_list[live_channel_index])
            time.sleep(1)

    microscope.close()
    print("Image acquisition process finished")

from utils import generate_dpc,save_dpc_image

def dpc_process(input_queue: mp.Queue, output_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    """
    Generates a Differential Phase Contrast (DPC) image from left and right half brightfield images.

    Args:
        input_queue (mp.Queue): Queue receiving fov_id (str) for processing (from image_acquisition).
        output_queue (mp.Queue): Queue to send fov_id (str) after DPC generation (to segmentation).
        shutdown_event (mp.Event): Event to signal process termination.
        start_event (mp.Event): Event to signal the start of processing.

    Shared Memory Input:
        shared_memory_acquisition[fov_id]: Reads 'left_half', 'right_half' (ndarrays, uint8).

    Shared Memory Output:
        shared_memory_dpc[fov_id] (dict): Contains 'dpc_image' (ndarray, (2800, 2800), float16).

    Queue Output:
        output_queue: Puts fov_id (str).
    """
    while not shutdown_event.is_set():

        start_event.wait()
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "DPC Process", "start")
            # Queue Input: fov_id (str)
            
            data = shared_memory_acquisition[fov_id]
            left_half = data['left_half'].astype(np.float16)/255
            right_half = data['right_half'].astype(np.float16)/255
            # Input from shared_mem: left_half, right_half (ndarrays, (2800, 2800), uint8) -> converted to float16
            
            dpc_image = generate_dpc(left_half, right_half,use_gpu=False) 
            # Generated: dpc_image (ndarray, (2800, 2800), float16)
            
            with dpc_lock:
                shared_memory_dpc[fov_id] = {'dpc_image': dpc_image}
                # Stored: dpc_image in shared_memory_dpc
            
            # Queue Output: fov_id (str) to segmentation_queue
            output_queue.put(fov_id)
            log_time(fov_id, "DPC Process", "end")
        except Empty:
            continue
        except Exception as e:
            logger = shared_config.setup_process_logger()
            logger.error(f"Unknown error in DPC process {e}")
            continue

    print("DPC process finished")

def segmentation_process(input_queue: mp.Queue, output_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    """
    Performs cell segmentation on the DPC image using a M2Unet model.

    Args:
        input_queue (mp.Queue): Queue receiving fov_id (str) (from dpc_process or image_acquisition_simulation).
        output_queue (mp.Queue): Queue to send fov_id (str) after segmentation (to classification).
        shutdown_event (mp.Event): Event to signal process termination.
        start_event (mp.Event): Event to signal the start of processing.

    Shared Memory Input:
        shared_memory_dpc[fov_id]: Reads 'dpc_image' (ndarray, float16).

    Shared Memory Output:
        shared_memory_segmentation[fov_id] (dict): Contains 'segmentation_map' (ndarray, (2800, 2800), uint8),
                                                     'n_cells' (int).

    Queue Output:
        output_queue: Puts fov_id (str).
    """
    from interactive_m2unet_inference import M2UnetInteractiveModel as m2u
    import torch
    from scipy.ndimage import label

    model_path = 'checkpoint/m2unet_model_flat_erode1_wdecay5_smallbatch/model_4000_11.pth'
    model = m2u(pretrained_model=model_path, use_trt=False)
    
    while not shutdown_event.is_set():
        start_event.wait()
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "Segmentation Process", "start")
            # Queue Input: fov_id (str)
            
            # Wait for DPC to finish
            while fov_id not in shared_memory_dpc:
                time.sleep(timeout)
            
            dpc_image = shared_memory_dpc[fov_id]['dpc_image']
            # Input from shared_mem: dpc_image (ndarray, (2800, 2800), float16)
            # convert dpc to np.int8
            dpc_image = (dpc_image*255).astype(np.uint8)
            # Converted: dpc_image (ndarray, (2800, 2800), uint8)
            
            result = model.predict_on_images(dpc_image)
            # Model Output: result (ndarray, (2800, 2800), float64)
            threshold = 0.5
            segmentation_mask = (255*(result > threshold)).astype(np.uint8)
            # Generated: segmentation_mask (ndarray, (2800, 2800), uint8)
            _, n_cells = label(segmentation_mask)
            # Generated: n_cells (int)
            
            with segmentation_lock:
                shared_memory_segmentation[fov_id] = {'segmentation_map': segmentation_mask, 'n_cells': n_cells}
                # Stored: segmentation_map, n_cells in shared_memory_segmentation
            
            # Queue Output: fov_id (str) to classification_queue
            output_queue.put(fov_id)
            log_time(fov_id, "Segmentation Process", "end")
        except Empty:
            continue
        except Exception as e:
            logger = shared_config.setup_process_logger()
            logger.error(f"Unknown error in segmentation process {e}")
            with segmentation_lock:
                shared_memory_segmentation[fov_id] = {'segmentation_map': np.zeros((2800, 2800)), 'n_cells': 0}
            continue

from utils import remove_background, resize_image_cp, detect_spots, prune_blobs, settings, seg_spot_filter_one_fov
MAX_SPOTS_THRESHOLD = 5000  # Maximum number of spots allowed

def fluorescent_spot_detection(input_queue: mp.Queue, output_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    """
    Detects fluorescent spots in the fluorescent image.

    Performs background removal, resizing, spot detection (DoG), pruning, and rescaling.

    Args:
        input_queue (mp.Queue): Queue receiving fov_id (str) (from image_acquisition*).
        output_queue (mp.Queue): Queue to send fov_id (str) after spot detection (to classification).
        shutdown_event (mp.Event): Event to signal process termination.
        start_event (mp.Event): Event to signal the start of processing.

    Shared Memory Input:
        shared_memory_acquisition[fov_id]: Reads 'fluorescent' (ndarray, uint8).

    Shared Memory Output:
        shared_memory_fluorescent[fov_id] (dict): Contains 'spot_indices' (ndarray, (N, 3), int64),
                                                    'abnormal_spots' (bool),
                                                    'spot_count' (int).

    Queue Output:
        output_queue: Puts fov_id (str).
    """
    
    while not shutdown_event.is_set():
        start_event.wait()
        logger = shared_config.setup_process_logger()
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "Fluorescent Spot Detection", "start")
            # Queue Input: fov_id (str)
            
            fluorescent = shared_memory_acquisition[fov_id]['fluorescent']
            # Input from shared_mem: fluorescent (ndarray, (2800, 2800, 3), uint8)

            I_fluorescence_bg_removed = remove_background(fluorescent,return_gpu_image=False)
            # Generated: I_fluorescence_bg_removed (ndarray, (2800, 2800, 3), float16)

            # Resize before spot detection
            resized_fluo = resize_image_cp(I_fluorescence_bg_removed, downsize_factor=settings['spot_detection_downsize_factor'])
            # Generated: resized_fluo (ndarray, (700, 700, 3), float64) - Note dtype change due to resize

            spot_list = detect_spots(resized_fluo, thresh=settings['spot_detection_threshold'])
            # Generated: spot_list (ndarray, (N, 3), ??) - Initial detection, dtype might vary
            
            if len(spot_list) > MAX_SPOTS_THRESHOLD:
                logger.info(f"Abnormal number of fluorescent spots detected in FOV {fov_id}: {len(spot_list)}")
                spot_list = spot_list[:MAX_SPOTS_THRESHOLD]
            
            if len(spot_list) > 0:
                spot_list = prune_blobs(spot_list)
                # After prune: spot_list (ndarray, (N_pruned, 3))

            spot_list = spot_list*settings['spot_detection_downsize_factor']
            # Rescaled: spot_list (ndarray, (N_pruned, 3), int64)
            
            with fluorescent_lock:
                    shared_memory_fluorescent[fov_id] = {
                        'spot_indices': spot_list,
                        'abnormal_spots': len(spot_list) > MAX_SPOTS_THRESHOLD,
                        'spot_count': len(spot_list)
                    }
                    # Stored: spot_indices, abnormal_spots, spot_count in shared_memory_fluorescent

            # free the memory
            del I_fluorescence_bg_removed
            
            # Queue Output: fov_id (str) to fluorescent_detection_queue
            output_queue.put(fov_id)
            log_time(fov_id, "Fluorescent Spot Detection", "end")

        except Empty:
            continue
        except Exception as e:
            logger.error(f"Unknown error in fluorescent spot detection {e}")
            with fluorescent_lock:
                shared_memory_fluorescent[fov_id] = {
                    'spot_indices': [],
                    'abnormal_spots': False,
                    'spot_count': 0
                }
            # Forward fov_id so classification doesn't stall waiting on this queue
            output_queue.put(fov_id)
            continue

    print("Fluorescent spot detection process finished")

from utils import get_spot_images_from_fov
from model import ResNet, run_model    

def classification_process(segmentation_queue: mp.Queue, fluorescent_queue: mp.Queue, save_queue: mp.Queue, ui_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    """
    Classifies detected fluorescent spots based on cropped DPC and fluorescent images using ResNet models.

    Waits for FOV IDs from both segmentation and fluorescent detection queues.
    Filters spots based on segmentation map, extracts cropped images, runs models, and calculates final scores.

    Args:
        segmentation_queue (mp.Queue): Queue receiving fov_id (str) from segmentation_process.
        fluorescent_queue (mp.Queue): Queue receiving fov_id (str) from fluorescent_spot_detection.
        save_queue (mp.Queue): Queue to send fov_id (str) for saving results.
        ui_queue (mp.Queue): Queue to send fov_id (str) for UI updates.
        shutdown_event (mp.Event): Event to signal process termination.
        start_event (mp.Event): Event to signal the start of processing.

    Shared Memory Input:
        shared_memory_segmentation[fov_id]: Reads 'segmentation_map' (ndarray, uint8).
        shared_memory_fluorescent[fov_id]: Reads 'spot_indices' (ndarray, int64).
        shared_memory_dpc[fov_id]: Reads 'dpc_image' (ndarray, float16).
        shared_memory_acquisition[fov_id]: Reads 'fluorescent' (ndarray, uint8).

    Shared Memory Output:
        shared_memory_classification[fov_id] (dict): Contains 'cropped_images' (ndarray, (M, 4, 31, 31), float16),
                                                      'scores' (ndarray, (M,), float32),
                                                      'filtered_spots' (ndarray, (M, 3), int64),
                                                      'spot_list' (ndarray, (N, 3), int64), # Original spots from fluorescent process
                                                      'filtered_spots_count' (int).
        shared_memory_final[fov_id] (dict): Initializes entry with {'saved': False, 'displayed': False}.

    Queue Output:
        save_queue: Puts fov_id (str).
        ui_queue: Puts fov_id (str).
    """

    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    segmentation_ready = set()
    fluorescent_ready = set()

    # Read model selection (set on the Start tab). Stays fixed for the life of this process;
    # changing it in the UI takes effect on the next analysis start.
    model_selection = shared_config.model_selection.value or 'O2.0'
    print(f"[classification_process] model_selection = {model_selection}")

    model1 = model2 = model_v8 = None
    if model_selection == 'v8':
        from model import load_v8_model
        model_v8 = load_v8_model('./checkpoint/v8_hardneg_single/best.pt', DEVICE)
    else:
        # O1.9 (ensemble) and O2.0 (single) both load model1; ensemble also loads model2.
        CHECKPOINT1 = './checkpoint/resnet18_en/version1/best.pt'
        model1 = ResNet('resnet18').to(device=DEVICE)
        model1.load_state_dict(torch.load(CHECKPOINT1))
        model1.eval()
        if model_selection == 'O1.9':
            CHECKPOINT2 = './checkpoint/resnet18_en/version2/best.pt'
            model2 = ResNet('resnet18').to(device=DEVICE)
            model2.load_state_dict(torch.load(CHECKPOINT2))
            model2.eval()

    while not shutdown_event.is_set():
        start_event.wait()
        logger = shared_config.setup_process_logger()
        try:
            # Check segmentation queue
            try:
                fov_id = segmentation_queue.get(timeout=timeout)
                segmentation_ready.add(fov_id)
            except Empty:
                pass

            # Check fluorescent queue
            try:
                fov_id = fluorescent_queue.get(timeout=timeout)
                fluorescent_ready.add(fov_id)
            except Empty:
                pass

            # Process FOVs that are ready in both queues
            ready_fovs = segmentation_ready.intersection(fluorescent_ready)
            for fov_id in ready_fovs:
                log_time(fov_id, "Classification Process", "start")
                # Processing: fov_id (str)
     
                segmentation_map = shared_memory_segmentation[fov_id]['segmentation_map']
                spot_list = shared_memory_fluorescent[fov_id]['spot_indices']
                # Input from shared_mem: segmentation_map (ndarray, (2800, 2800), uint8), spot_list (ndarray, (N, 3), int64)

                # save segmentation map
                save_path = shared_config.get_path()
                # save the map as bmp
                filename = os.path.join(save_path, f"{fov_id}_segmentation_map.bmp")
                cv2.imwrite(filename, segmentation_map)

                if len(spot_list) > 0:

                    filtered_spots = seg_spot_filter_one_fov(segmentation_map, spot_list)
                    # Generated: filtered_spots (ndarray, (M, 3), int64)
                    #filtered_spots = spot_list

                    if len(filtered_spots) > 0:
                        dpc_image = shared_memory_dpc[fov_id]['dpc_image']
                        fluorescence_image = shared_memory_acquisition[fov_id]['fluorescent'].astype(np.float16)/255
                        # Input from shared_mem: dpc_image (ndarray, (2800, 2800), float16), fluorescence_image (ndarray, (2800, 2800, 3), float16)
                
                        cropped_images = get_spot_images_from_fov(fluorescence_image,dpc_image,filtered_spots,r=15)
                        # Generated: cropped_images (ndarray, (M, 31, 31, 4), float16)
                        cropped_images = cropped_images.transpose(0, 3, 1, 2)
                        # Transposed: cropped_images (ndarray, (M, 4, 31, 31), float16)

                        if model_selection == 'v8':
                            from model import run_model_v8
                            scores = run_model_v8(model_v8, DEVICE, cropped_images, 1024)
                        elif model_selection == 'O1.9':
                            scores1 = run_model(model1, DEVICE, cropped_images, 1024)[:, 1]
                            scores2 = run_model(model2, DEVICE, cropped_images, 1024)[:, 1]
                            scores = np.minimum(scores1, scores2)
                        else:  # 'O2.0'
                            scores = run_model(model1, DEVICE, cropped_images, 1024)[:, 1]
                    else:
                        filtered_spots = np.array([])
                        scores = np.array([])
                        cropped_images = np.array([])
                        #spot_list = np.array([]) # Don't overwrite original spot_list here
                        # Case: No filtered spots
                else:
                    filtered_spots = np.array([])
                    scores = np.array([])
                    cropped_images = np.array([])
                    #spot_list = np.array([]) # Don't overwrite original spot_list here
                    # Case: No initial spots


                with classification_lock:
                    shared_memory_classification[fov_id] = {
                        'cropped_images': cropped_images,
                        'scores': scores,
                        'filtered_spots': filtered_spots,
                        'spot_list': spot_list, # Store original spot_list
                        'filtered_spots_count': len(filtered_spots)
                    }
                    # Stored: cropped_images, scores, filtered_spots, spot_list, filtered_spots_count in shared_memory_classification

                #del cropped_images, scores
                
                # Update shared_memory_final
                with final_lock:
                    if fov_id not in shared_memory_final:
                        shared_memory_final[fov_id] = {
                            'saved': False,
                            'displayed': False
                        }

                # Queue Output: fov_id (str) to save_queue & ui_queue
                save_queue.put(fov_id)
                ui_queue.put(fov_id)
                segmentation_ready.remove(fov_id)
                fluorescent_ready.remove(fov_id)

                log_time(fov_id, "Classification Process", "end")

        except Empty:
            continue
        except Exception as e:
            logger.error(f"Unknown error in classification process {e}")
            with classification_lock:
                shared_memory_classification[fov_id] = {
                    'cropped_images': [],
                    'scores': [],
                    'filtered_spots_count': 0
                }
            continue
    print("Classification process finished")

import os
from utils import draw_spot_bounding_boxes
def saving_process(input_queue: mp.Queue, output: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    """
    Saves processed data (images, spots, scores) to disk based on configuration.

    Generates and saves overlay images with bounding boxes.
    Updates shared memory flag and potentially queues FOV ID for cleanup.

    Args:
        input_queue (mp.Queue): Queue receiving fov_id (str) from classification_process.
        output (mp.Queue): Queue to send fov_id (str) for cleanup (if UI has displayed).
        shutdown_event (mp.Event): Event to signal process termination.
        start_event (mp.Event): Event to signal the start of processing.

    Shared Memory Input:
        shared_memory_classification[fov_id]: Reads 'cropped_images', 'scores', 'filtered_spots', 'spot_list'.
        shared_memory_acquisition[fov_id]: Reads 'fluorescent'.
        shared_memory_dpc[fov_id]: Reads 'dpc_image'.
        shared_memory_final[fov_id]: Reads 'displayed' flag.
        Reads shared_config for save paths and flags.

    Shared Memory Output:
        shared_memory_final[fov_id]: Sets 'saved' flag to True.

    Queue Output:
        output (cleanup_queue): Puts fov_id (str) if 'saved' and 'displayed' are both True.
    """
    
    while not shutdown_event.is_set():

        start_event.wait()
        try:
            fov_id = input_queue.get(timeout=timeout)
            log_time(fov_id, "Saving Process", "start")
            # Queue Input: fov_id (str)
            
            with final_lock:
                if fov_id in shared_memory_final and not shared_memory_final[fov_id]['saved']:
                    
                    # Access data first for logging before potential deletion/modification
                    cropped_images_data = shared_memory_classification[fov_id]['cropped_images']
                    scores_data = shared_memory_classification[fov_id]['scores']
                    filtered_spots_data = shared_memory_classification[fov_id]['filtered_spots']
                    spot_list_data = shared_memory_classification[fov_id]['spot_list']
                    fluorescent_image_data = shared_memory_acquisition[fov_id]['fluorescent']
                    dpc_image_data = shared_memory_dpc[fov_id]['dpc_image']

                    # Input from shared_mem: cropped_images (ndarray, (M, 4, 31, 31), float16)
                    # Input from shared_mem: scores (ndarray, (M,), float32)
                    # Input from shared_mem: filtered_spots (ndarray, (M, 3), int64)
                    # Input from shared_mem: spot_list (ndarray, (N, 3), int64)
                    # Input from shared_mem: fluorescent_image (ndarray, (2800, 2800, 3), uint8)
                    # Input from shared_mem: dpc_image (ndarray, (2800, 2800), float16)

                    # save the cropped images to png
                    cropped_images = (cropped_images_data*255).astype(np.uint8)
                    scores = scores_data # Already retrieved
                    # Converted: cropped_images (ndarray, (M, 4, 31, 31), uint8)
                    
                    save_path = shared_config.get_path()
                    
                    if shared_config.save_spot_images.value:
                        filename = os.path.join(save_path, f"{fov_id}_cropped.npy")
                        np.save(filename, cropped_images)
                        filename = os.path.join(save_path, f"{fov_id}_scores.npy")
                        np.save(filename, scores)
                        filename = os.path.join(save_path, f"{fov_id}_filtered_spots.npy")
                        np.save(filename, filtered_spots_data)
                        filename = os.path.join(save_path, f"{fov_id}_spot_list.npy")
                        np.save(filename, spot_list_data)
                    if shared_config.save_dpc_image.value:
                        #filename = os.path.join(save_path, f"{fov_id}_overlay.npy")
                        #fluorescent_image = shared_memory_acquisition[fov_id]['fluorescent']
                        filename = os.path.join(save_path, f"{fov_id}_dpc.npy")
                        dpc_image = dpc_image_data # Already retrieved
                        #fluorescent_image_int8 = fluorescent_image.astype(np.uint8)
                        dpc_image_int8 = (dpc_image*255).astype(np.uint8)
                        #img = np.stack([fluorescent_image_int8[:,:,0], fluorescent_image_int8[:,:,1], fluorescent_image_int8[:,:,2], dpc_image_int8], axis=0)
                        if shared_config.SAVE_NPY.value:
                            np.save(filename, dpc_image_int8)
                        else:
                            filename = os.path.join(save_path, f"{fov_id}_dpc.bmp")
                            cv2.imwrite(filename, dpc_image_int8)
                            # print(f"[DATA_LOG] Save: Saved {filename}")

                    # save the overlay image with bounding boxes
                    filename = os.path.join(save_path, f"{fov_id}_overlay_bb.bmp")
                    fluorescent_image = fluorescent_image_data # Already retrieved
                    #print("fluorescent_image",fluorescent_image.dtype)
                    dpc_image = dpc_image_data # Already retrieved
                    #print("dpc_image",dpc_image.dtype)
                    filtered_spots = filtered_spots_data # Already retrieved
                    spot_list = spot_list_data # Already retrieved
                    I_combined = draw_spot_bounding_boxes(np.array(fluorescent_image), np.array(dpc_image), spot_list, filtered_spots,spot_list2_scores = scores)
                    # Generated: I_combined (ndarray, (2800, 2800, 3), uint8)
                    #print("bounding box saving to ",filename)
                    cv2.imwrite(filename, I_combined)

                    temp_dict = shared_memory_final[fov_id]
                    temp_dict['saved'] = True
                    shared_memory_final[fov_id] = temp_dict

                    if shared_memory_final[fov_id]['displayed']:
                        output.put(fov_id)
                        # Queue Output: fov_id (str) to cleanup_queue
            
                    log_time(fov_id, "Saving Process", "end")
        
        except Empty:
            continue
        except Exception as e:
            logger = shared_config.setup_process_logger()
            logger.error(f"Unknown error in saving process {e}")
            continue

    print("Saving process finished")

import random
def cloud_upload_process(shutdown_event: mp.Event, start_event: mp.Event):
    """
    Uploads patient data (results, selected images) to Google Cloud Storage.

    Monitors a patient queue, waits for analysis completion markers (stats.txt, rbc_counts.csv),
    and uploads specified files to the configured GCS bucket.

    Args:
        shutdown_event (mp.Event): Event to signal process termination.
        start_event (mp.Event): Event to signal the start of processing (currently only affects sleep).

    Shared Memory Input:
        shared_memory_patient_queue (list): Reads patient IDs to process.
        Reads shared_config for patient data path and save flags.

    Shared Memory Output:
        Removes processed patient ID from shared_memory_patient_queue.

    Queue Output:
        None
    """
    # Check for Google Cloud credentials
    if 'SERVICE_ACCOUNT_JSON_KEY' not in os.environ:
        print("Error: Google Cloud credentials not found in environment variables.")
        return

    # Check for bucket name
    bucket_name = os.environ.get('BUCKET_NAME')
    if not bucket_name:
        print("Error: Google Cloud bucket name not found in environment variables.")
        return

    # Initialize Google Cloud client
    try:
        credentials = service_account.Credentials.from_service_account_file(
            os.environ['SERVICE_ACCOUNT_JSON_KEY'])
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"Error initializing Google Cloud client: {e}")
        print("The files will not be uploaded to the cloud")
        return

    while not shutdown_event.is_set():
        # Process any patients in the queue, regardless of the start_event state
        with patient_queue_lock:
            if len(shared_memory_patient_queue) > 0:
                patient_id = shared_memory_patient_queue[0]
            else:
                patient_id = None

        if patient_id:
            patient_path = shared_config.get_path()
            stats_file = os.path.join(patient_path, 'stats.txt')
            rbc_count_file = os.path.join(patient_path, 'rbc_counts.csv')

            # Wait for both files to exist
            while not (os.path.exists(stats_file) and os.path.exists(rbc_count_file)):
                print(f"Waiting for patient {patient_id} to be uploaded")
                if shutdown_event.is_set():
                    return
                time.sleep(3)  # Check every 5 seconds

            print(f"Uploading data for patient {patient_id}")

            try:
                # Upload all cropped images, spot lists, scores, and txt files
                for root, _, files in os.walk(patient_path):
                    for file in files:
                        if file.endswith(('.npy', '.txt','.csv')):
                            local_path = os.path.join(root, file)
                            cloud_path = f"{patient_id}/{file}"
                            blob = bucket.blob(cloud_path)
                            blob.upload_from_filename(local_path)
            
                fov_ids = [f.replace('_dpc.bmp', '') for f in os.listdir(patient_path) if f.endswith('_dpc.bmp')]
                print(f"There will be {len(fov_ids)} files uploaded for patient {patient_id}")
                # randomly select 10 files
                if len(fov_ids) > 5:
                    fov_ids = random.sample(fov_ids, 5)
                for fov_id in fov_ids:
                    if shared_config.save_dpc_image.value:  
                        local_path = os.path.join(patient_path, f"{fov_id}_dpc.bmp")
                        cloud_path = f"{patient_id}/{fov_id}_dpc.bmp"
                        blob = bucket.blob(cloud_path)
                        blob.upload_from_filename(local_path)
                    if shared_config.save_fluo_images.value:
                        local_path = os.path.join(patient_path, f"{fov_id}_fluorescent.bmp")
                        cloud_path = f"{patient_id}/{fov_id}_fluorescent.bmp"
                        blob = bucket.blob(cloud_path)
                        blob.upload_from_filename(local_path)
                print(f"Finished uploading data for patient {patient_id}")

            except Exception as e:
                print(f"Error uploading data for patient {patient_id}: {str(e)}")
            # Remove the patient ID from the queue
            with patient_queue_lock:
                shared_memory_patient_queue.pop(0)
        else:
            # If no patients in queue, sleep for a short time before checking again
            time.sleep(3)

    print("Cloud upload process finished")  

def cleanup_process(cleanup_queue: mp.Queue,shutdown_event: mp.Event,start_event: mp.Event):
    """
    Removes processed FOV data from all relevant shared memory dictionaries.

    Logs final counts and timing information before deleting entries.

    Args:
        cleanup_queue (mp.Queue): Queue receiving fov_id (str) from saving_process or ui_process.
        shutdown_event (mp.Event): Event to signal process termination.
        start_event (mp.Event): Event to signal the start of processing.

    Shared Memory Input:
        shared_memory_timing[fov_id]: Reads timing data for reporting.
        shared_memory_segmentation[fov_id]: Reads 'n_cells'.
        shared_memory_fluorescent[fov_id]: Reads 'spot_indices'.
        shared_memory_classification[fov_id]: Reads 'filtered_spots_count'.
        Checks for fov_id key existence in all shared memories before deletion.

    Shared Memory Output:
        Deletes the fov_id entry from:
         - shared_memory_acquisition
         - shared_memory_dpc
         - shared_memory_segmentation
         - shared_memory_fluorescent
         - shared_memory_classification
         - shared_memory_final
         - shared_memory_timing

    Queue Output:
        None
    """
    while not shutdown_event.is_set():
        start_event.wait()
        
        try:
            fov_id = cleanup_queue.get(timeout=timeout)
            # Queue Input: fov_id (str)
            
            # Log data before deletion (Optional - kept print for this specific case as it summarizes counts)
            with final_lock, timing_lock:
                try:
                    n_cells_val = shared_memory_segmentation.get(fov_id, {}).get('n_cells', 'N/A')
                    spot_indices_len = len(shared_memory_fluorescent.get(fov_id, {}).get('spot_indices', []))
                    filtered_spots_count_val = shared_memory_classification.get(fov_id, {}).get('filtered_spots_count', 'N/A')
                    print(f"[INFO] Cleanup: Data for {fov_id} - RBCs: {n_cells_val}, Spots: {spot_indices_len}, Filtered Spots: {filtered_spots_count_val}") # Changed prefix to INFO
                except Exception as log_err:
                    print(f"[WARN] Cleanup: Error logging data for {fov_id} before deletion: {log_err}") # Changed prefix to WARN

                # Calculate processing times and generate visualization
                timing_data = shared_memory_timing.get(fov_id, {})

                logger = shared_config.setup_process_logger()
                logger.info(f"\n{'=' * 50}")
                logger.info(f"Report for FOV {fov_id}:")
                logger.info(f"RBC counts: {shared_memory_segmentation[fov_id]['n_cells']}, spots: {len(shared_memory_fluorescent[fov_id]['spot_indices'])},filtered spots: {shared_memory_classification[fov_id]['filtered_spots_count']}")
                logger.info(f"{'=' * 50}")
                report(timing_data, fov_id, logger)

                shared_memories = [shared_memory_acquisition, shared_memory_dpc, shared_memory_segmentation, 
                                   shared_memory_fluorescent, shared_memory_classification, shared_memory_final, 
                                   shared_memory_timing]
                
                for shared_memory in shared_memories:
                    if fov_id in shared_memory:
                        del shared_memory[fov_id]
                
                # Confirm deletion
                if all(fov_id not in shared_memory for shared_memory in shared_memories):
                    logger.info(f"FOV {fov_id} memory has been successfully freed up")
                else:
                    logger.error(f"Warning: FOV {fov_id} may not have been completely removed from all shared memories")
        except Empty:
            continue
        except Exception as e:
            logger = shared_config.setup_process_logger()
            logger.error(f"Unknown error in cleanup process {e}")
            continue

    print("Cleanup process finished")

from ui import ui_process

if __name__ == "__main__":

    # get the first cmd argument
    import sys
    if len(sys.argv) > 1:
        simulation = sys.argv[1] == "simulation"
    else:
        simulation = False

    # Initialize shared config and pass it to simulation module
    set_shared_config(shared_config)

    # Create queues
    dpc_queue = mp.Queue()
    fluorescent_queue = mp.Queue()
    segmentation_queue = mp.Queue()
    fluorescent_detection_queue = mp.Queue()
    classification_queue = mp.Queue()
    save_queue = mp.Queue()
    ui_queue = mp.Queue()
    cleanup_queue = mp.Queue()

    # Create an event to signal shutdown
    shutdown_event = mp.Event()
    # Create an event to signal start
    start_event = mp.Event()

    # Create and start processes
    processes = [
        mp.Process(target=image_acquisition_simulation if simulation else image_acquisition, args=(dpc_queue, fluorescent_queue, shutdown_event,start_event), name="Image Acquisition"),
        mp.Process(target=dpc_process, args=(dpc_queue, segmentation_queue, shutdown_event,start_event), name="DPC Process"),
        mp.Process(target=fluorescent_spot_detection, args=(fluorescent_queue, fluorescent_detection_queue, shutdown_event,start_event), name="Fluorescent Spot Detection"),
        mp.Process(target=saving_process, args=(save_queue, cleanup_queue, shutdown_event,start_event), name="Saving Process"),
        mp.Process(target=cleanup_process, args=(cleanup_queue, shutdown_event,start_event), name="Cleanup Process")
    ]

    # Start the UI
    ui_process = mp.Process(target=ui_process, args=(ui_queue, cleanup_queue, shared_memory_final, shared_memory_classification, 
                                            shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, 
                                            shared_memory_timing, final_lock, timing_lock,start_event,shutdown_event,shared_config), name="UI Process")

    ui_process.start()

    for p in processes:
        p.start()

    classification_thread = threading.Thread(target=classification_process, 
                                             args=(classification_queue, fluorescent_detection_queue, save_queue, ui_queue, shutdown_event,start_event),name="Classification Process")
    segmentation_thread = threading.Thread(target=segmentation_process,
                                           args=(segmentation_queue, classification_queue, shutdown_event,start_event),name="Segmentation Process")

    classification_thread.start()
    segmentation_thread.start()

    # check if the google tools are imported, if so launch the cloud upload process
    if gcloud_available:
        # process
        cloud_upload_process = mp.Process(target=cloud_upload_process, args=(shutdown_event, start_event), name="Cloud Upload Process")
        cloud_upload_process.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(1)    
    except KeyboardInterrupt:
        #logger.info("Stopping all processes...")
        print("Stopping all processes...")
    finally:
        shutdown_event.set()
        for p in processes:
            if p.name == "Image Acquisition":
                p.join(timeout=2)  # Give image acquisition process more time to shut down
            else:
                p.join(timeout=1)  # Give other processes more time to shut down
            if p.is_alive():
                #logger.info(f"Force terminating process {p.name}")
                print(f"Force terminating process {p.name}")
                p.terminate()
        classification_thread.join(timeout=1)
        segmentation_thread.join(timeout=1)
        ui_process.join(timeout=1)
        if ui_process.is_alive():
            #logger.info("Force terminating UI process")
            print("Force terminating UI process")
            ui_process.terminate()
        #logger.info("All processes have been shut down.")
        if cloud_upload_process.is_alive():
            print("Waiting for cloud upload process to finish...")
            cloud_upload_process.join(timeout=1000)
            cloud_upload_process.terminate()
        print("All processes have been shut down.")
        if classification_thread.is_alive() or segmentation_thread.is_alive():
            #logger.info("Force terminating classification and segmentation threads")
            print("Force terminating classification and segmentation threads")
            import os
            os._exit(0)
