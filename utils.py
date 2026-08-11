import imageio
import cv2
import cupy as cp # conda install -c conda-forge cupy==10.2 or conda install -c conda-forge cupy cudatoolkit=11.0
import cupyx.scipy.ndimage
# from cupyx.scipy import ndimage # for 11.0
# from ndimage import laplace # for 11.0
from cupyx.scipy.ndimage import laplace # for 11.0
# from cupyx.scipy.ndimage.filters import laplace # for 10.2
from skimage.feature.blob import _prune_blobs
import numpy as np
from scipy import signal
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
import os, sys

def free_gpu_memory(func):
    def wrapper_func(*args, **kwargs):
        retval = func(*args, **kwargs)
        cp._default_memory_pool.free_all_blocks()
        return retval
    return wrapper_func

def imread_gcsfs(fs,file_path):
	img_bytes = fs.cat(file_path)
	I = imageio.core.asarray(imageio.imread(img_bytes, "bmp"))
	return I

@free_gpu_memory
def resize_cp(ar,downsize_factor=4):
	# by Rinni
    if not cp.is_available():
        raise Exception("No GPU device found")
    device_id = cp.cuda.Device()
    with cp.cuda.Device(device_id):
        s_ar = cp.zeros((int(ar.shape[0]/downsize_factor), int(ar.shape[0]/downsize_factor), 3))
        s_ar[:,:,0] = ar[:,:,0].reshape([int(ar.shape[0]/downsize_factor), int(downsize_factor), int(ar.shape[1]/downsize_factor), int(downsize_factor)]).mean(3).mean(1)
        s_ar[:,:,1] = ar[:,:,1].reshape([int(ar.shape[0]/downsize_factor), int(downsize_factor), int(ar.shape[1]/downsize_factor), int(downsize_factor)]).mean(3).mean(1)
        s_ar[:,:,2] = ar[:,:,2].reshape([int(ar.shape[0]/downsize_factor), int(downsize_factor), int(ar.shape[1]/downsize_factor), int(downsize_factor)]).mean(3).mean(1)
    
    return s_ar

@free_gpu_memory
def resize_image_cp(I,downsize_factor=4):
     # check if I is numpy array
    if not cp.is_available():
        raise Exception("No GPU device found")
    device_id = cp.cuda.Device()
    with cp.cuda.Device(device_id):
        if isinstance(I, np.ndarray):
            I = cp.asarray(I)   
            I = I.astype('float16')
        #I_resized = cp.copy(I)
        I_resized = resize_cp(I, downsize_factor)

        cp.get_default_memory_pool().free_all_blocks()
    return I_resized

@free_gpu_memory
def remove_background(img_cpu, return_gpu_image=True):
    # check if the cp device is available
    if not cp.is_available():
        raise Exception("No GPU device found")
    device_id = cp.cuda.Device()
    with cp.cuda.Device(device_id):
        tophat = cv2.getStructuringElement(2, ksize=(17,17))
        tophat_gpu = cp.asarray(tophat)
        img_g_gpu = cp.asarray(img_cpu, dtype=cp.float16)
        img_th_gpu = img_g_gpu
        for k in range(3):
            img_th_gpu[:,:,k] = cupyx.scipy.ndimage.white_tophat(img_g_gpu[:,:,k], footprint=tophat_gpu)
        if return_gpu_image:
            result = img_th_gpu
        else:
            result = cp.asnumpy(img_th_gpu)
         
        del tophat_gpu, img_g_gpu
        cp.get_default_memory_pool().free_all_blocks()
    
    return result

@free_gpu_memory
def gaussian_kernel_1d(n, std, normalized=True):
    if normalized:
        return cp.asarray(signal.windows.gaussian(n, std))/(np.sqrt(2 * np.pi)*std)
    return cp.asarray(signal.windows.gaussian(n, std), dtype=cp.float16)

@free_gpu_memory
def detect_spots(I, thresh = 12):
    # filters
    # check if the cp device is available
    if not cp.is_available():
        raise Exception("No GPU device found")
    device_id = cp.cuda.Device()
    with cp.cuda.Device(device_id):
        gauss_rs = np.array([4,6,8,10])
        gauss_sigmas = np.array([1,1.5,2,2.5])
        gauss_ts = np.divide(gauss_rs - 0.5,gauss_sigmas) # truncate value (to get desired radius)
        lapl_kernel = cp.array([[0,1,0],[1,-4,1],[0,1,0]])
        gauss_filters_1d = []
        for i in range(gauss_rs.shape[0]):
            gauss_filt_1d = gaussian_kernel_1d(gauss_rs[i]*2+1,gauss_sigmas[i],True)
            gauss_filt_1d = gauss_filt_1d.reshape(-1, 1)
            gauss_filters_1d.append(gauss_filt_1d)
        # apply all filters
        if len(I.shape) == 3:
            I = cp.average(I, axis=2, weights=cp.array([0.299,0.587,0.114],dtype=cp.float16))
        filtered_imgs = []
        for i in range(len(gauss_filters_1d)): # apply LoG filters
            filt_img = cupyx.scipy.ndimage.convolve(I, gauss_filters_1d[i])
            filt_img = cupyx.scipy.ndimage.convolve(filt_img, gauss_filters_1d[i].transpose())
            filt_img = cupyx.scipy.ndimage.convolve(filt_img, lapl_kernel)
            filt_img *= -(gauss_sigmas[i]**2)
            filtered_imgs.append(filt_img)
        img_max_proj = cp.max(np.stack(filtered_imgs), axis=0)
        img_max_filt = cupyx.scipy.ndimage.maximum_filter(img_max_proj, size=3)
        img_max_filt[img_max_filt < thresh] = 0
        img_traceback = cp.zeros(img_max_filt.shape)
        for i in range(len(filtered_imgs)):
            img_traceback[img_max_filt == filtered_imgs[i]] = i+1
        ind = np.where(img_traceback != 0)
        spots = np.zeros((ind[0].shape[0],3))
        for i in range(ind[0].shape[0]):
            spots[i][0] = int(ind[1][i])
            spots[i][1] = int(ind[0][i])
            spots[i][2] = int(img_traceback[spots[i][1]][spots[i][0]])
        spots = spots.astype(int)
    return spots

# filter spots to avoid overlapping ones
num_sigma = 4
min_sigma = 1
max_sigma = 2.5
scale = np.linspace(0, 1, num_sigma)[:, np.newaxis]

@free_gpu_memory
def prune_blobs(spots_list):
    overlap = .5
    sigma_list = scale * (max_sigma - min_sigma) + min_sigma
    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[spots_list[:, -1]-1]
    # select one sigma column, keeping dimension
    sigmas_of_peaks = sigmas_of_peaks[:, 0:1]
    # Replace sigma index with sigmas in-place
    spots_list[:, -1] = sigmas_of_peaks[:, 0]
    result_pruned = _prune_blobs(spots_list, overlap)
    return result_pruned

@free_gpu_memory
def highlight_spots(I,spot_list,contrast_boost=1.6):
	# bgremoved_fluorescence_spotBoxed = np.copy(bgremoved_fluorescence)
	I = I.astype('float16')/255 # this copies the image
	I = I*contrast_boost # enhance contrast
	for s in spot_list:
		add_bounding_box(I,int(s[0]),int(s[1]),int(s[2]))
	return I

def add_bounding_box(I,x,y,r,extension=2,color=[0.6,0.6,0]):
	ny, nx, nc = I.shape
	x_min = max(x - r - extension,0)
	y_min = max(y - r - extension,0)
	x_max = min(x + r + extension,nx-1)
	y_max = min(y + r + extension,ny-1)
	for i in range(3):
		I[y_min,x_min:x_max+1,i] = color[i]
		I[y_max,x_min:x_max+1,i] = color[i]
		I[y_min:y_max+1,x_min,i] = color[i]
		I[y_min:y_max+1,x_max,i] = color[i]

@free_gpu_memory
def remove_spots_in_masked_regions(spotList,mask):
	mask = mask.astype('float')/255
	mask = np.sum(mask,axis=-1) # masked out region has pixel value 0 ;# mask[mask>0] = 1 #         cv2.imshow('mask',mask) # cv2.waitKey(0)
	for s in spotList:
		x = s[0]
		y = s[1]
		if mask[int(y),int(x)] == 0:
			s[-1] = 0
	spot_list = np.array([s for s in spotList if s[-1] > 0])
	return spot_list

@free_gpu_memory
def extract_spot_data(I_background_removed,I_raw,spot_list,i,j,k,settings,extension=1):
	downsize_factor=settings['spot_detection_downsize_factor']
	extension = extension*downsize_factor
	ny, nx, nc = I_background_removed.shape
	I_background_removed = I_background_removed.astype('float16')
	I_raw = I_raw.astype('float16')/255
	columns = ['FOV_row','FOV_col','FOV_z','x','y','r','R','G','B','R_max','G_max','B_max','lap_total','lap_max','numPixels','numSaturatedPixels','idx']
	spot_data_pd = pd.DataFrame(columns=columns)
	idx = 0
	for s in spot_list:
		# get spot
		x = int(s[0])
		y = int(s[1])
		r = s[2]
		x_min = max(int((x - r - extension)),0)
		y_min = max(int((y - r - extension)),0)
		x_max = min(int((x + r + extension)),nx-1)
		y_max = min(int((y + r + extension)),ny-1)
		cropped = I_background_removed[y_min:(y_max+1),x_min:(x_max+1),:]
		cropped_raw = I_raw[y_min:(y_max+1),x_min:(x_max+1),:]
		# extract spot data
		B = cp.asnumpy(cp.sum(cropped[:,:,2]))
		G = cp.asnumpy(cp.sum(cropped[:,:,1]))
		R = cp.asnumpy(cp.sum(cropped[:,:,0]))
		B_max = cp.asnumpy(cp.max(cropped[:,:,2]))
		G_max = cp.asnumpy(cp.max(cropped[:,:,1]))
		R_max = cp.asnumpy(cp.max(cropped[:,:,0]))
		lap = laplace(cp.sum(cropped,2))
		lap_total = cp.asnumpy(cp.sum(cp.abs(lap)))
		lap_max = cp.asnumpy(cp.max(cp.abs(lap)))
		numPixels = cropped[:,:,0].size
		numSaturatedPixels = cp.asnumpy(cp.sum(cropped_raw == 1))
		# add spot entry
		spot_entry = pd.DataFrame.from_dict({'FOV_row':[i],'FOV_col':[j],'FOV_z':[k],'x':[x],'y':[y],'r':[r],'R':[R],'G':[G],'B':[B],'R_max':[R_max],'G_max':[G_max],'B_max':[B_max],'lap_total':[lap_total],'lap_max':[lap_max],'numPixels':[numPixels],'numSaturatedPixels':[numSaturatedPixels],'idx':[idx]})
		# spot_data_pd = spot_data_pd.append(spot_entry, ignore_index=True, sort=False)
		spot_data_pd = pd.concat([spot_data_pd,spot_entry])
		# increament idx
		idx = idx + 1
	return spot_data_pd

def process_spots(I_background_removed,I_raw,spot_list,i,j,k,settings,I_mask=None):
	# get rid of spots in masked out regions
	if I_mask!=None:
		spot_list = remove_spots_in_masked_regions(spot_list,I_mask)
	# extract spot statistics
	spot_data_pd = extract_spot_data(I_background_removed,I_raw,spot_list,i,j,k,settings)
	return spot_list, spot_data_pd


def seg_spot_filter_one_fov(mask, spots, crop_offset_x=0, crop_offset_y=0, overlap_threshold=6/9):
    
    # Erode the mask
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel)
    
    # List to store overlapping spots
    overlapping_spots = []
    
    # Process each spot
    for spot in spots:
        x, y, radius = spot[:3]  # We don't use the radius in this function, but we keep it for the output
        x, y = int(x), int(y)
        
        # Calculate overlap
        roi = eroded_mask[crop_offset_y+y-1:crop_offset_y+y+2, crop_offset_x+x-1:crop_offset_x+x+2]
        if roi.size == 9:  # Ensure we have a full 3x3 region
            overlap = np.sum(roi > 0) / 9
            
            # If overlap is above threshold, add to list of overlapping spots
            if overlap >= overlap_threshold:
                overlapping_spots.append(spot)
    
    # Convert list of overlapping spots to numpy array
    return np.array(overlapping_spots)

@free_gpu_memory
def generate_dpc(I1, I2, use_gpu=False):
    if use_gpu:
        # Convert numpy arrays to CuPy arrays
        I1 = cp.asarray(I1)
        I2 = cp.asarray(I2)

        # Add a small constant to avoid divide-by-zero
        epsilon = cp.float16(1e-7)

        # Compute the sum once
        I_sum = I1 + I2 + epsilon

        # Compute DPC
        I_dpc = cp.divide(I1 - I2, I_sum)

        # Shift and clip values
        I_dpc = cp.clip(I_dpc + 0.5, 0, 1)

        # Convert the result back to a numpy array
        I_dpc = cp.asnumpy(I_dpc)

    else:
        # Add a small constant to avoid divide-by-zero
        epsilon = np.float16(1e-7)
        
        # Compute the sum once
        I_sum = I1 + I2 + epsilon
        
        # Compute DPC
        I_dpc = np.divide(I1 - I2, I_sum)
        
        # Shift and clip values
        I_dpc = np.clip(I_dpc + 0.5, 0, 1)
    
    return I_dpc

def draw_bounding_box(image, spot, r, box_color):
    x, y = int(spot[0]), int(spot[1])
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(image.shape[1] - 1, x + r), min(image.shape[0] - 1, y + r)
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 1)

def draw_spot_bounding_boxes(I_fluorescence, I_dpc, spot_list1, spot_list2, 
                             box_color1=(0, 255, 0), box_color2=(0, 0, 255), r=15, spot_list2_scores=None):
    # Ensure I_dpc is a single channel image
    if len(I_dpc.shape) == 3:
        I_dpc = I_dpc[:,:,1]
    
    # Normalize I_fluorescence and I_dpc
    #assert I_fluorescence.dtype == np.int8
    #assert I_dpc.dtype == np.float16
    
    # Combine fluorescence (0.67) and DPC images (0.33)
    I_fluorescence_fp = I_fluorescence.astype(np.float16)/255
    # dpc to three channels
    I_dpc_fp = np.dstack([I_dpc,I_dpc,I_dpc])
    I_combined = I_fluorescence_fp*0.67 + I_dpc_fp*0.33
    I_combined = (I_combined*255).astype(np.uint8)
    
    # Draw bounding boxes for spot_list1
    for spot in spot_list1:
        draw_bounding_box(I_combined, spot, r, box_color1)

    # Draw bounding boxes for spot_list2
    for spot in spot_list2:
        draw_bounding_box(I_combined, spot, r, box_color2)

    if spot_list2_scores is not None:
        for spot,score in zip(spot_list2,spot_list2_scores):
            # color text of score red if score is greater than 0.31 otherwise blue
            if score > 0.5:
                cv2.putText(I_combined, f"{score:.2f}", (int(spot[0])+r, int(spot[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(I_combined, f"{score:.2f}", (int(spot[0])+r, int(spot[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return I_combined

def get_spot_images_from_fov(I_fluorescence,I_dpc,spot_list,r=15):
    if(len(I_dpc.shape)==3):
        I_dpc = I_dpc[:,:,1]

    height,width,channels = I_fluorescence.shape

    num_spots = len(spot_list)
    I = np.zeros((num_spots, 2*r+1, 2*r+1, 4), np.float16)  # preallocate memory

    for counter, s in enumerate(spot_list):
        x = int(s[0])
        y = int(s[1])

        x_start = max(0,x-r)
        x_end = min(x+r,width-1)
        y_start = max(0,y-r)
        y_end = min(y+r,height-1)

        x_idx_FOV = slice(x_start,x_end+1)
        y_idx_FOV = slice(y_start,y_end+1)

        x_cropped_start = x_start - (x-r)
        x_cropped_end = (2*r+1-1) - ((x+r)-x_end)
        y_cropped_start = y_start - (y-r)
        y_cropped_end = (2*r+1-1) - ((y+r)-y_end)

        x_idx_cropped = slice(x_cropped_start,x_cropped_end+1)
        y_idx_cropped = slice(y_cropped_start,y_cropped_end+1)

        I[counter, y_idx_cropped, x_idx_cropped, :3] = I_fluorescence[y_idx_FOV,x_idx_FOV,:]
        I[counter, y_idx_cropped, x_idx_cropped, 3] = I_dpc[y_idx_FOV,x_idx_FOV]

    if num_spots == 0:
        print('no spot in this FOV')
        return None
    else:
        return I
    
import imageio
import numpy as np
import cv2

def numpy2png(img,filename,resize_factor=5):
	img = img.transpose(1,2,0)
	img_fluorescence = img[:,:,[2,1,0]]
	img_dpc = img[:,:,3]
	img_dpc = np.dstack([img_dpc,img_dpc,img_dpc])
	img_overlay = 0.64*img_fluorescence + 0.36*img_dpc
	x = resize_factor
	img_overlay = cv2.resize(img_overlay, (int(img_overlay.shape[1]*x), int(img_overlay.shape[0]*x)), interpolation=cv2.INTER_NEAREST)
	imageio.imwrite(filename + "_overlay.png", np.uint8(img_overlay))


def numpy2png_ui(img, resize_factor=5):
    try:
        # Ensure the image is in the correct shape (H, W, C)
        if img.shape[0] == 4:  # If the first dimension is 4, it's likely (C, H, W)
            img = img.transpose(1, 2, 0)
        
        # Separate fluorescence and DPC channels
        img_fluorescence = img[:, :, [2,1,0]]  # First 3 channels, but in reverse order
        img_dpc = img[:, :, 3]  # Last channel

        # Normalize the fluorescence image
        epsilon = 1e-7
        img_fluorescence = (img_fluorescence - img_fluorescence.min()) / (img_fluorescence.max() - img_fluorescence.min() + epsilon)
        img_fluorescence = (img_fluorescence * 255).astype(np.uint8)

        # Normalize the DPC image
        img_dpc = (img_dpc - img_dpc.min()) / (img_dpc.max() - img_dpc.min() + epsilon)
        img_dpc = (img_dpc * 255).astype(np.uint8)
        img_dpc = np.dstack([img_dpc, img_dpc, img_dpc])  # Make it 3 channels

        # Combine fluorescence and DPC
        img_overlay = cv2.addWeighted(img_fluorescence, 0.64, img_dpc, 0.36, 0)

        # Resize
        if resize_factor is not None:
            if resize_factor >=1:
                img_overlay = cv2.resize(img_overlay, (img_overlay.shape[1]*resize_factor, img_overlay.shape[0]*resize_factor), interpolation=cv2.INTER_NEAREST)
            if resize_factor < 1:
                img_overlay = cv2.resize(img_overlay, (int(img_overlay.shape[1]*resize_factor), int(img_overlay.shape[0]*resize_factor)), interpolation=cv2.INTER_NEAREST)

        return img_overlay
    except Exception as e:
        print(f"Error in numpy2png: {e}")
        return None

def save_flourescence_image(img,filename):
	# 3 channels image
	img = img[:,:,[2,1,0]]
	imageio.imwrite(filename + "_fluorescence.png", np.uint8(img))
def save_dpc_image(img,filename):
	# grey scale image
	img = img*255.0
	
	imageio.imwrite(filename + "_dpc.png", np.uint8(img))
	

settings = {'spot_detection_downsize_factor': 4, 'spot_detection_threshold': 10}

def gpu_cuda_available():
    """Best-effort check of whether the GPU is actually usable right now.

    cp.is_available() is supposed to return False when there's no GPU, but a
    stuck CUDA context (e.g. left behind after the machine suspends/resumes)
    makes it raise instead, so guard the call explicitly rather than letting
    that exception propagate.
    """
    try:
        return bool(cp.is_available())
    except Exception:
        return False

def reset_nvidia_uvm():
    """Reload the nvidia_uvm kernel module to clear a stuck CUDA context
    (common after suspend/resume). Prompts the user for their password via
    pkexec/PolicyKit — never runs silently. Returns (success, message).

    Note: this only fixes the driver at the OS level. A process that already
    has a poisoned CUDA handle (e.g. the app that's currently showing the
    warning) can't recover in place and must be restarted.
    """
    import subprocess
    try:
        result = subprocess.run(
            ['pkexec', 'sh', '-c', 'modprobe -r nvidia_uvm && modprobe nvidia_uvm'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return True, "nvidia_uvm reloaded."
        return False, (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
    except Exception as e:
        return False, str(e)

import torch.multiprocessing as mp
class SharedConfig:
    def __init__(self):
        self.manager = mp.Manager()
        self.path = self.manager.Value('s', '')  # 's' for string
        self.save_bf_images = self.manager.Value('b', False)      # 'b' for boolean
        self.save_fluo_images = self.manager.Value('b', False)  # 'b' for boolean
        self.save_spot_images = self.manager.Value('b', False)     # 'b' for boolean
        self.save_dpc_image = self.manager.Value('b', False)     # 'b' for boolean
        self.nx = self.manager.Value('i', 0)  # 'i' for integer
        self.ny = self.manager.Value('i', 0)  # 'i' for integer

        self.patient_id = self.manager.Value('s', '')

        self.finished_scanning = self.manager.Value('b', False)

        # for simulation mode
        self.simulation_path = self.manager.Value('s', './sample_inputs')

        # for live viewing
        self.is_live_view_active = self.manager.Value('b', False)
        self.live_channel_selected = self.manager.Value('i', 0)
        self.live_channels_list = self.manager.list(["BF LED matrix left half","BF LED matrix right half","Fluorescence 405 nm Ex"])
        self.live_x = self.manager.Value('f', -1)
        self.live_y = self.manager.Value('f', -1)
        self.live_z = self.manager.Value('f', -1)

        self.is_auto_focus_calibration = self.manager.Value('b', False)

        # indicator for auto-focusing
        self.auto_focus_indicator = self.manager.Value('b', False)
        
        self.IMAGE_SHAPE = (2800,2800,3)
        self.IMAGE_SIZE = self.IMAGE_SHAPE[0] * self.IMAGE_SHAPE[1] * self.IMAGE_SHAPE[2]
        self.live_view_image_array = mp.RawArray('f', self.IMAGE_SIZE)
        self.live_view_image_lock = mp.Lock()
        self.frame_rate = self.manager.Value('f', 30.0)

        # for position loading and scanning
        self.position_lock = self.manager.Lock()
        self.to_loading = self.manager.Value('b', False)  # 'b' for boolean
        self.to_scanning = self.manager.Value('b', False)  # 'b' for boolean

        self.log_file = self.manager.Value('s', './')  # Shared string for log file path

        self.SAVE_NPY = self.manager.Value('b', False)

        # General acquisition mode (independent of malaria pipeline)
        self.ga_active = self.manager.Value('b', False)
        self.ga_running = self.manager.Value('b', False)
        self.ga_objective = self.manager.Value('s', '20x (Olympus UPlanFL N 0.50)')
        self.ga_nx = self.manager.Value('i', 3)
        self.ga_ny = self.manager.Value('i', 3)
        self.ga_overlap_pct = self.manager.Value('f', 10.0)
        self.ga_af_mode = self.manager.Value('s', 'focus_map')  # 'focus_map' | 'every_n' | 'none'
        self.ga_af_every_n = self.manager.Value('i', 3)
        self.ga_af_start_mm = self.manager.Value('f', 6.3)
        self.ga_af_end_mm = self.manager.Value('f', 6.5)
        self.ga_channels = self.manager.list(["BF LED matrix left half"])
        self.ga_save_name = self.manager.Value('s', 'acq')
        self.ga_save_path = self.manager.Value('s', '')  # set by acquisition loop, read by UI for tile view

        # Classification model selection. One of: 'O1.9', 'O2.0', 'v8'.
        # O1.9 = ensemble (resnet18_en v1 + v2, np.minimum), threshold 0.31
        # O2.0 = single resnet18_en v1, threshold 0.5 (legacy default)
        # v8   = v8_hardneg_single (Heguang), threshold 0.99502 (per t_spot.json)
        self.model_selection = self.manager.Value('s', 'O2.0')

        # Per-channel runtime setting overrides (exposure_ms, analog_gain, illumination_intensity).
        # UI writes; image_acquisition reads on dirty flag, mutates Configuration objects, and
        # re-applies the current channel. Session-only — not written back to XML.
        self.channel_overrides = self.manager.dict()
        self.channel_overrides_dirty = self.manager.Value('b', False)
        self.channel_overrides_lock = self.manager.Lock()

    def set_auto_focus_indicator(self, value):
        self.auto_focus_indicator.value = value
    
    def set_log_file(self, log_file):
        self.log_file.value = log_file

    def set_path(self, new_path):
        self.path.value = new_path

    def get_path(self):
        return self.path.value
    
    def set_to_loading(self):
        self.to_loading.value = True

    def set_to_scanning(self):
        self.to_scanning.value = True

    def reset_to_loading(self):
        self.to_loading.value = False

    def reset_to_scanning(self):    
        self.to_scanning.value = False

    def set_live_view_image(self, np_array):
        
        if np_array.shape != self.IMAGE_SHAPE:
            # if only the first two dimension match
            if np_array.shape[:2] == self.IMAGE_SHAPE[:2]:
                 # copy the single changle to all channels
                np_array = np.repeat(np_array[:, :, np.newaxis], 3, axis=2)
            else:
                raise ValueError(f"Input array must have shape {self.IMAGE_SHAPE}, instead got {np_array.shape}")
        
        with self.live_view_image_lock:
            # Create a numpy array backed by shared memory
            shared_array = np.frombuffer(self.live_view_image_array, dtype=np.float32).reshape(self.IMAGE_SHAPE)
            # Copy data efficiently
            np.copyto(shared_array, np_array)

    def get_live_view_image(self):
        with self.live_view_image_lock:
            # Create a copy of the shared array to avoid issues with threading
            return np.frombuffer(self.live_view_image_array, dtype=np.float32).reshape(self.IMAGE_SHAPE).copy()

    def set_channels_list(self, channels_list):
        self.live_channels_list[:] = channels_list

    def set_channel_selected(self, channel_idx):
        self.live_channel_selected.value = channel_idx

    def setup_process_logger(self):
        logger = logging.getLogger(f"{__name__}_{mp.current_process().name}")
        logger.setLevel(logging.INFO)

        # Remove all existing handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Get the log file path from the shared config
        log_file_path = os.path.join(self.log_file.value, "log.txt")

        # Create a RotatingFileHandler with 'a' mode for appending
        file_handler = RotatingFileHandler(log_file_path, mode='a', maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_handler)

        # Redirect stdout and stderr to the logger
        #sys.stdout = StreamToLogger(logger, logging.INFO)
        #sys.stderr = StreamToLogger(logger, logging.ERROR)

        return logger
        
        
class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
    def flush(self):
        pass