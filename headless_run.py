import os
import numpy as np
import cv2
from pathlib import Path
import argparse
from utils import generate_dpc, draw_spot_bounding_boxes
from interactive_m2unet_inference import M2UnetInteractiveModel as m2u
from utils import remove_background, resize_image_cp, detect_spots, prune_blobs, settings
from model import ResNet, run_model
from tqdm import tqdm
import torch

# Constants from original code
MINIMUM_SCORE_THRESHOLD = 0.5
MAX_SPOTS_THRESHOLD = 10000

def process_dpc(left_half, right_half):
    """Generate DPC from left and right halves"""
    left_half = left_half.astype(np.float16)/255
    right_half = right_half.astype(np.float16)/255
    return generate_dpc(left_half, right_half, use_gpu=False)

def process_segmentation(dpc_image, model):
    """Perform segmentation on DPC image"""
    # convert dpc to np.int8
    dpc_image = (dpc_image*255).astype(np.uint8)
    result = model.predict_on_images(dpc_image)
    threshold = 0.5
    segmentation_mask = (255*(result > threshold)).astype(np.uint8)
    from scipy.ndimage import label
    _, n_cells = label(segmentation_mask)
    return segmentation_mask, n_cells

def process_fluorescent(fluorescent_image):
    """Process fluorescent image and detect spots"""
    I_fluorescence_bg_removed = remove_background(fluorescent_image, return_gpu_image=False)
    
    spot_list = detect_spots(resize_image_cp(I_fluorescence_bg_removed,
                                           downsize_factor=settings['spot_detection_downsize_factor']),
                                           thresh=settings['spot_detection_threshold'])

    if len(spot_list) > MAX_SPOTS_THRESHOLD:
        spot_list = spot_list[:MAX_SPOTS_THRESHOLD]
    
    if len(spot_list) > 0:
        spot_list = prune_blobs(spot_list)

    spot_list = spot_list*settings['spot_detection_downsize_factor']
    
    return spot_list

def process_classification(dpc_image, fluorescent_image, spot_list, models):
    """Classify detected spots"""
    from utils import get_spot_images_from_fov, seg_spot_filter_one_fov
    
    if len(spot_list) > 0:
        # Get segmentation mask for filtering
        seg_model = m2u(pretrained_model='checkpoint/m2unet_model_flat_erode1_wdecay5_smallbatch/model_4000_11.pth', 
                       use_trt=False)
        segmentation_mask, _ = process_segmentation(dpc_image, seg_model)
        
        # Filter spots using segmentation mask
        filtered_spots = seg_spot_filter_one_fov(segmentation_mask, spot_list)
        
        if len(filtered_spots) > 0:
            cropped_images = get_spot_images_from_fov(fluorescent_image, dpc_image, filtered_spots, r=15)
            cropped_images = cropped_images.transpose(0, 3, 1, 2)

            scores = run_model(models[0], 'cuda', cropped_images, 1024)[:,1]
        else:
            cropped_images = np.array([])
            scores = np.array([])
            filtered_spots = np.array([])
    else:
        cropped_images = np.array([])
        scores = np.array([])
        filtered_spots = np.array([])
    
    return cropped_images, scores, filtered_spots, spot_list

def calculate_stats(fov_count,rbc_count, positives_count):
    """Calculate statistics based on RBC and positives count"""
    parasite_per_ul = round(positives_count * (5000000 / (rbc_count + 1)), 2)
    parasitemia_percentage = round(positives_count / (rbc_count + 1) * 100, 2)
    
    stats = (f"FoVs: {fov_count} | RBCs Count: {rbc_count:,} | "
            f"Positives: {positives_count:,} | "
            f"Parasites / μl: {int(parasite_per_ul):,} | "
            f"Parasitemia: {parasitemia_percentage:.2f}%")
    
    return stats, parasite_per_ul, parasitemia_percentage

def save_results(output_dir, fov_id, dpc_image, fluorescent_image, left_half=None, right_half=None,
                segmentation_mask=None, cropped_images=None, scores=None, 
                filtered_spots=None, spot_list=None, save_npy=False):
    """Save all results to the output directory"""
    
    # Save fluorescent image
    cv2.imwrite(os.path.join(output_dir, f"{fov_id}_fluorescent.bmp"), fluorescent_image)
    
    # Save DPC
    if save_npy:
        np.save(os.path.join(output_dir, f"{fov_id}_dpc.npy"), dpc_image)
    else:
        cv2.imwrite(os.path.join(output_dir, f"{fov_id}_dpc.bmp"), 
                    (dpc_image*255).astype(np.uint8))
    
    # Save left/right halves if they exist
    if left_half is not None and right_half is not None:
        if save_npy:
            np.save(os.path.join(output_dir, f"{fov_id}_left_half.npy"), left_half)
            np.save(os.path.join(output_dir, f"{fov_id}_right_half.npy"), right_half)
        else:
            cv2.imwrite(os.path.join(output_dir, f"{fov_id}_left_half.bmp"), left_half)
            cv2.imwrite(os.path.join(output_dir, f"{fov_id}_right_half.bmp"), right_half)
    
    # Save segmentation mask if it exists
    if segmentation_mask is not None:
        cv2.imwrite(os.path.join(output_dir, f"{fov_id}_segmentation_map.bmp"), 
                    segmentation_mask)
    
    # Save cropped images and scores if they exist
    if cropped_images is not None and len(cropped_images) > 0:
        np.save(os.path.join(output_dir, f"{fov_id}_cropped.npy"), cropped_images)
        np.save(os.path.join(output_dir, f"{fov_id}_scores.npy"), scores)
    
    # Save spot lists if they exist
    if filtered_spots is not None:
        np.save(os.path.join(output_dir, f"{fov_id}_filtered_spots.npy"), filtered_spots)
    if spot_list is not None:
        np.save(os.path.join(output_dir, f"{fov_id}_spot_list.npy"), spot_list)
    
    # Create and save overlay image with bounding boxes
    if dpc_image is not None and fluorescent_image is not None and spot_list is not None:
        overlay_img = draw_spot_bounding_boxes(
            fluorescent_image,
            dpc_image,
            spot_list,
            filtered_spots if filtered_spots is not None else np.array([]),
            spot_list2_scores=scores if scores is not None else np.array([])
        )
        cv2.imwrite(os.path.join(output_dir, f"{fov_id}_overlay_bb.bmp"), overlay_img)

def main():
    parser = argparse.ArgumentParser(description='Process saved microscope images')
    parser.add_argument('input_dir', help='Input directory containing patient folders')
    parser.add_argument('output_dir', help='Output directory for processed results')
    parser.add_argument('--save-npy', action='store_true', help='Save images as NPY files instead of BMP')
    args = parser.parse_args()

    # set save_npy to false by default
    args.save_npy = False

    # Initialize classification models
    class_model1 = ResNet('resnet18').to(device='cuda')
    class_model1.load_state_dict(torch.load('./saved_data/reprocess/brown_v1_30x_brown/best.pt'))
    class_model1.eval()
    

    # Process each patient directory
    for patient_dir in Path(args.input_dir).iterdir():
        if not patient_dir.is_dir():
            continue
            
        print(f"Processing patient directory: {patient_dir}")
        
        # Create output directory for this patient, add a timestamp to the name
        patient_output_dir = Path(args.output_dir) / f"{patient_dir.name}"
        # if patient_output_dir exists, skip
        if patient_output_dir.exists():
            print(f"Skipping patient directory: {patient_dir}")
            continue
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        
        total_rbc_count = 0
        total_positives = 0
        fov_count = 0
        
        # Process each FOV (using fluorescent images as reference)
        def get_fov_number(filename):
            # Extract the number from filenames like "fov_1_fluorescent.bmp" or "1_fluorescent.bmp"
            name = filename.stem.replace("_fluorescent", "")
            try:
                # Extract just the numeric part and convert to integer
                num = int(''.join(filter(str.isdigit, name)))
                return num
            except ValueError:
                return float('inf')  # Put non-numeric names at the end
                
        for fov_file in tqdm(sorted(patient_dir.glob("*_fluorescent.*"), key=get_fov_number)):
            fov_id = fov_file.stem.replace("_fluorescent", "")
            #print(f"Processing FOV: {fov_id}")
            
            # Load fluorescent image
            fluorescent = cv2.imread(str(fov_file))
            
            # Try to load DPC first, if not available generate from left/right halves
            dpc_path = patient_dir / f"{fov_id}_dpc.npy"
            if dpc_path.exists():
                dpc_image = np.load(str(dpc_path))
                left_half = right_half = None  # DPC exists, don't need these
            else:
                dpc_path = patient_dir / f"{fov_id}_dpc.bmp"
                if dpc_path.exists():
                    dpc_image = cv2.imread(str(dpc_path), cv2.IMREAD_GRAYSCALE).astype(np.float16)/255
                    left_half = right_half = None
                else:
                    # Load and process left/right halves
                    left_half = cv2.imread(str(patient_dir / f"{fov_id}_left_half.bmp"), 
                                         cv2.IMREAD_GRAYSCALE)
                    right_half = cv2.imread(str(patient_dir / f"{fov_id}_right_half.bmp"), 
                                          cv2.IMREAD_GRAYSCALE)
                    dpc_image = process_dpc(left_half, right_half)
            
            # Process images
            seg_model = m2u(pretrained_model='checkpoint/m2unet_model_flat_erode1_wdecay5_smallbatch/model_4000_11.pth', 
                           use_trt=False)
            segmentation_mask, rbc_count = process_segmentation(dpc_image, seg_model)
            spot_list = process_fluorescent(fluorescent)
            cropped_images, scores, filtered_spots, all_spots = process_classification(
                dpc_image, 
                fluorescent.astype(np.float16)/255, 
                spot_list, 
                [class_model1]
            )

            I_combined = draw_spot_bounding_boxes(np.array(fluorescent), np.array(dpc_image), all_spots, filtered_spots,spot_list2_scores = scores)
            cv2.imwrite(os.path.join(patient_output_dir, f"{fov_id}_overlay_bb.bmp"), I_combined)
            
            # Calculate positives
            positives_count = sum(1 for score in scores if score >= MINIMUM_SCORE_THRESHOLD)
            
            # Update totals
            total_rbc_count += rbc_count
            total_positives += positives_count
            fov_count += 1
            
            # Save all results
            save_results(
                patient_output_dir, 
                fov_id, 
                dpc_image,
                fluorescent,
                left_half,
                right_half,
                segmentation_mask,
                cropped_images,
                scores,
                filtered_spots,
                all_spots,
                args.save_npy
            )
            
            # Save RBC count for this FOV
            with open(patient_output_dir / "rbc_counts.csv", "a") as f:
                f.write(f"{fov_id},{rbc_count}\n")
        
        # Calculate and save final stats for patient
        final_stats, parasite_per_ul, parasitemia = calculate_stats(
            fov_count, total_rbc_count, total_positives)
        
        with open(patient_output_dir / "stats.txt", "w", encoding='utf-8') as f:
            f.write(final_stats)
        
        print(f"Completed processing patient {patient_dir.name}")
        print(f"Final stats: {final_stats}")

if __name__ == "__main__":
    main()
