"""
General multi-FOV acquisition mode (independent of the malaria detection pipeline).

Runs inside the existing image_acquisition process so it shares the microscope
object. Triggered by SharedConfig.ga_active.value; clears the flag when done.
"""
import os
import json
import datetime
import time

import cv2
import numpy as np

from simulation import crop_image
from control.utils import generate_scan_grid, interpolate_focus


# Sensor + system constants. See analyses/updates-to-octopi-malaria-gui/optics-and-step-size.md.
SENSOR_PIXEL_SIZE_UM = 1.85       # Daheng MER2-1220-32U3C (Sony IMX226)
CROPPED_IMAGE_PX = 2800           # simulation.crop_image output
SYSTEM_TUBE_LENS_MM = 50.0        # Hikrobot 50mm f/2.4

# Per-objective presets. design_tube_lens_mm = the focal length the objective is
# designed for (Olympus 180, Nikon/Mitutoyo 200, Zeiss 165, Leica 200). To add or
# edit objectives, modify this dict directly. The dropdown order follows insertion
# order.
OBJECTIVE_PRESETS = {
    '20x (Olympus UPlanFL N 0.50)':  {'magnification': 20,  'design_tube_lens_mm': 180, 'NA': 0.50},
    '40x (LWD Plan 0.6)':            {'magnification': 40,  'design_tube_lens_mm': 180, 'NA': 0.60},
    '50x (LMPlan 0.70)':             {'magnification': 50,  'design_tube_lens_mm': 180, 'NA': 0.70},
    '60x (Nikon Plan Apo 0.95)':     {'magnification': 60,  'design_tube_lens_mm': 200, 'NA': 0.95},
    '100x (generic)':                {'magnification': 100, 'design_tube_lens_mm': 200, 'NA': 1.40},
}


def compute_fov_step_mm(objective_name: str, overlap_pct: float) -> float:
    """FOV size (minus overlap), in mm, for the given objective and overlap %.

    effective_mag = nominal_mag * (system_tube_lens / design_tube_lens)
    pixel_at_sample = sensor_pixel_um / effective_mag
    fov_mm = cropped_px * pixel_at_sample / 1000
    """
    preset = OBJECTIVE_PRESETS.get(objective_name)
    if preset is None:
        # Fallback for any string the UI might still hold (e.g., legacy '20x')
        preset = next(iter(OBJECTIVE_PRESETS.values()))
    effective_mag = preset['magnification'] * (SYSTEM_TUBE_LENS_MM / preset['design_tube_lens_mm'])
    pixel_size_um = SENSOR_PIXEL_SIZE_UM / effective_mag
    fov_mm = CROPPED_IMAGE_PX * pixel_size_um / 1000.0
    return fov_mm * (1.0 - overlap_pct / 100.0)


def _safe_channel_name(channel: str) -> str:
    return channel.replace(' ', '_').replace('/', '_')


def _process_image_for_save(image, channel: str):
    """Per-channel processing before save, matching the malaria pipeline's conventions:
       - BF left/right half  -> middle channel only -> grayscale BMP
       - BF full / low NA / fluorescence -> RGB array, flipped to BGR for cv2.imwrite
       Camera returns RGB (control/camera.py read_frame -> convert("RGB")), so
       3-channel saves must be flipped or external viewers and cv2.imread see
       blue/red swapped (the orange-instead-of-blue issue).
    """
    if image.ndim == 3 and ('left half' in channel.lower() or 'right half' in channel.lower()):
        image = image[:, :, 1]
    elif image.ndim == 3 and image.shape[2] == 3:
        image = image[:, :, ::-1]
    return crop_image(image)


def run_general_acquisition(microscope, shared_config, shutdown_event, logger):
    """Execute a single general acquisition run. Returns when scan is done or aborted."""
    cfg = shared_config

    nx = max(1, int(cfg.ga_nx.value))
    ny = max(1, int(cfg.ga_ny.value))
    objective = cfg.ga_objective.value
    overlap = float(cfg.ga_overlap_pct.value)
    af_mode = cfg.ga_af_mode.value
    af_every_n = max(1, int(cfg.ga_af_every_n.value))
    af_start = float(cfg.ga_af_start_mm.value)
    af_end = float(cfg.ga_af_end_mm.value)
    channels = [c for c in cfg.ga_channels]
    save_name = cfg.ga_save_name.value or 'acq'

    if not channels:
        logger.error("General acquisition: no channels selected, aborting")
        cfg.ga_active.value = False
        return

    step_mm = compute_fov_step_mm(objective, overlap)
    logger.info(f"General acquisition: {nx}x{ny} FOVs, step {step_mm*1000:.1f} um "
                f"({objective}, {overlap:.0f}% overlap), AF={af_mode}, channels={channels}")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('saved_data', 'general_acq', f"{save_name}_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    cfg.ga_save_path.value = base_dir

    offset_x_mm = microscope.get_x()
    offset_y_mm = microscope.get_y()
    offset_z_mm = microscope.get_z()

    scan_grid = generate_scan_grid(step_mm, step_mm, nx, ny, offset_x_mm, offset_y_mm, S_scan=True)

    # Z map: per-FOV target Z
    z_map = [offset_z_mm] * len(scan_grid)
    focus_map_points = []

    cfg.ga_running.value = True
    try:
        if af_mode == 'focus_map':
            nx_focus = int(np.ceil(nx / 10) + 1)
            ny_focus = int(np.ceil(ny / 10) + 1)
            fx = offset_x_mm + np.linspace(0, (nx - 1) * step_mm, nx_focus)
            fy = offset_y_mm + np.linspace(0, (ny - 1) * step_mm, ny_focus)
            microscope.set_channel("BF LED matrix left half")
            z_running = offset_z_mm
            for i, yi in enumerate(fy):
                if shutdown_event.is_set() or not cfg.ga_active.value:
                    return
                microscope.move_y_to(yi)
                x_iter = fx if i % 2 == 0 else fx[::-1]
                for xi in x_iter:
                    if shutdown_event.is_set() or not cfg.ga_active.value:
                        return
                    microscope.move_x_to(xi)
                    z_focus, _ = microscope.run_autofocus(
                        step_size_mm=[0.01, 0.001],
                        start_z_mm=af_start,
                        end_z_mm=af_end,
                    )
                    focus_map_points.append((float(xi), float(yi), float(z_focus)))
                    z_running = z_focus
            z_map = interpolate_focus(scan_grid, focus_map_points)

        # Scan loop
        prev_x, prev_y = None, None
        last_af_z = None
        for i, ((x, y), z) in enumerate(zip(scan_grid, z_map)):
            if shutdown_event.is_set() or not cfg.ga_active.value:
                logger.info(f"General acquisition aborted at FOV {i+1}/{len(scan_grid)}")
                return

            if x != prev_x:
                microscope.move_x_to(x)
                prev_x = x
            if y != prev_y:
                microscope.move_y_to(y)
                prev_y = y

            if af_mode == 'every_n' and (i % af_every_n == 0):
                microscope.set_channel("BF LED matrix left half")
                start_mm = last_af_z - (af_end - af_start) / 2 if last_af_z is not None else af_start
                end_mm = last_af_z + (af_end - af_start) / 2 if last_af_z is not None else af_end
                z_focus, _ = microscope.run_autofocus(
                    step_size_mm=[0.01, 0.001],
                    start_z_mm=start_mm,
                    end_z_mm=end_mm,
                )
                last_af_z = z_focus
                microscope.move_z_to(z_focus)
            elif af_mode == 'every_n' and last_af_z is not None:
                microscope.move_z_to(last_af_z)
            else:
                microscope.move_z_to(z)

            fov_id = f"{i+1}"
            logger.info(f"General acq FOV {fov_id} at x={x:.3f} y={y:.3f} z={microscope.get_z():.3f}")

            for channel in channels:
                microscope.set_channel(channel)
                image = microscope.acquire_image()
                # Live view preview gets the raw cropped RGB (matches LIVE button behavior)
                cfg.set_live_view_image(crop_image(image))
                # Save path applies channel-specific processing (BGR flip, middle channel, etc.)
                processed = _process_image_for_save(image, channel)
                filename = f"{fov_id}_{_safe_channel_name(channel)}.bmp"
                cv2.imwrite(os.path.join(base_dir, filename), processed)

            cfg.live_x.value = microscope.get_x()
            cfg.live_y.value = microscope.get_y()
            cfg.live_z.value = microscope.get_z()

        # Write metadata
        metadata = {
            'timestamp': timestamp,
            'objective': objective,
            'nx': nx,
            'ny': ny,
            'overlap_pct': overlap,
            'step_mm': step_mm,
            'af_mode': af_mode,
            'af_every_n': af_every_n if af_mode == 'every_n' else None,
            'af_start_mm': af_start,
            'af_end_mm': af_end,
            'channels': channels,
            'origin_x_mm': offset_x_mm,
            'origin_y_mm': offset_y_mm,
            'focus_map_points': focus_map_points if af_mode == 'focus_map' else [],
        }
        with open(os.path.join(base_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"General acquisition complete: {base_dir}")

    finally:
        cfg.ga_running.value = False
        cfg.ga_active.value = False
