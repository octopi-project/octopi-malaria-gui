from microscope import Microscope
import control.utils as utils
import numpy as np
import pandas as pd
import time

INIT_FOCUS_RANGE_START_MM = 3
INIT_FOCUS_RANGE_END_MM = 7
SCAN_FOCUS_SEARCH_RANGE_MM = 0.1


microscope = Microscope(is_simulation=True)

microscope.camera.start_streaming()
microscope.camera.set_software_triggered_acquisition()
microscope.camera.disable_callback()

microscope.home_xyz()
microscope.to_scanning_position()

microscope.move_x_to(20)
microscope.move_y_to(20)

microscope.run_autofocus(step_size_mm = [0.1, 0.01, 0.0015], start_z_mm = INIT_FOCUS_RANGE_START_MM, end_z_mm = INIT_FOCUS_RANGE_END_MM)

'''
# for i in range(5):
#     for channel in ["BF LED matrix left half","BF LED matrix right half","Fluorescence 405 nm Ex"]:
#         microscope.set_channel(channel)
#         image = microscope.acquire_image()
#         print(image)
'''

# scan settings
dx_mm = 0.9
dy_mm = 0.9
Nx = 5
Ny = 5
Nx_focus = 2
Ny_focus = 2
offset_x_mm = microscope.get_x()
offset_y_mm = microscope.get_y()
offset_z_mm = microscope.get_z()

# generate scan grid
scan_grid = utils.generate_scan_grid(dx_mm, dy_mm, Nx, Ny, offset_x_mm, offset_y_mm, S_scan=True)

# generate focus map
x = offset_x_mm + np.linspace(0, (Nx - 1) * dx_mm, Nx_focus)
y = offset_y_mm + np.linspace(0, (Ny - 1) * dy_mm, Ny_focus)
focus_map = []
microscope.set_channel("BF LED matrix left half")
for yi in y:
    microscope.move_y_to(yi)
    for xi in x:
        microscope.move_x_to(xi)
        z_focus = microscope.run_autofocus(step_size_mm = [0.01, 0.0015], start_z_mm = offset_z_mm - SCAN_FOCUS_SEARCH_RANGE_MM/2, end_z_mm = offset_z_mm + SCAN_FOCUS_SEARCH_RANGE_MM/2)
        focus_map.append((xi, yi, z_focus))
        offset_z_mm = z_focus
print(focus_map)
z_map = utils.interpolate_focus(scan_grid, focus_map)

# scan using focus map
prev_x, prev_y = None, None
coordinates = []
for i, ((x, y), z) in enumerate(zip(scan_grid, z_map)):
    if x != prev_x:
        microscope.move_x_to(x)
        prev_x = x
    if y != prev_y:
        microscope.move_y_to(y)
        prev_y = y
    microscope.move_z_to(z)
    coordinates.append({
        'i': i,
        'x_mm': x,
        'y_mm': y,
        'z_mm': z,
        'time': time.time()
    })
    for channel in ["BF LED matrix left half","BF LED matrix right half","Fluorescence 405 nm Ex"]:
        microscope.set_channel(channel)
        image = microscope.acquire_image()
        print(image)

df = pd.DataFrame(coordinates)
df.to_csv('coordinates.csv', index=False)

# microscope.start_live()
# time.sleep(2)
# microscope.stop_live()

microscope.close()
