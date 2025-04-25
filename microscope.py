import time
import os
import sys
from control._def import *
import control.core as core
import control.camera as camera
import control.microcontroller as microcontroller
import control.serial_peripherals as serial_peripherals
import control.utils as utils

import numpy as np
from scipy.interpolate import griddata

from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *

from simulation import crop_image

class Microscope(QObject):

    def __init__(self, is_simulation=False):
        self.is_simulation = is_simulation
        self.initialize_components()
        self.setup_connections()

    def initialize_components(self):
        # Initialize camera
        self.initialize_camera()

        # Initialize microcontroller
        self.initialize_microcontroller()

        # Initialize core components
        self.initialize_core_components()

        # Initialize peripherals
        self.initialize_peripherals()

        # Set up connections
        self.setup_connections()

    def initialize_camera(self):
        if self.is_simulation:
            self.camera = camera.Camera_Simulation(rotate_image_angle=ROTATE_IMAGE_ANGLE, flip_image=FLIP_IMAGE)
        else:
            sn_camera_main = camera.get_sn_by_model(MAIN_CAMERA_MODEL)
            self.camera = camera.Camera(sn=sn_camera_main, rotate_image_angle=ROTATE_IMAGE_ANGLE, flip_image=FLIP_IMAGE)
        
        self.camera.open()
        self.camera.set_pixel_format(DEFAULT_PIXEL_FORMAT)
        self.camera.set_software_triggered_acquisition()

    def initialize_microcontroller(self):
        if self.is_simulation:
            self.microcontroller = microcontroller.Microcontroller_Simulation()
        else:
            self.microcontroller = microcontroller.Microcontroller(version=CONTROLLER_VERSION, sn=CONTROLLER_SN)
        
        self.microcontroller.reset()
        time.sleep(0.5)
        self.microcontroller.initialize_drivers()
        time.sleep(0.5)
        self.microcontroller.configure_actuators()

        self.home_x_and_y_separately = False

    def initialize_core_components(self):
        self.configurationManager = core.ConfigurationManager(filename='./config/channel_configurations.xml')
        self.objectiveStore = core.ObjectiveStore()
        self.streamHandler = core.StreamHandler(display_resolution_scaling=DEFAULT_DISPLAY_CROP/100)
        self.liveController = core.LiveController(self.camera, self.microcontroller, self.configurationManager, self)
        self.navigationController = core.NavigationController(self.microcontroller, self.objectiveStore)
        self.autofocusController = core.AutoFocusController(self.camera, self.navigationController, self.liveController)
        self.slidePositionController = core.SlidePositionController(self.navigationController,self.liveController)

    def initialize_peripherals(self):
        if USE_ZABER_EMISSION_FILTER_WHEEL:
            self.emission_filter_wheel = serial_peripherals.FilterController(FILTER_CONTROLLER_SERIAL_NUMBER, 115200, 8, serial.PARITY_NONE, serial.STOPBITS_ONE)
            self.emission_filter_wheel.start_homing()
        elif USE_OPTOSPIN_EMISSION_FILTER_WHEEL:
            self.emission_filter_wheel = serial_peripherals.Optospin(SN=FILTER_CONTROLLER_SERIAL_NUMBER)
            self.emission_filter_wheel.set_speed(OPTOSPIN_EMISSION_FILTER_WHEEL_SPEED_HZ)

    def setup_connections(self):
        self.streamHandler.signal_new_frame_received.connect(self.liveController.on_new_frame)
        self.camera.set_callback(self.streamHandler.on_new_frame)
        self.camera.enable_callback()

    def set_channel(self,channel):
        self.liveController.set_channel(channel)

    def acquire_image(self):
        if self.liveController.trigger_mode == TriggerMode.SOFTWARE:
            self.liveController.turn_on_illumination()
            self._wait_till_operation_is_completed()
            self.camera.send_trigger()
        elif self.liveController.trigger_mode == TriggerMode.HARDWARE:
            self.microcontroller.send_hardware_trigger(control_illumination=True,illumination_on_time_us=self.camera.exposure_time*1000)
        
        image = self.camera.read_frame()

        if image is None:
            print('self.camera.read_frame() returned None')
        
        # tunr of the illumination if using software trigger
        if self.liveController.trigger_mode == TriggerMode.SOFTWARE:
            self.liveController.turn_off_illumination()

        return image

    def home_xyz(self):
        # retract the objective
        self.navigationController.home_z()
        # wait for the operation to finish
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print('z homing timeout, the program will exit')
                sys.exit(1)
        print('objective retracted')

         # set encoder arguments
        # set axis pid control enable
        # only ENABLE_PID_X and HAS_ENCODER_X are both enable, can be enable to PID
        if HAS_ENCODER_X == True:
            self.navigationController.set_axis_PID_arguments(0, PID_P_X, PID_I_X, PID_D_X)
            self.navigationController.configure_encoder(0, (SCREW_PITCH_X_MM * 1000) / ENCODER_RESOLUTION_UM_X, ENCODER_FLIP_DIR_X)
            self.navigationController.set_pid_control_enable(0, ENABLE_PID_X)
        if HAS_ENCODER_Y == True:
            self.navigationController.set_axis_PID_arguments(1, PID_P_Y, PID_I_Y, PID_D_Y)
            self.navigationController.configure_encoder(1, (SCREW_PITCH_Y_MM * 1000) / ENCODER_RESOLUTION_UM_Y, ENCODER_FLIP_DIR_Y)
            self.navigationController.set_pid_control_enable(1, ENABLE_PID_Y)
        if HAS_ENCODER_Z == True:
            self.navigationController.set_axis_PID_arguments(2, PID_P_Z, PID_I_Z, PID_D_Z)
            self.navigationController.configure_encoder(2, (SCREW_PITCH_Z_MM * 1000) / ENCODER_RESOLUTION_UM_Z, ENCODER_FLIP_DIR_Z)
            self.navigationController.set_pid_control_enable(2, ENABLE_PID_Z)

        time.sleep(0.5)

        # homing, set zero and set software limit
        self.navigationController.set_x_limit_pos_mm(100)
        self.navigationController.set_x_limit_neg_mm(-100)
        self.navigationController.set_y_limit_pos_mm(100)
        self.navigationController.set_y_limit_neg_mm(-100)
        print('start homing')
        # self.slidePositionController.move_to_slide_scanning_position()
        # while self.slidePositionController.slide_scanning_position_reached == False:
        #     time.sleep(0.005)
        self.to_scanning_position()
        print('homing finished')
        self.navigationController.set_x_limit_pos_mm(SOFTWARE_POS_LIMIT.X_POSITIVE)
        self.navigationController.set_x_limit_neg_mm(SOFTWARE_POS_LIMIT.X_NEGATIVE)
        self.navigationController.set_y_limit_pos_mm(SOFTWARE_POS_LIMIT.Y_POSITIVE)
        self.navigationController.set_y_limit_neg_mm(SOFTWARE_POS_LIMIT.Y_NEGATIVE)
        print('Check the limit of z !!!!!!!!!!!!!!')
        print("pos", SOFTWARE_POS_LIMIT.Z_POSITIVE)
        print("neg", SOFTWARE_POS_LIMIT.Z_NEGATIVE)

    def move_x(self,distance,blocking=True):
        self.navigationController.move_x(distance)
        if blocking:
            self._wait_till_operation_is_completed()

    def move_y(self,distance,blocking=True):
        self.navigationController.move_y(distance)
        if blocking:
            self._wait_till_operation_is_completed()

    def move_x_to(self,position,blocking=True):
        self.navigationController.move_x_to(position)
        if blocking:
            self._wait_till_operation_is_completed()

    def move_y_to(self,position,blocking=True):
        self.navigationController.move_y_to(position)
        if blocking:
            self._wait_till_operation_is_completed()

    def get_x(self):
        return self.navigationController.x_pos_mm

    def get_y(self):
        return self.navigationController.y_pos_mm

    def get_z(self):
        return self.navigationController.z_pos_mm

    def move_z_to(self,z_mm,blocking=True):
        clear_backlash = True if (z_mm < self.navigationController.z_pos_mm and self.navigationController.get_pid_control_flag(2)==False) else False
        # clear backlash if moving backward in open loop mode
        self.navigationController.move_z_to(z_mm)
        if blocking:
            self._wait_till_operation_is_completed()
            if clear_backlash:
                _usteps_to_clear_backlash = 160
                self.navigationController.move_z_usteps(-_usteps_to_clear_backlash)
                self._wait_till_operation_is_completed()
                self.navigationController.move_z_usteps(_usteps_to_clear_backlash)
                self._wait_till_operation_is_completed()

    # def to_loading_position(self):
    #     self.slidePositionController.move_to_slide_loading_position()

    # def to_scanning_position(self):
    #     self.slidePositionController.move_to_slide_scanning_position()

    def to_loading_position(self):
        # retract z
        timestamp_start = time.time()
        self.slidePositionController.z_pos = self.navigationController.z_pos # zpos at the beginning of the scan
        self.navigationController.move_z_to(OBJECTIVE_RETRACTED_POS_MM)
        self._wait_till_operation_is_completed()
        print('z retracted')
        self.slidePositionController.objective_retracted = True
        
        # move to position
        # for well plate
        if self.slidePositionController.is_for_wellplate:
            # reset limits
            self.navigationController.set_x_limit_pos_mm(100)
            self.navigationController.set_x_limit_neg_mm(-100)
            self.navigationController.set_y_limit_pos_mm(100)
            self.navigationController.set_y_limit_neg_mm(-100)
            # home for the first time
            if self.slidePositionController.homing_done == False:
                print('running homing first')
                timestamp_start = time.time()
                # x needs to be at > + 20 mm when homing y
                self.navigationController.move_x(20)
                self._wait_till_operation_is_completed()
                # home y
                self.navigationController.home_y()
                self._wait_till_operation_is_completed()
                self.navigationController.zero_y()
                # home x
                self.navigationController.home_x()
                self._wait_till_operation_is_completed()
                self.navigationController.zero_x()
                self.slidePositionController.homing_done = True
            # homing done previously
            else:
                timestamp_start = time.time()
                self.navigationController.move_x_to(20)
                self._wait_till_operation_is_completed()
                self.navigationController.move_y_to(SLIDE_POSITION.LOADING_Y_MM)
                self._wait_till_operation_is_completed()
                self.navigationController.move_x_to(SLIDE_POSITION.LOADING_X_MM)
                self._wait_till_operation_is_completed()
            # set limits again
            self.navigationController.set_x_limit_pos_mm(SOFTWARE_POS_LIMIT.X_POSITIVE)
            self.navigationController.set_x_limit_neg_mm(SOFTWARE_POS_LIMIT.X_NEGATIVE)
            self.navigationController.set_y_limit_pos_mm(SOFTWARE_POS_LIMIT.Y_POSITIVE)
            self.navigationController.set_y_limit_neg_mm(SOFTWARE_POS_LIMIT.Y_NEGATIVE)
        else:
            # for glass slide
            if self.slidePositionController.homing_done == False or SLIDE_POTISION_SWITCHING_HOME_EVERYTIME:
                if self.home_x_and_y_separately:
                    timestamp_start = time.time()
                    self.navigationController.home_x()
                    self._wait_till_operation_is_completed()
                    self.navigationController.zero_x()
                    self.navigationController.move_x(SLIDE_POSITION.LOADING_X_MM)
                    self._wait_till_operation_is_completed()
                    self.navigationController.home_y()
                    self._wait_till_operation_is_completed()
                    self.navigationController.zero_y()
                    self.navigationController.move_y(SLIDE_POSITION.LOADING_Y_MM)
                    self._wait_till_operation_is_completed()
                else:
                    timestamp_start = time.time()
                    self.navigationController.home_xy()
                    self._wait_till_operation_is_completed()
                    self.navigationController.zero_x()
                    self.navigationController.zero_y()
                    self.navigationController.move_x(SLIDE_POSITION.LOADING_X_MM)
                    self._wait_till_operation_is_completed()
                    self.navigationController.move_y(SLIDE_POSITION.LOADING_Y_MM)
                    self._wait_till_operation_is_completed()
                self.slidePositionController.homing_done = True
            else:
                timestamp_start = time.time()
                self.navigationController.move_y(SLIDE_POSITION.LOADING_Y_MM-self.navigationController.y_pos_mm)
                self._wait_till_operation_is_completed()
                self.navigationController.move_x(SLIDE_POSITION.LOADING_X_MM-self.navigationController.x_pos_mm)
                self._wait_till_operation_is_completed()

    def to_scanning_position(self):
        # move to position
        # for well plate
        if self.slidePositionController.is_for_wellplate:
            # home for the first time
            if self.slidePositionController.homing_done == False:
                timestamp_start = time.time()

                # x needs to be at > + 20 mm when homing y
                self.navigationController.move_x(20)
                self._wait_till_operation_is_completed()
                # home y
                self.navigationController.home_y()
                self._wait_till_operation_is_completed()
                self.navigationController.zero_y()
                # home x
                self.navigationController.home_x()
                self._wait_till_operation_is_completed()
                self.navigationController.zero_x()
                self.slidePositionController.homing_done = True
                # move to scanning position
                self.navigationController.move_x_to(SLIDE_POSITION.SCANNING_X_MM)
                self._wait_till_operation_is_completed()

                self.navigationController.move_y_to(SLIDE_POSITION.SCANNING_Y_MM)
                self._wait_till_operation_is_completed()
                   
            else:
                timestamp_start = time.time()
                self.navigationController.move_x_to(SLIDE_POSITION.SCANNING_X_MM)
                self._wait_till_operation_is_completed()
                self.navigationController.move_y_to(SLIDE_POSITION.SCANNING_Y_MM)
                self._wait_till_operation_is_completed()
        else:
            if self.slidePositionController.homing_done == False or SLIDE_POTISION_SWITCHING_HOME_EVERYTIME:
                if self.home_x_and_y_separately:
                    timestamp_start = time.time()
                    self.navigationController.home_y()
                    self._wait_till_operation_is_completed()
                    self.navigationController.zero_y()
                    self.navigationController.move_y(SLIDE_POSITION.SCANNING_Y_MM)
                    self._wait_till_operation_is_completed()
                    self.navigationController.home_x()
                    self._wait_till_operation_is_completed()
                    self.navigationController.zero_x()
                    self.navigationController.move_x(SLIDE_POSITION.SCANNING_X_MM)
                    self._wait_till_operation_is_completed()
                else:
                    timestamp_start = time.time()
                    self.navigationController.home_xy()
                    self._wait_till_operation_is_completed()
                    self.navigationController.zero_x()
                    self.navigationController.zero_y()
                    self.navigationController.move_y(SLIDE_POSITION.SCANNING_Y_MM)
                    self._wait_till_operation_is_completed()
                    self.navigationController.move_x(SLIDE_POSITION.SCANNING_X_MM)
                    self._wait_till_operation_is_completed()
                self.slidePositionController.homing_done = True
            else:
                timestamp_start = time.time()
                self.navigationController.move_y(SLIDE_POSITION.SCANNING_Y_MM-self.navigationController.y_pos_mm)
                self._wait_till_operation_is_completed()
                self.navigationController.move_x(SLIDE_POSITION.SCANNING_X_MM-self.navigationController.x_pos_mm)
                self._wait_till_operation_is_completed()

        # restore z
               
        self.navigationController.move_z_to(OBJECTIVE_DEFAULT_POS_MM)
        self._wait_till_operation_is_completed()
        print('z position restored')
        '''
        if self.slidePositionController.objective_retracted:
            if self.navigationController.get_pid_control_flag(2) is False:
                _usteps_to_clear_backlash = max(160,20*self.navigationController.z_microstepping)
                self.navigationController.microcontroller.move_z_to_usteps(self.slidePositionController.z_pos - STAGE_MOVEMENT_SIGN_Z*_usteps_to_clear_backlash)
                self._wait_till_operation_is_completed()
                self.navigationController.move_z_usteps(_usteps_to_clear_backlash)
                self._wait_till_operation_is_completed()
            else:
                self.navigationController.microcontroller.move_z_to_usteps(self.slidePositionController.z_pos)
                self._wait_till_operation_is_completed()
            self.slidePositionController.objective_retracted = False
            
        '''
    
    def run_autofocus(self, step_size_mm = [0.1, 0.01, 0.0015], start_z_mm = 3, end_z_mm = 7,shared_config=None):
        def focus_search(start_z_mm, end_z_mm, step_size_mm):
            z_positions = np.arange(start_z_mm, end_z_mm + step_size_mm/2, step_size_mm)
            focus_measures = []

            for z in z_positions:
                self.move_z_to(z)
                image = self.acquire_image()
                if shared_config is not None:
                    shared_config.live_z.value = z
                    shared_config.set_live_view_image(crop_image(image))

                focus_measure = utils.calculate_focus_measure(image, FOCUS_MEASURE_OPERATOR)
                focus_measures.append(focus_measure)
            
            # Create formatted strings for z positions and focus measures
            z_positions_str = " ".join([f"{z:.3f}" for z in z_positions])
            focus_measures_str = " ".join([f"{fm:.3f}" for fm in focus_measures])

            # Print (or log) the formatted strings
            print(f"Z positions (mm): {z_positions_str}")
            print(f"Focus measures: {focus_measures_str}")
            best_focus_index = np.argmax(focus_measures)
            return z_positions[best_focus_index], focus_measures[best_focus_index]

        # Move to start position (this will apply backlash compensation if moving down)
        self.move_z_to(start_z_mm)

        best_z = start_z_mm
        best_focus = float('-inf')

        for i, step_size in enumerate(step_size_mm):
            print(f"Stage {i+1}: step size = {step_size:.4f} mm")
            search_range = min(step_size * 10, end_z_mm - start_z_mm)
            if search_range <= step_size:
                continue
            
            if i == 0:  # First stage
                start_z = start_z_mm
                end_z = end_z_mm
            else:
                start_z = max(start_z_mm, best_z - search_range/2)
                end_z = min(end_z_mm, best_z + search_range/2)

            #print(f"Searching in range: {start_z:.6f} mm to {end_z:.6f} mm")
            stage_best_z, stage_best_focus = focus_search(start_z, end_z, step_size)

            if stage_best_focus > best_focus:
                best_z = stage_best_z
                best_focus = stage_best_focus

            #print(f"Stage {i+1} best focus: z = {best_z:.6f} mm, focus measure = {best_focus:.2f}")

        # Move to the best focus position
        self.move_z_to(best_z)

        print(f"Final best focus found at z = {best_z:.5f} mm with focus measure: {best_focus:.4f}")
    
        return best_z, best_focus

    def start_live(self):
        self.camera.start_streaming()
        self.liveController.start_live()

    def stop_live(self):
        self.liveController.stop_live()
        self.camera.stop_streaming()

    def _wait_till_operation_is_completed(self):
        while self.microcontroller.is_busy():
            time.sleep(SLEEP_TIME_S)

    def close(self):
        self.stop_live()
        self.camera.close()
        self.microcontroller.close()
        if USE_ZABER_EMISSION_FILTER_WHEEL or USE_OPTOSPIN_EMISSION_FILTER_WHEEL:
            self.emission_filter_wheel.close()