from __future__ import print_function, absolute_import, division

import numpy as np
import pandas as pd
import os
import time

from numpy import random
from random import shuffle
from psychopy import core, visual, event, monitors, logging, gui
from psychopy.hardware import keyboard
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
from sys import stdout
from os import system
from uldaq import get_daq_device_inventory, DaqDevice, InterfaceType, AOutFlag
from ast import literal_eval

#suppress PsychoPy warnings:
logging.console.setLevel(logging.CRITICAL)

class ShockExp:
    def __init__(self, subj, amperage, export_dir = "/Users/Dan/Documents/shockbox_files/subject_files/testing_phase_data", stimuli_dir = "/Users/Dan/Documents/shockbox_files/stimuli_art/", trial_dir = "/Users/Dan/Documents/shockbox_files/trials/trials"):
        self.subj = subj
        self.amperage = amperage
        self.escape = False
        self.escape_time = None
        self.total_stim_time = 0
        # self.total_duration_of_trials = 0
        self.total_persistence = 0
        self.keys = []
        self.export_dir = export_dir
        self.stimuli_dir = stimuli_dir
        self.trial_dir = trial_dir
        self.trial_df = None
        self.daq_device = None
        self.ao_device = None
        self.timer = core.Clock()
        self.anxiety_rating_history = []
        self.attention_check_history = []
        self.shock_intensity = 0
        self.show_early_environment = " "

    # generate n random trials of k epochs each based on statistics of the environment:
    def generate_trials(self, n_trials, file_name, highest_threat_prob = .4, max_epochs_per_trial = 6):

        print("Statistics of early-threat (ET) and late-threat (LT) environments:")
        # Calculate probability density function for each environment:
        temp_early_threat_pdf = [highest_threat_prob,
                                 (max_epochs_per_trial-1)*[(1-highest_threat_prob)/(max_epochs_per_trial-1)]]
        early_threat_pdf = []
        early_threat_pdf.append(temp_early_threat_pdf[0]) #should do this w/ list comprehension
        for i in temp_early_threat_pdf[1]:
            early_threat_pdf.append(i)
        late_threat_pdf = early_threat_pdf[::-1]
        print("ET PDF:", np.round(early_threat_pdf, 2), "LT PDF:", np.round(late_threat_pdf, 2))

        # Calculate cumulative density function for each environment:
        early_threat_cdf = np.cumsum(early_threat_pdf)
        late_threat_cdf = np.cumsum(late_threat_pdf)
        print("ET CDF:", np.round(early_threat_cdf, 2), "LT CDF:", np.round(late_threat_cdf, 2))

        # Calculate survival rate for each environment:
        # (Note: can use opposite environment's inverse cdf here
        # because the pdfs are balanced/inverted — a special case)
        early_threat_survival = late_threat_cdf[::-1]
        late_threat_survival = early_threat_cdf[::-1]
        print("ET surv:", np.round(early_threat_survival, 2), "LT surv:", np.round(late_threat_survival, 2))

        # Calculate hazard rates; i.e., the probability
        # of encountering a hazard given that you haven't yet:
        early_threat_hazard = []
        late_threat_hazard = []
        for i, val in enumerate(early_threat_pdf):
            early_threat_hazard.append(val/early_threat_survival[i])
        for i, val in enumerate(late_threat_pdf):
            late_threat_hazard.append(val/late_threat_survival[i])
        print("ET hzrd:", np.round(late_threat_hazard, 2), "LT hzrd:", np.round(early_threat_hazard, 2))

        # Loop over n_trials, randomly selecting either and ET or LT environments
        # and then creating a trial that adheres to the statistics of that environment:
        list_of_trials = []
        for i in range(n_trials):
            # Randomly draw from 1 to 100 without replacement to get a list of shuffled trial types:
            trial_types = np.random.choice(n_trials, n_trials, replace=False)
            #why isn't np.random.shuffle() working???

        sum_early_durations = 0
        sum_late_durations = 0
        count = 1

        # Randomly assign circle or square stimuli to early- and late-threat environments:
        early_late = [["circle.png", "circle_shock.png"], ["square.png", "square_shock.png"]]
        shuffle(early_late)

        # Use shuffled
        for i, each_trial in enumerate(trial_types):
            # Construct EARLY-THREAT environment trials:
            trial = []
            feedback_intervals = [(i+1) * 5 for i in range(max_epochs_per_trial)]
            if each_trial % 2 == 0:
                # Pseudo-randomize trial duration:
                early_threat_stim_duration = np.random.choice(feedback_intervals, 1, p=early_threat_pdf)
                sum_early_durations += early_threat_stim_duration
                # Build the trial:
                trial.append("fixation_cross.png")
                for i in range(int(early_threat_stim_duration/5)):
                    trial.append(early_late[0][0])
                trial.append(early_late[0][1])
                list_of_trials.append(trial)
                count += 1

            # Construct LATE-THREAT environment trials:
            else:
                # Pseudo-randomize trial duration:
                late_threat_stim_duration = np.random.choice(feedback_intervals, 1, p=late_threat_pdf)
                sum_late_durations += late_threat_stim_duration
                # Build the trial:
                trial.append("fixation_cross.png")
                for i in range(int(late_threat_stim_duration/5)):
                    trial.append(early_late[1][0])
                trial.append(early_late[1][1])
                list_of_trials.append(trial)
                count += 1

        print(f"\ntotal time (s) in ET: {sum_early_durations}\ntotal time (s) in LT: {sum_late_durations}\n")

        # Create Pandas DataFrame and export
        self.trial_df = pd.DataFrame({"STIMULI" : list_of_trials})
        self.trial_df.to_csv(os.path.join(self.trial_dir, f"{file_name}_early_{early_late[0][0][:-4]}.csv"))
        return None

    def connect_device(self):
        #!/usr/bin/env python
        # -*- coding: UTF-8 -*-

        """
        UL call demonstrated:             AoDevice.a_out()

        Purpose:                          Writes to a D/A output channel

        Demonstration:                    Outputs a user-specified voltage
                                          on analog output channel 0

        Steps:
        1. Call get_daq_device_inventory() to get the list of available DAQ devices
        2. Call DaqDevice() to create a DaqDevice object
        3. Call DaqDevice.get_ao_device() to get the AoDevice object for the analog
           output subsystem
        4. Verify the AoDevice object is valid
        5. Call DaqDevice.connect() to connect to the device
        6. Enter a value to output for the D/A channel
        7. Call AoDevice.a_out() to write a value to a D/A output channel
        8. Call DaqDevice.disconnect() and DaqDevice.release() before exiting the
           process
        """


        # Constants -- can be deleted
        CURSOR_UP = '\x1b[1A'
        ERASE_LINE = '\x1b[2K'

        """Analog output... connect?"""
        interface_type = InterfaceType.ANY
        self.output_channel = 0

        try:
            # Get descriptors for all of the available DAQ devices.
            devices = get_daq_device_inventory(interface_type)
            number_of_devices = len(devices)

            # Verify at least one DAQ device is detected.
            if number_of_devices == 0:
                raise RuntimeError('Error: No DAQ device is detected')

            print('Found', number_of_devices, 'DAQ device(s):')
            for i in range(number_of_devices):
                print('  [', i, '] ', devices[i].product_name, ' (',
                      devices[i].unique_id, ')', sep='')

            # Create the DAQ device from the descriptor at the specified index.
            self.daq_device = DaqDevice(devices[0])
            self.ao_device = self.daq_device.get_ao_device()

            # Verify the specified DAQ device supports analog output.
            if self.ao_device is None:
                raise RuntimeError('Error: The DAQ device does not support analog '
                                   'output')

            # Establish a connection to the device.
            descriptor = self.daq_device.get_descriptor()
            print('\nConnecting to', descriptor.dev_string, '- please wait...')
            # For Ethernet devices using a connection_code other than the default
            # value of zero, change the line below to enter the desired code.
            self.daq_device.connect(connection_code=0)

            ao_info = self.ao_device.get_info()
            # Select the first supported range.
            self.output_range = ao_info.get_ranges()[0]

            if self.daq_device.is_connected():
                print("Device connected")

        except RuntimeError as error:
            print('\n', error)


    def deliver_shock(self):
        #Deliver shock:
        if self.ao_device:
            #Deliver a low-amp shock to prime device for the desired amperage:
            recall_correct_amperage = self.amperage
            self.daq_device.connect()
            self.amperage = 1
            self.ao_device.a_out(self.output_channel, self.output_range, AOutFlag.DEFAULT, float(self.amperage))
            time.sleep(0.01)
            self.amperage = 0
            self.ao_device.a_out(self.output_channel, self.output_range, AOutFlag.DEFAULT, float(self.amperage))
            time.sleep(0.01)

            #Reset the amperage to the desired level and administer a 2nd shock:
            self.amperage = recall_correct_amperage

            self.ao_device.a_out(self.output_channel, self.output_range, AOutFlag.DEFAULT, float(self.amperage))
            print("SHOCK administered", self.output_channel, self.output_range, AOutFlag.DEFAULT, float(self.amperage))
            self.daq_device.disconnect()
        else:
            print('Oops. Something has gone awry, and there is no AO device.')


    def test_intensity(self):
        for i in np.arange(1, 11):
            print(i)
            self.amperage = 0
            self.deliver_shock()
            time.sleep(0.5)
            self.amperage = i
            self.deliver_shock()
            time.sleep(0.5)


    def disconnect_device(self):
        if self.daq_device:
            if self.daq_device.is_connected():
                self.daq_device.disconnect()
                print("Device disconnected")
            self.daq_device.release()
            print("Device released")

        else:
            print("No device connected")



    def show_stim(self, stim_image_file, size = ([1440, 900])):
        #Set window parameters:
        my_window = visual.Window(size = size, fullscr = True, units = 'pix')
        #Create stimulus object:
        stim = visual.ImageStim(my_window, image=os.path.join(self.stimuli_dir, stim_image_file))
        return None



    def prompt_rating(self, win):
        # Create rating-scale object:
        anxiety_rating_scale = visual.RatingScale(win, marker='glow', markerStart=2, size=1.5, pos=[0.0, -0.4], scale="How anxious did the last shape make you feel?\n\n1 = not at all anxious\n5 = extremely anxious\n\n", choices=[1, 2, 3, 4, 5], respKeys=['1', '2', '3', '4', '5'], showAccept=True, acceptSize=2.5, acceptPreText="select then ENTER")
        # If subject hasn't input a rating, wait. After input, move on.
        while anxiety_rating_scale.noResponse==True:
            anxiety_rating_scale.draw()
            win.flip()
        # Keep track of ONLY the ratings that the subject ultimately enters:
        self.anxiety_rating_history.append((anxiety_rating_scale.getHistory()[-1][0], np.round(self.timer.getTime(), 2)))
        return None



    def prompt_attention_check(self, win):
        # Create rating-scale object:
        attention_check = visual.RatingScale(win, marker='glow', markerStart=1, size=1.5, pos=[0.0, -0.4], scale="Which shape did you just see?", choices=["□", "○"], respKeys=['1', '2'], showAccept=True, acceptSize=2.5, acceptPreText="select then ENTER")
        # If subject hasn't input their answer, wait. After they answer, move on.
        while attention_check.noResponse==True:
            attention_check.draw()
            win.flip()
        # Keep track of ONLY the answers that the subject ultimately enters:
        self.attention_check_history.append((attention_check.getHistory()[-1][0], np.round(self.timer.getTime(), 2)))
        return None



    def run_trial(self, trial_csv, escapable = True, shock = True, rating_frequency = 3, attn_frequency = 5, fixation_stim_time = 1, env_stim_time = 5, shock_stim_time = 0.1, set_coin_flip = None):
        #Reset core.Clock to track timing of trial events:
        self.timer.reset()

        #Create the PsychoPy window object:
        my_window = visual.Window(fullscr = True, allowGUI = True, units = 'pix')

        def display_stim(stim_image_file, stim_display_time):
            my_stim = visual.ImageStim(my_window, image=os.path.join(self.stimuli_dir, stim_image_file))
            #Set a clock to record inter-trial escape timing
            trial_timer = core.Clock()
            if escapable:
                self.escape_time = np.nan
                done = False
                frame_count = 0
                while not done:
                    my_stim.draw()
                    my_window.flip() #runs at 60fps
                    frame_count += 1
                    #Record and check keystrokes:
                    self.keys = event.getKeys(['space'])
                    if 'space' in self.keys:
                        #Don't let subjects escape from select stimuli (e.g., fixation crosses
                        #and inescapable circles/squares):
                        if stim_image_file == "square.png" or stim_image_file == "circle.png":
                            self.escape_time = np.round(self.timer.getTime(), 2)
                            print(f"escaped at {self.escape_time}")
                            inter_trial_persistence = np.round(trial_timer.getTime(), 2)
                            print(f"inter-trial persistence {inter_trial_persistence}")
                            self.total_persistence += inter_trial_persistence
                            print(f"total persistence {self.total_persistence}")
                            print(f"keys={self.keys}\n")
                            self.escape = True
                            trial_timer.reset()
                            done = True
                    elif frame_count > stim_display_time * 60:
                        if stim_image_file == "square.png" or stim_image_file == "circle.png":
                            inter_trial_persistence = np.round(trial_timer.getTime(), 2)
                            print(f"inter-trial persistence {inter_trial_persistence}")
                            self.total_persistence += inter_trial_persistence
                            print(f"total persistence {self.total_persistence}")
                        else:
                            pass
                        print('no escape')
                        done = True
            elif not escapable:
                my_stim.draw()
                my_window.flip()
                core.wait(stim_display_time)

        #Open subject's trial.csv as Pandas DataFrame:
        trial_df = pd.read_csv(os.path.join(self.trial_dir, trial_csv))
        total_early_stim_time = np.sum(trial_df.STIMULI.str.count("early_stim")*env_stim_time)
        total_late_stim_time = np.sum(trial_df.STIMULI.str.count("late_stim")*env_stim_time)
        trial_df["STIMULI"] = trial_df["STIMULI"].apply(literal_eval)
        self.total_stim_time = total_early_stim_time + total_late_stim_time

        if set_coin_flip != None:
            coin_flip = set_coin_flip
        elif set_coin_flip == None:
            #Coin flip to randomize square and circle stimuli as late or early environments.
            #The coin flip will be printed after running the first phase, but it MUST be
            #manually entered before running the second phae.
            coin_flip = [["circle.png", "circle_shock.png", "no_escape_circle.png"], ["square.png", "square_shock.png", "no_escape_square.png"]]
            shuffle(coin_flip)

        self.show_early_environment = coin_flip[0][0]

        #Create lists of stimuli from trial DataFrame:
        temp_list_of_trials = []
        for i in range(len(trial_df)):
            temp_list_of_trials.append(trial_df.iloc[i, 1])

        if escapable == False:
            list_of_trials = []
            for each_sublist in temp_list_of_trials:
                each_sublist = [coin_flip[0][2] if x == "early_stim" else x for x in each_sublist]
                each_sublist = [coin_flip[0][1] if x == "early_shock" else x for x in each_sublist]
                each_sublist = [coin_flip[1][2] if x == "late_stim" else x for x in each_sublist]
                each_sublist = [coin_flip[1][1] if x == "late_shock" else x for x in each_sublist]
                list_of_trials.append(each_sublist)

        elif escapable == True:
            list_of_trials = []
            for each_sublist in temp_list_of_trials:
                if each_sublist[1] == "early_stim":
                    each_sublist[1] = coin_flip[0][2]
                else:
                    each_sublist[1] = coin_flip[1][2]
                each_sublist = [coin_flip[0][0] if x == "early_stim" else x for x in each_sublist]
                each_sublist = [coin_flip[0][1] if x == "early_shock" else x for x in each_sublist]
                each_sublist = [coin_flip[1][0] if x == "late_stim" else x for x in each_sublist]
                each_sublist = [coin_flip[1][1] if x == "late_shock" else x for x in each_sublist]
                list_of_trials.append(each_sublist)


        print(list_of_trials)
        print(coin_flip)
        print(self.show_early_environment)
        print(f"early environment = {coin_flip[0][0][:6]}")
        print(f"total time in early environment = {total_early_stim_time}")
        print(f"total time in late environment = {total_late_stim_time}\n\n")


        timing_dict = {"fixation_cross.png" : fixation_stim_time, "no_escape_circle.png" : env_stim_time, "no_escape_square.png" : env_stim_time, "circle.png" : env_stim_time, "square.png" : env_stim_time, "circle_shock.png" : shock_stim_time, "square_shock.png" : shock_stim_time}

        #Iterate over stimuli list, displaying and logging each:
        timing_for_log = []
        escape_times_for_log = []
        stim_for_log = []
        anxiety_ratings_for_log = []
        attention_checks_for_log = []
        correct_response = " "
        for n, trial in enumerate(list_of_trials):
            self.keys = []
            # self.total_duration_of_trials = env_stim_time * (len(trial)-2)
            for i, stimuli in enumerate(trial):
                if 'space' in self.keys:
                    pass
                else:
                    on_time = np.round(self.timer.getTime(), 2)
                    stim_timing = (f"stim number {i+1}, {stimuli}:", np.round(self.timer.getTime(), 2))
                    print(stim_timing)
                    #Trigger shock when the list contains an image with "shock" in its name:
                    if stimuli[-5] == "k":
                        if shock == True:
                            self.deliver_shock() # ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡
                    else:
                        pass
                    display_stim(stimuli, timing_dict.get(stimuli))

                    off_time = np.round(self.timer.getTime(), 2)
                    timing_for_log.append((on_time, off_time))
                    # if self.escape_time != None:
                    escape_times_for_log.append(self.escape_time)
                    # else:
                    #     escape_times_for_log.append(np.nan)
                    stim_for_log.append(stimuli)
                    anxiety_ratings_for_log.append(np.nan)
                    attention_checks_for_log.append(np.nan)


            # Get subject's real-time anxiety ratings every rating_frequency trials:
            if (n+1) % rating_frequency == 0:
                rating_on_time = np.round(self.timer.getTime(), 2)
                self.prompt_rating(my_window)
                anxiety_ratings_for_log.append(self.anxiety_rating_history[-1][0])
                timing_for_log.append((rating_on_time, np.round(self.timer.getTime(), 2)))
                stim_for_log.append("RATING")
                attention_checks_for_log.append(np.nan)
                escape_times_for_log.append(np.nan)

            # Prompt for attention checks:
            if (n+1) % attn_frequency == 0:
                attn_on_time = np.round(self.timer.getTime(), 2)
                self.prompt_attention_check(my_window)
                attn = self.attention_check_history[-1][0]
                attention_checks_for_log.append(attn)
                timing_for_log.append((attn_on_time, np.round(self.timer.getTime(), 2)))
                stim_for_log.append("ATTN")
                anxiety_ratings_for_log.append(np.nan)
                escape_times_for_log.append(np.nan)


        # Create DataFrame of stimuli timing and subject's responses, and export it to .csv:
        dat = {"time (on/off)" : timing_for_log, "stim type" : stim_for_log, "anx. ratings" : anxiety_ratings_for_log, "attn. checks" : attention_checks_for_log, "escape" : escape_times_for_log}
        log_df = pd.DataFrame(dat)
        if escapable == False:
            log_df.to_csv(os.path.join(self.export_dir, f"{self.subj}_training.csv"))
        else:
            log_df.to_csv(os.path.join(self.export_dir, f"{self.subj}.csv"))

        print(f"coin flip = {coin_flip}")

        if set_coin_flip != None:
            print(f"possible persistence time = {self.total_stim_time}")
            print(f"bonus payment = {self.total_persistence/100}")
            print(f"persistence = {self.total_persistence/self.total_stim_time}")


                                    ############################################
                                    #                                          #
                                    #                REMINDERS!                #
                                    #           Connect to port A16            #
                                    #       Enable self.deliver_shock()        #
                                    #      set 'escapable' to True/False       #
                                    #           Connect/test device            #
                                    #         Titrate shock intensity          #
                                    #       (a_out.py: 2 ENTER; 0 ENTER)       #
                                    #  set level when creating ShockExp object #
                                    #                                          #
                                    ############################################


################################################################################
# GENERATE DUMMY TRIALS:
# sub_names = ["{0:04}".format(i) for i in range(10000)]
# for i in range(len(sub_names)):
#     x = ShockExp(subj=sub_names[i], amperage=5, trial_dir = "/Users/Dan/Documents/shockbox_files/trials/dummy trials")
#     x.generate_trials(n_trials=100, highest_threat_prob=.4, max_epochs_per_trial=6, file_name=f"{x.subj}")
################################################################################


################################################################################
# GENERATE AND RUN TRIALS:
x = ShockExp(subj=, amperage=NUM) #SET AMPERAGE TO MAX TOLERATED DURING WORKUP
x.connect_device()
print(f"subject: {x.subj} — shock level 1-10: {x.amperage}")
# x.generate_trials(n_trials=4, highest_threat_prob=.4, max_epochs_per_trial=6, file_name=f"{x.subj}_trials")


# FOR TROUBLESHOOTING/TESTING:
# x.run_trial(trial_csv = "test_trial.csv", escapable = True, shock = False, rating_frequency = 1, attn_frequency = 1, fixation_stim_time = 1, env_stim_time = 2, shock_stim_time = 0.1, set_coin_flip = [["square.png", "square_shock.png", "no_escape_square.png"], ["circle.png", "circle_shock.png", "no_escape_circle.png"]])


# FOR STUDY PARTICIPANTS:
# FIRST, run get_amperage.py to for shock workup; be sure to set "amperage" in ShockExp object

# SECOND, randomly draw trials using Uncertainty Study Setup.ipynb; enter those file names below

# THIRD, run the NO ESCAPES phase:
# x.run_trial(trial_csv = "6280.csv", escapable = False, shock = True, rating_frequency = 4, attn_frequency = 3, fixation_stim_time = 1, env_stim_time = 5, shock_stim_time = 0.1, set_coin_flip = None)

# FOURTH, run the ESCAPES phase:
x.run_trial(trial_csv = "5369.csv", escapable = True, shock = True, rating_frequency = 4, attn_frequency = 3, fixation_stim_time = 1, env_stim_time = 5, shock_stim_time = 0.1, set_coin_flip = ])

# Enter one of these two, based on the 'coin flip' returned from the THIRD step. Early environment is first.
# [["circle.png", "circle_shock.png", "no_escape_circle.png"], ["square.png", "square_shock.png", "no_escape_square.png"]]
# [["square.png", "square_shock.png", "no_escape_square.png"], ["circle.png", "circle_shock.png", "no_escape_circle.png"]]

################################################################################
