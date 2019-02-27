# Psyc161FinalProject
## Basic EEG Analyses and Data Visualisation Using a Previously Collected EEG Data

My main purpose in the course to learn better data visualisation methods. So, in this project I want to work on an EEG project of which data I finished collecting myself last year. I ran statistical analyses, even found significant effects which I hypothesised but all ended up not writing the paper. As part of this course I want to learn a software to preprocess EEG data (I only need to do some final steps in preprocessing since the data is already preprocessed in BrainVision Analyzer) and visualise EEG data in multiple ways. I want to use MNE software unless you have a better suggestion. I normally benefit from software/toolboxes/packages to a certain degree. So, I imagine -especially during plotting- I will be appealing to other python packages. In the end of this project I want to have publishable figures. My tentative plan is as follows:

1.  Going over the introduction and data structures documentations <https://martinos.org/mne/stable/documentation.html#>
2.  Starting with the pre-processing data (I will only need to implement a few final steps here)
3.  Drawing ERPs (My design has 3x6 conditions. So it won't be very simple I guess)
4.  Visualising changes in ERP components' amplitude and latency (Possibly using pandas, numpy, altair and seaborne)
5.  Wavelet Analyses

## Description of the Workflow

I. Trial synchronization between psychophysical and eeg data:

**Problem 1:** *The behavioural data was collected in three sessions of which eeg data was continuous. Each session corresponds to a motion type condition and the order was randomised.*
    
**Problem 2:** *EEG data of this study went through two steps of artefact rejection. The trial numbers in the eeg files do not match the trial files in behavioural data folders.*

II. Behavioural trial rejection: Determine valid trials according to the behavioural response criteria    

    1. __Import subject data information__
    2. __Combine 3 condition files of each subject according to the session order__
    3. __Mark invalid key press:__
        * log files registered these as __key='space'__ or __'-'__, outputted here as __key=0__
    4. __Discard rejected trials at Artefact Rejection 1 & 2__
    5. __Mark trials that are not valid according to Reaction Time criteria:__
        * AVG-2SD > RT > AVG + 2SD
    6. __Write the new data file in .csv format and save it in the data folder:__
        * fname: _{<sub_init>}_subdatbhv.csv_

III. Register session statistics to further examine any potential confounds _i.e. low SNR due to high rate of trial rejection_

    1. __Register following variables in a .csv file:__
        * _info_sxnstats.csv_
        * sub_no, sub_name, bhv_ptr, eeg_ptr, qTrlRej, qTrlVld, qTrlVldKey, qTrlVldRT

IV. Behavioural data analysis: psychophysical curve fitting

    1. __Read each '{<sub_init>}_subdatbhv.csv' file__
    2. __Separate each condition 3x6 (Motion Type x Coherence Level)__
    3. __Compute number of correct responses for each condition__
    4. __Write in a .csv file in the following format:__ 
        * sub_no, cond_no, coh_lvl, qOK, qTrl
        * *datbhv_pfit.csv*
    5. __Register the index number of valid and correct trials in another .csv file:__
        * sub_no, trl_vld, trl_ok
        * *info_trials.csv*


