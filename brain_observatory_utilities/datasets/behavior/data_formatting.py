import numpy as np
import pandas as pd

import brain_observatory_utilities.utilities.general_utilities as utilities
import brain_observatory_utilities.datasets.stimulus_alignment.data_formatting as stimulus_alignment
from allensdk.brain_observatory.behavior.trials_processing import calculate_reward_rate



def add_mean_running_speed_to_stimulus_presentations(stimulus_presentations,
                                                     running_speed,
                                                     time_window=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains
    the mean running speed in a range relative to
    the stimulus start time.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of
            stimulus presentations.
            Must contain: 'start_time'
        running_speed (pd.DataFrame): dataframe of running speed.
            Must contain: 'speed', 'timestamps'
        time_window: array
            timestamps in seconds, relative to the start of each stimulus
            to average the running speed.
            default = [-3,3]
    Returns:
        stimulus_presentations with new column
        "mean_running_speed" containing the
        mean running speed within the specified window
        following each stimulus presentation.

    Example:
        # get visual behavior cache
        from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc  # noqa E501
        cache_dir = SOME_LOCAL_DIR
        cache = bpc.from_s3_cache(cache_dir=cache_dir)

        # load data for one experiment
        ophys_experiment = cache.get_behavior_ophys_experiment(experiment_id)

        # get necessary tables
        stimulus_presentations = ophys_experiment.stimulus_presentations.copy()
        running_speed = ophys_experiment.running_speed.copy()

        # add running_speed to stim presentations
        stimulus_presentations = add_mean_running_speed_to_stimulus_presentations(stimulus_presentations, running_speed)  # noqa E501
    '''

    stim_running_speed = stimulus_presentations.apply(
        lambda row: utilities.get_trace_average(
            running_speed['speed'].values,
            running_speed['timestamps'].values,
            row["start_time"] + time_window[0],
            row["start_time"] + time_window[1]), axis=1,)
    stimulus_presentations["mean_running_speed"] = stim_running_speed
    return stimulus_presentations


def add_mean_pupil_to_stimulus_presentations(stimulus_presentations, eye_tracking, column_to_use='pupil_area', time_window=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains
    the mean pupil area, diameter, or radius in a range relative to
    the stimulus start time.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.  # noqa E501
            Must contain: 'start_time'
        eye_tracking (pd.DataFrame): dataframe of eye tracking data.
            Must contain: timestamps', 'pupil_area', 'pupil_width', 'likely_blinks'
        time_window (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the pupil area.
        column_to_use: column in eyetracking table to use to get mean, options: 'pupil_area', 'pupil_width', 'pupil_radius', 'pupil_diameter'
                        if 'pupil_diameter' or 'pupil_radius' are provided, they will be calculated from 'pupil_area'
                        if 'pupil_width' is provided, the column 'pupil_width' will be directly used from eye_tracking table
    Returns:
        stimulus_presentations table with new column "mean_pupil_"+column_to_use with the
        mean pupil value within the specified window following each stimulus presentation.

    Example:
        # get visual behavior cache
        from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc  # noqa E501
        cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_cache'
        cache = bpc.from_s3_cache(cache_dir=cache_dir)

        # load data for one experiment
        ophys_experiment = cache.get_behavior_ophys_experiment(experiment_id)

        # get necessary tables
        stimulus_presentations = ophys_experiment.stimulus_presentations.copy()
        eye_tracking = ophys_experiment.eye_tracking.copy()

        # add pupil area to stim presentations
        stimulus_presentations = add_mean_pupil_to_stimulus_presentations(stimulus_presentations, eye_tracking, column_to_use='pupil_area')  # noqa E501
    '''

    eye_tracking = data_access.get_pupil_data(eye_tracking, interpolate_likely_blinks=True, normalize_to_gray_screen=True, zscore=False,
                   interpolate_to_ophys=False, ophys_timestamps=None, stimulus_presentations=stimulus_presentations)

    eye_tracking_timeseries = eye_tracking[column_to_use].values
    mean_pupil_around_stimulus = stimulus_presentations.apply(
        lambda row: utilities.get_trace_average(
            eye_tracking_timeseries,
            eye_tracking['timestamps'].values,
            row["start_time"] + time_window[0],
            row["start_time"] + time_window[1],
        ), axis=1,)
    stimulus_presentations["mean_"+column_to_use] = mean_pupil_around_stimulus
    return stimulus_presentations


def add_rewards_to_stimulus_presentations(stimulus_presentations,
                                          rewards,
                                          time_window=[0,3]):
    '''
    Append a column to stimulus_presentations which contains
    the timestamps of rewards that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of
            stimulus presentations.
            Must contain: 'start_time'
        rewards (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        time_window (list with 2 elements): start and end of the range
            relative to the start of each stimulus
            to average the running speed.
    Returns:
        stimulus_presentations with a new column called "reward" that contains
        reward times that fell within the window relative to each stim time

    Example:
        # get visual behavior cache
        from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc  # noqa E501
        cache_dir = SOME_LOCAL_DIRECTORY
        cache = bpc.from_s3_cache(cache_dir=cache_dir)

        # load data for one experiment
        ophys_experiment = cache.get_behavior_ophys_experiment(experiment_id)

        # get necessary tables
        stimulus_presentations = ophys_experiment.stimulus_presentations.copy()
        rewards = ophys_experiment.rewards.copy()

        # add rewards to stim presentations
        stimulus_presentations = add_rewards_to_stimulus_presentations(stimulus_presentations, rewards)  # noqa E501
    '''

    reward_times = rewards['timestamps'].values
    rewards_each_stim = stimulus_presentations.apply(
        lambda row: reward_times[
            ((reward_times > row["start_time"] + time_window[0]) & (reward_times < row["start_time"] + time_window[1]))],
        axis=1,
    )
    stimulus_presentations["rewards"] = rewards_each_stim
    return stimulus_presentations


def add_licks_to_stimulus_presentations(stimulus_presentations,
                                        licks,
                                        time_window=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which
    contains the timestamps of licks that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations (pd.DataFrame): 
            dataframe of stimulus presentations.
            Must contain: 'start_time'
        licks (pd.DataFrame): lick dataframe. Must contain 'timestamps'
        time_window (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.  # noqa E501
    Returns:
        stimulus_presentations with a new column called "licks" that contains
        lick times that fell within the window relative to each stim time


    Example:
        # get visual behavior cache
        from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc  # noqa E501
        cache_dir = SOME_LOCAL_DIRECTORY
        cache = bpc.from_s3_cache(cache_dir=cache_dir)

        # load data for one experiment
        ophys_experiment = cache.get_behavior_ophys_experiment(experiment_id)

        # get necessary tables
        stimulus_presentations = ophys_experiment.stimulus_presentations.copy()
        licks = ophys_experiment.licks.copy()

        # add licks to stim presentations
        stimulus_presentations = add_licks_to_stimulus_presentations(stimulus_presentations, licks)
    '''

    lick_times = licks['timestamps'].values
    licks_each_stim = stimulus_presentations.apply(
        lambda row: lick_times[
            ((lick_times > row["start_time"] + time_window[0]) & (lick_times < row["start_time"] + time_window[1]))],
        axis=1,
    )
    stimulus_presentations["licks"] = licks_each_stim
    return stimulus_presentations


def add_reward_rate_to_stimulus_presentations(stimulus_presentations,
                                              trials):
    '''
    Parameters:
    ____________
    trials: Pandas.DataFrame
        ophys_experiment.trials
    stimulus_presentation: Pandas.DataFrame
        ophys_experiment.stimulus_presentations

    Returns:
    ___________
    stimulus_presentation: Pandas.DataFrame
        with 'reward_rate_trials' column

    'reward_rate' is calculated by the SDK based on the rolling reward rate over trials (not stimulus presentations)
    https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/trials_processing.py#L941
    '''

    last_time = 0
    reward_rate_by_frame = []
    if 'reward_rate' not in trials:
        trials['reward_rate'] = calculate_reward_rate(trials['response_latency'].values,
                                                      trials['start_time'],
                                                      window=.5)

    trials = trials[trials['aborted'] == False]  # NOQA
    for change_time in trials.change_time.values:
        reward_rate = trials[trials.change_time ==  # NOQA
                             change_time].reward_rate.values[0]
        for start_time in stimulus_presentations.start_time:
            if (start_time < change_time) and (start_time > last_time):
                reward_rate_by_frame.append(reward_rate)
                last_time = start_time
    # fill the last flashes with last value
    for i in range(len(stimulus_presentations) - len(reward_rate_by_frame)):
        reward_rate_by_frame.append(reward_rate_by_frame[-1])

    stimulus_presentations['reward_rate'] = reward_rate_by_frame
    return stimulus_presentations


def add_epochs_to_stimulus_presentations(stimulus_presentations, time_column='start_time', epoch_duration_mins=10):
    """
    Add column called 'epoch' with values as an index for the epoch within a session, for a given epoch duration.

    :param stimulus_presentations: dataframe with a column indicating event start times
    :param time_column: name of column in dataframe indicating event times
    :param epoch_duration_mins: desired epoch length in minutes
    :return: input dataframe with epoch column added
    """
    start_time = stimulus_presentations[time_column].values[0]
    stop_time = stimulus_presentations[time_column].values[-1]
    epoch_times = np.arange(start_time, stop_time, epoch_duration_mins * 60)
    stimulus_presentations['epoch'] = None
    for i, time in enumerate(epoch_times):
        if i < len(epoch_times) - 1:
            indices = stimulus_presentations[(stimulus_presentations[time_column] >= epoch_times[i]) &
                                             (stimulus_presentations[time_column] < epoch_times[i + 1])].index.values
        else:
            indices = stimulus_presentations[(stimulus_presentations[time_column] >= epoch_times[i])].index.values
        stimulus_presentations.at[indices, 'epoch'] = i
    return stimulus_presentations


def add_trials_id_to_stimulus_presentations(stimulus_presentations, trials):
    """
    Add trials_id to stimulus presentations by finding the closest change time to each stimulus start time
    If there is no corresponding change time, the trials_id is NaN
    :param: stimulus_presentations: stimulus_presentations attribute of BehaviorOphysExperiment object, must have 'start_time'
    :param trials: trials attribute of BehaviorOphysExperiment object, must have 'change_time'
    """
    # for each stimulus_presentation, find the trials_id that is closest to the start time
    # add to a new column called 'trials_id'
    for idx, stimulus_presentation in stimulus_presentations.iterrows():
        start_time = stimulus_presentation['start_time']
        query_string = 'change_time > @start_time - 1 and change_time < @start_time + 1'
        trials_id = (np.abs(start_time - trials.query(query_string)['change_time']))
        if len(trials_id) == 1:
            trials_id = trials_id.idxmin()
        else:
            trials_id = np.nan
        stimulus_presentations.loc[idx, 'trials_id'] = trials_id
    return stimulus_presentations


def add_trials_data_to_stimulus_presentations_table(stimulus_presentations, trials):
    """
    Add trials_id to stimulus presentations table then join relevant columns of trials with stimulus_presentations
    :param: stimulus_presentations: stimulus_presentations attribute of BehaviorOphysExperiment object, must have 'start_time'
    :param trials: trials attribute of BehaviorOphysExperiment object, must have 'change_time'
    """
    # add trials_id and merge to get trial type information
    stimulus_presentations = add_trials_id_to_stimulus_presentations(stimulus_presentations, trials)
    # only keep certain columns
    trials = trials[['change_time', 'go', 'catch', 'aborted', 'auto_rewarded',
                    'hit', 'miss', 'false_alarm', 'correct_reject',
                    'response_time', 'response_latency', 'reward_time', 'reward_volume', ]]
    # merge trials columns into stimulus_presentations
    stimulus_presentations = stimulus_presentations.reset_index().merge(trials, on='trials_id', how='left')
    stimulus_presentations = stimulus_presentations.set_index('stimulus_presentations_id')
    return stimulus_presentations



def add_time_from_last_change_to_stimulus_presentations(stimulus_presentations):
    '''
    Adds a column to stimulus_presentations called 'time_from_last_change', which is the time, in seconds since the last image change

    ARGS: SDK session object
    MODIFIES: session.stimulus_presentations
    RETURNS: stimulus_presentations
    '''
    stimulus_times = stimulus_presentations["start_time"].values
    change_times = stimulus_presentations.query('is_change')['start_time'].values
    time_from_last_change = general_utilities.time_from_last(stimulus_times, change_times)
    stimulus_presentations["time_from_last_change"] = time_from_last_change

    return stimulus_presentations


def create_lick_rate_df(dataset, bin_size=6):
    '''
    Returns a dataframe containing columns for 'timestamps', 'licks', and 'lick_rate'. where values are from
    'licks' is a binary array of the length of stimulus timestamps where frames with no lick are 0 and frames with a lick are 1,
    'lick_rate' contains values of 'licks' averaged over a rolling window using the provided 'bin_size' in units of acquisition frames
    Can be used to compute event triggered average lick rate using brain_observatory_utilities.utilities.general_utilities.event_triggered_response

    Parameters:
    -----------
    dataset: obj
        AllenSDK BehaviorOphysExperiment object, BehaviorSession object, or BehaviorEcephysSession object
        See:
        https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py  # noqa E501
        https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_session.py  # noqa E501
        https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/ecephys/behavior_ecephys_session.py  # noqa E501
    bin_size: int
        number of frames (timestamps) to average over to get rolling lick rate
        default = 6 frames to give lick rate in licks / 100ms (assuming 60Hz acquisition rate for licks)
    Returns:
    --------
    Pandas.DataFrame with columns 'timestamps', 'licks', and 'lick_rate' in units of licks / 100ms
    '''

    timestamps = dataset.stimulus_timestamps.copy()
    licks = dataset.licks.copy()
    lick_array = np.zeros(timestamps.shape)
    lick_array[licks.frame.values] = 1
    licks_df = pd.DataFrame(data=timestamps, columns=['timestamps'])
    licks_df['licks'] = lick_array
    licks_df['lick_rate'] = licks_df['licks'].rolling(window=bin_size, min_periods=1, win_type='triang').mean()

    return licks_df


def filter_eye_tracking(eye_tracking, interpolate_likely_blinks=False, normalize_to_gray_screen=False, zscore=False,
                   interpolate_to_ophys=False, ophys_timestamps=None, stimulus_presentations=None):
    """
    removes 'likely_blinks' from all columns in eye_tracking and converts pupil_area to pupil_diameter and pupil_radius
    interpolates over NaNs resulting from removing likely_blinks if interpolate = true
    :param eye_tracking: eye_tracking attribute of AllenSDK BehaviorOphysExperiment object
    :param column_to_use: 'pupil_area', 'pupil_width', 'pupil_diameter', or 'pupil_radius'
                            'pupil_area' and 'pupil_width' are existing columns of the eye_tracking table
                            'pupil_diameter' and 'pupil_radius' are computed from 'pupil_area' column
    :param interpolate: Boolean, whether or not to interpolate points where likely_blinks occured

    :return:
    """
    import scipy

    # set index to timestamps so they dont get overwritten by subsequent operations
    eye_tracking = eye_tracking.set_index('timestamps')

    # compute pupil_diameter and pupil_radius from pupil_area
    eye_tracking['pupil_diameter'] = np.sqrt(eye_tracking.pupil_area) / np.pi  # convert pupil area to pupil diameter
    eye_tracking['pupil_radius'] = np.sqrt(eye_tracking['pupil_area'] * (1 / np.pi))  # convert pupil area to pupil radius

    # set all timepoints that are likely blinks to NaN for all eye_tracking columns
    if True in eye_tracking.likely_blink.unique(): # only can do this if there are likely blinks to filter out
        eye_tracking.loc[eye_tracking['likely_blink'], :] = np.nan

    # add timestamps column back in
    eye_tracking['timestamps'] = eye_tracking.index.values

    # interpolate over likely blinks, which are now NaNs
    if interpolate_likely_blinks:
        eye_tracking = eye_tracking.interpolate()

    # divide all columns by average of gray screen period prior to behavior session
    if normalize_to_gray_screen:
        assert stimulus_presentations is not None, 'must provide stimulus_presentations if normalize_to_gray_screen is True'
        spontaneous_frames = stimulus_alignment.get_spontaneous_frames(stimulus_presentations,
                                                             eye_tracking.timestamps.values,
                                                             gray_screen_period_to_use='before')
        for column in eye_tracking.keys():
            if (column != 'timestamps') and (column!='likely_blink'):
                gray_screen_mean_value = np.nanmean(eye_tracking[column].values[spontaneous_frames])
                eye_tracking[column] = eye_tracking[column] / gray_screen_mean_value
    # z-score pupil data
    if zscore:
        for column in eye_tracking.keys():
            if (column != 'timestamps') and (column != 'likely_blink'):
                eye_tracking[column] = scipy.stats.zscore(eye_tracking[column], nan_policy='omit')

    # interpolate to ophys timestamps
    if interpolate_to_ophys:
        assert ophys_timestamps is not None, 'must provide ophys_timestamps if interpolate_to_ophys is True'
        eye_tracking_ophys_time = pd.DataFrame({'timestamps': ophys_timestamps})
        for column in eye_tracking.keys():
            if (column != 'timestamps') and (column != 'likely_blink'):
                f = scipy.interpolate.interp1d(eye_tracking['timestamps'], eye_tracking[column], bounds_error=False)
                eye_tracking_ophys_time[column] = f(eye_tracking_ophys_time['timestamps'])
                eye_tracking_ophys_time[column].fillna(method='ffill', inplace=True)
        eye_tracking = eye_tracking_ophys_time

    return eye_tracking
