import numpy as np
import pandas as pd
import brain_observatory_utilities.datasets.optical_physiology.data_formatting as ophys


def get_stimulus_name(dataObject):
    """gets the stimulus name for a dataset object
    Parameters
    ----------
    dataObject : (BehaviorSesson, BehaviorOphysExperiment)
        Objects provided via allensdk.brain_observatory module

    Returns
    -------
    string
        the stimulus name for a given session or experiment
    """
    stimulus_name = dataObject.metadata['session_type']
    return stimulus_name


def get_lick_timestamps(dataObject):
    """gets the timestamps of licks

    Parameters
    ----------
    dataObject : (BehaviorSesson, BehaviorOphysExperiment)
        Objects provided via allensdk.brain_observatory module
    Returns
    -------
    array
        array of lick timestamps
    """
    lick_timestamps = dataObject.licks["timestamps"].values
    return lick_timestamps


def get_reward_timestamps(dataObject, reward_type="all"):
    """gets the timestamps of water rewards
    Parameters
    ----------
    dataObject : (BehaviorSesson, BehaviorOphysExperiment)
        Objects provided via allensdk.brain_observatory module
    reward_type : string
        by default "all", options:
            "all": all rewards (auto and earned)
            "auto": only free or unearned rewards
            "earned": only earned (hit on a go trial) rewards

    Returns
    -------
    array
        arry of reward timestamps
    """

    rewards_df = dataObject.rewards
    if reward_type == "all":
        reward_timestamps = rewards_df['timestamps'].values
    elif reward_type == "auto":
        reward_timestamps = rewards_df.loc[rewards_df["autorewarded"] == True]["timestamps"].values  # noqa: E501
    elif reward_type == "earned":
        reward_timestamps = rewards_df.loc[rewards_df["autorewarded"] == False]["timestamps"].values  # noqa: E501
    return reward_timestamps


def get_running_speed_timeseries(dataObject):
    """gets the mouse running speed timeseries

    Parameters
    ----------
    ophysObject : (BehaviorSesson, BehaviorOphysExperiment)
        Objects provided via allensdk.brain_observatory module

    Returns
    -------
    tuple:
        running_speed (cm/sec), timestamps (sec)
    """
    running_speed = dataObject.running_speed["speed"].values
    timestamps = dataObject.running_speed["timestamps"].values
    return running_speed, timestamps


def get_pupil_area_timeseries(ophysObject):
    """gets mouse's pupil area timeseries

    Parameters
    ----------
    ophysObject : (BehaviorOphysExperiment)
        Object provided via allensdk.brain_observatory
        module

    Returns
    -------
   tuple
        pupil area (pixels ^2), timestamps
    """
    pupil_area = ophysObject.eye_tracking["pupil_area"].values
    timestamps = ophysObject.eye_tracking["timestamps"].values
    return pupil_area, timestamps


def get_trial_type(trials_df, trial_type,
                   include_aborted):
    """filters the trials table to a specific trial type

    Parameters
    ----------
    trials_df : pandas dataframe
        _description_
    trial_type : str
        options:
            "all": all trial types (both go and catch)
            "go": go trials (change presented)
            "catch": catch trials (no change presented)
    include_aborted : bool,
        include aborted trials,
    Returns
    -------
    pandas dataframe
        filtered trials table
    """
    if trial_type == "all":
        if include_aborted is False:
            filtered_trials = trials_df.loc[trials_df["aborted"] == False]  # noqa: E501
        else:
            filtered_trials = trials_df
    else:
        filtered_trials = \
            trials_df.loc[trials_df[trial_type] == True]
    return filtered_trials


def get_response_type(trials_df, response_type,
                      include_aborted):
    """filters the trials table to a specific mouse
       behavior response type.

    Parameters
    ----------
    trials_df : pandas dataframe
        _description_
    response_type : str,
        options:
            "all": all responses
            "hit": "hit" responses, only occurs on go trial type
            "miss": "miss" responses, only occurs on
                go trial type
            "false_alarm": "false_alarm" responses,
                only occurs on catch trial type
            "correct_reject": "correct_reject" responses,
                only occurs on catch trial type
    include_aborted : bool
        include aborted trials, by default True

    Returns
    -------
    pandas dataframe
        filtered trials table
    """
    if response_type == "all":
        if include_aborted is False:
            filtered_trials = trials_df.loc[trials_df["aborted"] == False]
        else:
            filtered_trials = trials_df
    else:
        filtered_trials = trials_df.loc[trials_df[response_type] == True]
    return filtered_trials


def get_trials_data(dataObject,
                    trial_type="all", response_type="all",
                    include_aborted_trials=True):
    """gets the trials dataframe attribute and can optionally
       filter for specific trial or behavior response types.

    Parameters
    ----------
    dataObject : (BehaviorSesson, BehaviorOphysExperiment)
        Objects provided via allensdk.brain_observatory module
    trial_type : str, optional
        by default "all", options:
            "all": all trial types (both go and catch)
            "go": go trials (change presented)
            "catch": catch trials (no change presented)
    response_type : str, optional
        by default "all", options:
            "all": all responses
            "hit": "hit" responses, only occurs on go trial type
            "miss": "miss" responses, only occurs on
                go trial type
            "false_alarm": "false_alarm" responses,
                only occurs on catch trial type
            "correct_reject": "correct_reject" responses,
                only occurs on catch trial type
    include_aborted_trials : bool, optional
        whether to include aborted trials, by default True

    Returns
    -------
    pandas dataframe
        trials dataframe, optionally filtered
    """
    trials_df = dataObject.trials
    filtered_df = get_trial_type(trials_df,
                                 trial_type=trial_type,
                                 include_aborted=include_aborted_trials)
    filtered_df = get_response_type(filtered_df,
                                    response_type=response_type,
                                    include_aborted=include_aborted_trials)
    return filtered_df


def get_pupil_data(eye_tracking, interpolate_likely_blinks=False, normalize_to_gray_screen=False, zscore=False,
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
    # convert pupil area to pupil diameter
    eye_tracking['pupil_diameter'] = np.sqrt(eye_tracking.pupil_area) / np.pi
    eye_tracking['pupil_radius'] = np.sqrt(
        eye_tracking['pupil_area'] * (1 / np.pi))  # convert pupil area to pupil radius

    # set all timepoints that are likely blinks to NaN for all eye_tracking columns
    if True in eye_tracking.likely_blink.unique():  # only can do this if there are likely blinks to filter out
        eye_tracking.loc[eye_tracking['likely_blink'], :] = np.nan

    # add timestamps column back in
    eye_tracking['timestamps'] = eye_tracking.index.values

    # interpolate over likely blinks, which are now NaNs
    if interpolate_likely_blinks:
        eye_tracking = eye_tracking.interpolate()

    # divide all columns by average of gray screen period prior to behavior session
    if normalize_to_gray_screen:
        assert stimulus_presentations is not None, 'must provide stimulus_presentations if normalize_to_gray_screen is True'
        spontaneous_frames = ophys.get_spontaneous_frames(stimulus_presentations,
                                                          eye_tracking.timestamps.values,
                                                          gray_screen_period_to_use='before')
        for column in eye_tracking.keys():
            if (column != 'timestamps') and (column != 'likely_blink'):
                gray_screen_mean_value = np.nanmean(
                    eye_tracking[column].values[spontaneous_frames])
                eye_tracking[column] = eye_tracking[column] / \
                    gray_screen_mean_value
    # z-score pupil data
    if zscore:
        for column in eye_tracking.keys():
            if (column != 'timestamps') and (column != 'likely_blink'):
                eye_tracking[column] = scipy.stats.zscore(
                    eye_tracking[column], nan_policy='omit')

    # interpolate to ophys timestamps
    if interpolate_to_ophys:
        assert ophys_timestamps is not None, 'must provide ophys_timestamps if interpolate_to_ophys is True'
        eye_tracking_ophys_time = pd.DataFrame(
            {'timestamps': ophys_timestamps})
        for column in eye_tracking.keys():
            if (column != 'timestamps') and (column != 'likely_blink'):
                f = scipy.interpolate.interp1d(
                    eye_tracking['timestamps'], eye_tracking[column], bounds_error=False)
                eye_tracking_ophys_time[column] = f(
                    eye_tracking_ophys_time['timestamps'])
                eye_tracking_ophys_time[column].fillna(
                    method='ffill', inplace=True)
        eye_tracking = eye_tracking_ophys_time

    return eye_tracking
