import numpy as np

import brain_observatory_utilities.datasets.optical_physiology.data_formatting as ophys


def get_stimulus_block_description(stimulus_block, dataset='VBN'):
    """
    For a given dataset, will return a string description of what was shown during the provided stimulus_block.
    Currently only applies to dataset='VBN' because stimulus_block has not yet been applied to VBO

    :param stimulus_block: int between 0 and 5
    :param dataset: 'VBN' = Visual Behavior Neuropixels, 'VBO' = Visual Behavior Ophys
    :return: string description of what stimulus was shown during the provided stimulus_block
    """
    if dataset == 'VBO':
        print('stimulus block is not currently used in the Visual Behavior Ophys dataset')
        stimulus_block_lookup_dict = {}
    elif dataset == 'VBN':
        stimulus_block_lookup_dict = {0: 'change detection active behavior',
                                      1: '10 seconds gray screen',
                                      2: 'gabor patches receptive field mapping',
                                      3: '5 minutes gray screen',
                                      4: 'full field flashes',
                                      5: 'change detection passive replay'}
        print(stimulus_block_lookup_dict[stimulus_block], 'is shown during Visual Behavior Neuropixels stimulus_block', stimulus_block)
    return stimulus_block_lookup_dict[stimulus_block]


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



