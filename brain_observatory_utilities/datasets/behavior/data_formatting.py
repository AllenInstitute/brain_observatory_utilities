import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.trials_processing import calculate_reward_rate
from brain_observatory_utilities.utilities import general_utilities as utilities
import brain_observatory_utilities.datasets.behavior.data_access as data_access



def limit_to_behavior_session_block(stimulus_presentations): 
    '''
    Function to limit stimulus presentations table to only the active change detection behavior block (stimulus_block=0)
    
    Args:
        stimulus_presentations (pd.DataFrame): SDK session dataframe of stimulus presentations.

    Returns:
        stimulus_presentations dataframe restricted to stimuli where stimulus_block==0. 
    '''
    
    if 'stimulus_block' in stimulus_presentations.keys():
        stimulus_presentations = stimulus_presentations[stimulus_presentations.stimulus_block==0]
    return stimulus_presentations


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


def add_mean_pupil_to_stimulus_presentations(stimulus_presentations, eye_tracking,
                                             column_to_use='pupil_area', time_window=[0, 0.75]):
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


def add_rewards_to_stimulus_presentations(stimulus_presentations, rewards, time_window=[0,0.75]):
    '''
    Append a column called 'rewards', to stimulus_presentations which contains the timestamps of rewards that occured
    in a range relative to the onset of the stimulus, as well as a Boolean column
    called 'rewarded' which indicates whether there was a reward delivered on a given stimulus presentation or not

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
    stimulus_presentations['rewarded'] = [True if len(rewards) > 0 else False for rewards in
                                          stimulus_presentations.rewards.values]
    return stimulus_presentations


def add_licks_to_stimulus_presentations(stimulus_presentations, licks, time_window=[0, 0.75]):
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

    stimulus_presentations = stimulus_presentations.copy()
    lick_times = licks['timestamps'].values
    licks_each_stim = stimulus_presentations.apply(
        lambda row: lick_times[
            ((lick_times > row["start_time"] + time_window[0]) & (lick_times < row["start_time"] + time_window[1]))],
        axis=1,
    )
    stimulus_presentations["licks"] = licks_each_stim

    stimulus_presentations['licked'] = [True if len(licks) > 0 else False for licks in stimulus_presentations.licks.values]
    stimulus_presentations['lick_on_next_flash'] = stimulus_presentations['licked'].shift(-1)
    stimulus_presentations['lick_latency'] = [stimulus_presentations.iloc[row].licks[0] -
                                              stimulus_presentations.iloc[row].start_time if
                                              len(stimulus_presentations.iloc[row].licks) > 0 else
                                              np.nan for row in range(len(stimulus_presentations))]


    return stimulus_presentations


def add_reward_rate_to_trials(trials):
    '''
    Parameters:
    ____________
    trials: Pandas.DataFrame, SDK trials object

    Returns:
    ___________
    trials: Pandas.DataFrame
        with 'reward_rate_trials' column

    'reward_rate' is calculated by the SDK based on the rolling reward rate over trials (not stimulus presentations)
    https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/trials_processing.py#L941
    '''
    
    # need to calculate response latency because SDK doesnt include it for VBN
    if 'change_time_no_display_delay' in trials.keys(): # this means its from VBN
        trials['response_latency'] = trials.response_time-trials.change_time_no_display_delay
        trials['change_time'] = trials.change_time_no_display_delay
    
    last_time = 0
    reward_rate_by_frame = []
    if 'reward_rate' not in trials:
        trials['reward_rate'] = calculate_reward_rate(trials['response_latency'].values,
                                                      trials['start_time'],
                                                      window=.5)
    return trials

        
def add_reward_rate_to_stimulus_presentations(stimulus_presentations, trials):
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
    
    # limit to behavior block otherwise passive stimuli will be included and throw off the rolling reward rate
    stimulus_presentations = limit_to_behavior_session_block(stimulus_presentations)

    # need to calculate response latency because SDK doesnt include it for VBN
    if 'change_time_no_display_delay' in trials.keys(): # this means its from VBN
        trials['response_latency'] = trials.response_time-trials.change_time_no_display_delay
        trials['change_time'] = trials.change_time_no_display_delay
    
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


def add_change_trials_id_to_stimulus_presentations(stimulus_presentations, trials):
    """
    Add trials_id for change and sham change times to stimulus presentations by
    finding the closest change (or sham change) time to each stimulus start time.
    Column is called 'change_trials_id' to distinguish from 'trials_id' which is applied to all flashes in a trial
    If there is no corresponding change (or sham change) time, the change_trials_id is NaN
    i.e. this function only assigns a trials_id to the stimulus presentations corresponding to go and catch trials as defined in the trials table.
    If you want to assign trials_id to every stimulus presentation that is part of a trial, use
        add_trials_id_to_stimulus_presentations
    :param: stimulus_presentations: stimulus_presentations attribute of BehaviorOphysExperiment object, must have 'start_time'
    :param trials: trials attribute of BehaviorOphysExperiment object, must have 'change_time'
    """
    # for each stimulus_presentation, find the change_time that is closest to the start time
    # then add the corresponding trials_id to stimulus_presentations
    trials = trials.copy()
    stimulus_presentations = stimulus_presentations.copy()
    for row in range(len(stimulus_presentations)):
        this_start_time = stimulus_presentations.iloc[row].start_time
        # if its not the last row / stimulus presentation
        if row <= len(stimulus_presentations)-2:
            # get the start time of the next stimulus
            next_start_time = stimulus_presentations.iloc[row+1].start_time
        else:
            # if it is the last row, infer that the next start would be 750ms from now
            next_start_time = stimulus_presentations.iloc[row].start_time + 0.75
        # find the trial where change time falls between the current and next stimulus start times
        trial_data = trials[(trials.change_time>this_start_time) & (trials.change_time<=next_start_time)]
        if len(trial_data) > 0:
            trials_id = trial_data.index.values[0]
        else:
            trials_id = np.nan
        stimulus_presentations.loc[row, 'change_trials_id'] = trials_id
    return stimulus_presentations


def add_change_trial_outcomes_to_stimulus_presentations(stimulus_presentations, trials):
    """
    Add trials_id to stimulus presentations table, just for go and catch trials,
    then join columns of trials table indicating whether a go trial was hit or miss
    and whether a catch trial was a false alarm or correct reject
    with stimulus_presentations,
    relevant columns here are
    :param: stimulus_presentations: stimulus_presentations attribute of BehaviorOphysExperiment object, must have 'start_time'
    :param trials: trials attribute of BehaviorOphysExperiment object, must have 'change_time'
    """
    # add trials_id and merge to get trial type information
    stimulus_presentations = add_change_trials_id_to_stimulus_presentations(stimulus_presentations, trials)
    # only keep certain columns
    trials = trials[['change_time', 'hit', 'miss',
                     'false_alarm', 'correct_reject',
                     'reward_time', 'reward_volume']]
    # merge trials columns into stimulus_presentations

    stimulus_presentations = stimulus_presentations.reset_index().merge(trials, left_on='change_trials_id',
                                                                        right_on='trials_id', how='left')
    stimulus_presentations = stimulus_presentations.set_index('stimulus_presentations_id')
    return stimulus_presentations


def add_trials_id_to_stimulus_presentations(stimulus_presentations, trials):
    """
    Add trials_id to stimulus presentations by finding the closest trial start_time to each stimulus start_time
    This function adds a trials_id to every stimulus presentation based on which trial each stimulus belongs to
    If you want to assign trials_id only to stimulus presentations corresponding to go and catch trials as defined in the trials table, use:
        add_change_trials_id_to_stimulus_presentations
    :param: stimulus_presentations: stimulus_presentations attribute of BehaviorOphysExperiment object, must have 'start_time'
    :param trials: trials attribute of BehaviorOphysExperiment object, must have 'start_time'
    """

    # add trials_id and trial_stimulus_index
    stimulus_presentations = stimulus_presentations.copy()
    stimulus_presentations['trials_id'] = np.searchsorted(trials.start_time, stimulus_presentations.start_time) -1 # subtract 1 so its indexed at zero

    return stimulus_presentations


def add_trial_type_to_stimulus_presentations(stimulus_presentations, trials):
    """
    Add trials_id to stimulus presentations table
    then join trial type columns of trials table with stimulus_presentations
    trial types = ['aborted', 'auto_rewarded', 'go', 'catch']
    :param: stimulus_presentations: stimulus_presentations attribute of BehaviorOphysExperiment object, must have 'start_time'
    :param trials: trials attribute of BehaviorOphysExperiment object, must have 'start_time'
    """
    # add trials_id for all stimulus presentations and merge to get trial type information
    stimulus_presentations = add_trials_id_to_stimulus_presentations(stimulus_presentations, trials)
    # get trial type columns
    trials = trials[['go', 'catch', 'aborted', 'auto_rewarded']]
    # merge trial type columns into stimulus_presentations
    stimulus_presentations = stimulus_presentations.reset_index().merge(trials, on='trials_id', how='left')
    stimulus_presentations = stimulus_presentations.set_index('stimulus_presentations_id')
    return stimulus_presentations


def add_stimulus_count_within_trial_to_stimulus_presentations(stimulus_presentations, trials):
    """
    Add a column to stimulus_presentations that indicates how many stimuli have been shown since the trial start

    :param: stimulus_presentations: stimulus_presentations attribute of BehaviorOphysExperiment object, must have 'start_time'
    :param trials: trials attribute of BehaviorOphysExperiment object, must have 'start_time'
    """
    stimulus_presentations = stimulus_presentations.copy()
    # if trials_id is not a column of stimulus_presentations, add it
    if 'trials_id' not in stimulus_presentations.keys():
        stimulus_presentations = add_trials_id_to_stimulus_presentations(stimulus_presentations, trials)

    # label each stimulus presentation based on the number of stimuli since the trial start
    stimulus_presentations['stimulus_count_within_trial'] = None
    stimulus_count_within_trial = 0
    last_trial_id = -1 # start at -1 as baseline to compare with first trials_id (which is 0)
    # iterate over every stimulus
    for idx, row in stimulus_presentations.iterrows():
        trials_id = row.trials_id
        # if trials_id is the same as the last trial, increment stimulus_count_within_trial
        if trials_id == last_trial_id:
            stimulus_count_within_trial += 1
        else:
            stimulus_count_within_trial = 0
            # the current trial will be the last_trial_id for the next row / iteration
            last_trial_id = trials_id
        stimulus_presentations.at[idx, 'stimulus_count_within_trial'] = stimulus_count_within_trial

    return stimulus_presentations


def add_could_change_to_stimulus_presentations(stimulus_presentations, trials, licks):
    """
    Adds a column called could_change to the stimulus presentations table that indicates whether a given
    stimulus presentation could have been a change or sham change based on the known change time distribution.
    A stimulus could change if it is at least 4 stimulus flashes after the trial start,
    and there was not a lick on the previous stimulus, and the current or next stimulus is not omitted.
    :param stimulus_presentations: SDK stimulus presentations table
    :param trials: SDK trials table
    :return:
    """

    licks = licks.copy()
    trials = trials.copy()
    stimulus_presentations = stimulus_presentations.copy()
    # if trials_id is not a column of stimulus_presentations, add it
    if 'trials_id' not in stimulus_presentations.keys():
        stimulus_presentations = add_trials_id_to_stimulus_presentations(stimulus_presentations, trials)
        # if trials_id is not a column of stimulus_presentations, add it
    if 'licks' not in stimulus_presentations.keys():
        stimulus_presentations = add_licks_to_stimulus_presentations(stimulus_presentations, licks)
        # if trials_id is not a column of stimulus_presentations, add it
    if 'stimulus_count_within_trial' not in stimulus_presentations.keys():
        stimulus_presentations = add_stimulus_count_within_trial_to_stimulus_presentations(stimulus_presentations, trials)

    # add previous_image_name
    stimulus_presentations['previous_image_name'] = stimulus_presentations['image_name'].shift()
    # add previous_response_on_trial
    stimulus_presentations['previous_response_on_trial'] = False
    stimulus_presentations['previous_change_on_trial'] = False
    # set 'stimulus_presentations_id' and 'trials_id' as indices to speed lookup
    stimulus_presentations = stimulus_presentations.reset_index().set_index(['stimulus_presentations_id', 'trials_id'])
    for idx, row in stimulus_presentations.iterrows():
        stim_id, trials_id = idx
        # get all stimuli before the current on the current trial
        mask = (stimulus_presentations.index.get_level_values(0) < stim_id) & (
            stimulus_presentations.index.get_level_values(1) == trials_id)
        # check to see if any previous stimuli have a response lick
        stimulus_presentations.at[idx, 'previous_response_on_trial'] = stimulus_presentations[mask]['licked'].any()
        stimulus_presentations.at[idx, 'previous_change_on_trial'] = stimulus_presentations[mask]['is_change'].any()
    # set the index back to being just 'stimulus_presentations_id'
    stimulus_presentations = stimulus_presentations.reset_index().set_index('stimulus_presentations_id')

    # add could_change column to indicate whether the stimulus presentation falls into the range in which changes can occur
    stimulus_presentations['could_change'] = False
    for idx, row in stimulus_presentations.iterrows():
        # check if we meet conditions where a change could occur on this stimulus (at least 4th flash of trial, no previous change on trial)
        # changes can only happen after the 4th stimulus within a trial
        # changes can only happen if there is no previous response on the trial (i.e. not on aborted trials)
        # changes can only happen if there is no previous change on that trial (i.e. not during the grace period / consumption window)
        # changes can only happen if the previous stimulus was not omitted
        if row['stimulus_count_within_trial'] >= 4 and row['previous_response_on_trial'] is False \
                and row['previous_change_on_trial'] is False \
                and row['image_name'] != 'omitted' \
                and row['previous_image_name'] != 'omitted':
            stimulus_presentations.at[idx, 'could_change'] = True

    return stimulus_presentations


def add_time_from_last_change_to_stimulus_presentations(stimulus_presentations):
    '''
    Adds a column to stimulus_presentations called 'time_from_last_change',
    which is the time, in seconds since the last image change

    ARGS: SDK stimulus presentations table
    MODIFIES: session.stimulus_presentations
    RETURNS: stimulus_presentations
    '''
    stimulus_times = stimulus_presentations["start_time"].values
    change_times = stimulus_presentations.query('is_change')['start_time'].values
    time_from_last_change = utilities.time_from_last(stimulus_times, change_times)
    stimulus_presentations["time_from_last_is_change"] = time_from_last_change

    return stimulus_presentations


def add_time_from_last_omission_to_stimulus_presentations(stimulus_presentations):
    '''
    Adds a column to stimulus_presentations called 'time_from_last_omission',
    which is the time, in seconds since the last stimulus omission

    ARGS: SDK stimulus presentations table
    MODIFIES: session.stimulus_presentations
    RETURNS: stimulus_presentations
    '''
    stimulus_times = stimulus_presentations["start_time"].values
    omission_times = stimulus_presentations.query('omitted')['start_time'].values
    time_from_last_omission = utilities.time_from_last(stimulus_times, omission_times)
    stimulus_presentations["time_from_last_omitted"] = time_from_last_omission

    return stimulus_presentations


def add_time_from_last_lick_to_stimulus_presentations(stimulus_presentations, licks):
    '''
    Adds a column to stimulus_presentations called 'time_from_last_lick',
    which is the time, in seconds since the last lick response

    ARGS: SDK stimulus presentations table, SDK licks table
    MODIFIES: session.stimulus_presentations
    RETURNS: stimulus_presentations
    '''

    stimulus_presentations = stimulus_presentations.copy()

    stimulus_times = stimulus_presentations["start_time"].values
    lick_times = licks.timestamps.values
    time_from_last_lick = utilities.time_from_last(stimulus_times, lick_times)
    stimulus_presentations["time_from_last_lick"] = time_from_last_lick

    return stimulus_presentations


def add_time_from_last_reward_to_stimulus_presentations(stimulus_presentations, rewards):
    '''
    Adds a column to stimulus_presentations called 'time_from_last_reward',
    which is the time, in seconds since the last reward was delivered

    ARGS: SDK stimulus presentations table, SDK rewards table
    MODIFIES: session.stimulus_presentations
    RETURNS: stimulus_presentations
    '''
    stimulus_times = stimulus_presentations["start_time"].values
    reward_times = rewards.timestamps.values
    time_from_last_reward = utilities.time_from_last(stimulus_times, reward_times)
    stimulus_presentations["time_from_last_reward"] = time_from_last_reward

    return stimulus_presentations


def add_time_from_last_to_stimulus_presentations(stimulus_presentations, licks, rewards):
    """
    Adds multiple columns to stimulus presentations with the number of stimuli that have elapsed
    since the last change, omission, lick, and reward using the Boolean columns 'is_change', 'omitted', 'licked', and 'rewarded'
    If these Boolean columns are not included in stimulus_presentations, they will be added

    :param stimulus_presentations: SDK stimulus_presentations table
    :return: stimulus_presentations with 'stimulus_count_from_last' columns added

    """
    stimulus_presentations = stimulus_presentations.copy()
    stimulus_presentations = add_time_from_last_change_to_stimulus_presentations(stimulus_presentations)
    stimulus_presentations = add_time_from_last_omission_to_stimulus_presentations(stimulus_presentations)
    stimulus_presentations = add_time_from_last_reward_to_stimulus_presentations(stimulus_presentations, rewards)
    stimulus_presentations = add_time_from_last_lick_to_stimulus_presentations(stimulus_presentations, licks)

    return stimulus_presentations


def stimulus_count_from_last(stimulus_presentations, column_to_count_from='is_change'):
    """
    Takes a Boolean column in stimulus_presentations and counts the number of stimulus presentations
    after each True instance of the column_to_count_from column value,
    adds a column called 'stimulus_count_from_last_'+column_to_count_from that contains that count value

    Example: if using column_to_count_from = 'is_change', and the first 10 stimuli are [F, F, F, F, T, F, F, F, F],
    the [stimulus_count_since_last_is_change' column will be [1, 2, 3, 4, 0, 1, 2, 3, 4]

    :param stimulus_presentations: SDK stimulus presentations table, must include Boolean column for column_to_count_from
    :param column_to_count_from: column value in stimulus_presentations to use to count number of stimulus presentations
                                since the column value was last True
    :return: stimulus_presentations with added column 'stimulus_count_from_last_'+column_to_count_from
    """
    stimulus_presentations = stimulus_presentations.copy()
    # label each stimulus presentation based on the number of stimuli since the trial start
    stimulus_presentations['stimulus_count_from_last_' + column_to_count_from] = None
    stimulus_count = 0
    # iterate over every stimulus
    for idx, row in stimulus_presentations.iterrows():
        current_value = row[column_to_count_from]
        # if current_value is not True, increment the counter
        if current_value == False:
            stimulus_count += 1
        elif current_value == True:  # if current value is True, reset the counter
            stimulus_count = 0
        else:
            stimulus_count = np.nan
        stimulus_presentations.at[idx, 'stimulus_count_from_last_' + column_to_count_from] = stimulus_count

    return stimulus_presentations


def add_stimulus_count_from_last_to_stimulus_presentations(stimulus_presentations, licks, rewards):
    """
    Adds multiple columns to stimulus presentations with the number of stimuli that have elapsed
    since the last change, omission, lick, and reward using the Boolean columns 'is_change', 'omitted', 'licked', and 'rewarded'
    If these Boolean columns are not included in stimulus_presentations, they will be added

    :param stimulus_presentations: SDK stimulus_presentations table
    :return: stimulus_presentations with 'stimulus_count_from_last' columns added

    """
    stimulus_presentations = stimulus_presentations.copy()
    if 'licked' not in stimulus_presentations.keys():
        stimulus_presentations = add_licks_to_stimulus_presentations(stimulus_presentations, licks,
                                                                              time_window=[0, 0.75])
    if 'rewarded' not in stimulus_presentations.keys():
        stimulus_presentations = add_rewards_to_stimulus_presentations(stimulus_presentations, rewards,
                                                                                time_window=[0, 0.75])
    for column in ['is_change', 'omitted', 'licked', 'rewarded']:
        stimulus_presentations = stimulus_count_from_last(stimulus_presentations, column_to_count_from=column)

    return stimulus_presentations


def add_behavior_info_to_stimulus_presentations(stimulus_presentations, trials, licks, rewards,
                                                running_speed, eye_tracking):
    """
    Adds a variety of useful columns to the stimulus presentations table by incorporating information
    from the trials, licks, rewards, running_speed and eye_tracking tables.
    Useful to filter stimuli based on behavioral information.
    Added columns include trials_id, go, catch, aborted, auto-rewarded, hit, miss, false_alarm, correct_reject,
    as well as lick and reward times for each stimulus presentation,
    reward rate in rewards / min (computed across trials not stimulus presentations),
    and mean running speed and pupil width for each stimulus

    :param stimulus_presentations: SDK stimulus_presentations table
    :param trials: SDK trials table
    :param licks: SDK licks table
    :param rewards: SDK rewards table
    :param running_speed: SDK running_speed table
    :param eye_tracking: SDK eye tracking table, will use pupil_width column of eye tracking table
    :return:
    """

    stimulus_presentations = stimulus_presentations.copy()
    # add columns from trials table: ['change_time', 'hit', 'miss', 'false_alarm', 'correct_reject', 'reward_time', 'reward_volume']
    # applied only to stimulus presentations corresponding to change and sham change times (go and catch)
    stimulus_presentations = add_change_trials_id_to_stimulus_presentations(stimulus_presentations, trials)
    stimulus_presentations = add_change_trial_outcomes_to_stimulus_presentations(stimulus_presentations, trials)
    # add columns from trials table: ['go', 'catch', 'aborted', 'auto_rewarded']
    # applied to all stimulus presentations belonging to that trial
    stimulus_presentations = add_trials_id_to_stimulus_presentations(stimulus_presentations, trials)
    stimulus_presentations = add_trial_type_to_stimulus_presentations(stimulus_presentations, trials)
    # add the timing of licks and rewards for each stimulus presentation
    stimulus_presentations = add_licks_to_stimulus_presentations(stimulus_presentations, licks, time_window=[0, 0.75])
    stimulus_presentations = add_rewards_to_stimulus_presentations(stimulus_presentations, rewards, time_window=[0, 0.75])
    # add reward rate across trials to stimulus presentations
    stimulus_presentations = add_reward_rate_to_stimulus_presentations(stimulus_presentations, trials)
    # add mean running speed and pupil diameter for each stimulus presentation
    stimulus_presentations = add_mean_running_speed_to_stimulus_presentations(stimulus_presentations, running_speed, time_window=[0, 0.75])
    stimulus_presentations = add_mean_pupil_to_stimulus_presentations(stimulus_presentations, eye_tracking,
                                             column_to_use='pupil_width', time_window=[0, 0.75])

    return stimulus_presentations


def add_timing_info_to_stimulus_presentations(stimulus_presentations, trials, licks, rewards):
    """
    Annotate stimulus presentations table with information about timing relative to events of interest,
    such as time from last change / omission / lick / reward,
    whether a stimulus was before or after a change or an omission,
    and whether a given stimulus could have been a change based on known change time distribution (4-12 flashes from the time of the last lick)

    :param stimulus_presentations: SDK stimulus presentations table
    :param trials: SDK trials table
    :param licks: SDK licks table
    :param rewards: SDK rewards table
    :return:
    """

    # add columns indicating wither previous or subsequent stimulus presentations are a change or omission
    stimulus_presentations['pre_change'] = stimulus_presentations['is_change'].shift(-1)
    stimulus_presentations['post_change'] = stimulus_presentations['is_change'].shift(1)
    stimulus_presentations['pre_omitted'] = stimulus_presentations['omitted'].shift(-1)
    stimulus_presentations['post_omitted'] = stimulus_presentations['omitted'].shift(1)

    # add stimulus count within each behavioral trial
    stimulus_presentations = add_stimulus_count_within_trial_to_stimulus_presentations(stimulus_presentations, trials)
    # add stimulus count from last change / omission / lick / reward
    stimulus_presentations = add_stimulus_count_from_last_to_stimulus_presentations(stimulus_presentations, licks, rewards)
    # add time elapsed since last change / omission / lick / reward
    stimulus_presentations = add_time_from_last_to_stimulus_presentations(stimulus_presentations, licks, rewards)

    # add column indicating whether a given stimulus presentations could have been a change stimulus
    # given the known change time distribution and whether or not the mouse licked on a given stimulus (thus aborting and reseting the trial)
    stimulus_presentations = add_could_change_to_stimulus_presentations(stimulus_presentations, trials, licks)

    return stimulus_presentations