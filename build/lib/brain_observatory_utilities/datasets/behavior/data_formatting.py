import pandas as pd
import numpy as np
from brain_observatory_utilities.utilities.general_utilities import get_trace_average
import brain_observatory_utilities.datasets.behavior.data_access as data_access
from brain_observatory_utilities.utilities import general_utilities



def limit_stimulus_presentations_to_change_detection(stimulus_presentations):
    '''
    if column 'stimulus_block_name' is in stimulus_presentations table (as in SDK v2.16.2),
    limit stimulus presentations table to the change detection block
    '''
    if 'stimulus_block_name' in stimulus_presentations:
        stimulus_presentations = stimulus_presentations[stimulus_presentations.stimulus_block_name.str.contains('change_detection')]
        # change a few columns from type Boolean to bool (they were previously Boolean so they could contain NaNs for non-change detection stim blocks)
        # stimulus_presentations = convert_boolean_cols_to_bool(stimulus_presentations)
    return stimulus_presentations


def convert_boolean_cols_to_bool(stimulus_presentations):
    '''
    For any dataframe containing columns derived from the stimulus_presentations table,
    go through all columns and identify those that are type boolean (which occurs when the column has NaNs and bools)
    and convert NaNs to False then set dtype to bool.

    This is needed because many operations fail on columns of type boolean.
    Some columns in stimulus_presentations are boolean in new SDK outputs because of the new stimulus_blocks,
    as many values specific to change_detection task are set to NaN in other stimulus blocks, which
    means that the entire column gets the dtype boolean instead of bool.
    '''
    for column in stimulus_presentations.columns.values:
        try:
            if type(stimulus_presentations[column].dtype).__name__ == 'BooleanDtype':
                row_ids = stimulus_presentations[stimulus_presentations[column].isnull()].index
                stimulus_presentations.loc[row_ids, column] = False
                stimulus_presentations[column] = stimulus_presentations[column].astype('bool')
        except:
            if stimulus_presentations[column].dtype == 'boolean':
                # remove NaNs and make bool
                row_ids = stimulus_presentations[stimulus_presentations[column].isnull()].index
                stimulus_presentations.loc[row_ids, column] = False
                stimulus_presentations[column] = stimulus_presentations[column].astype('bool')
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
        lambda row: get_trace_average(
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
        lambda row: get_trace_average(
            eye_tracking_timeseries,
            eye_tracking['timestamps'].values,
            row["start_time"] + time_window[0],
            row["start_time"] + time_window[1],
        ), axis=1,)
    stimulus_presentations["mean_"+column_to_use] = mean_pupil_around_stimulus
    return stimulus_presentations


def add_rewards_to_stimulus_presentations(stimulus_presentations,
                                          rewards,
                                          time_window=[0, 3]):
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


def calculate_reward_rate(response_latency=None,
                          starttime=None,
                          window=0.75,
                          trial_window=25,
                          initial_trials=10):

    assert len(response_latency) == len(starttime)

    df = pd.DataFrame({'response_latency': response_latency,
                       'starttime': starttime})

    # adds a column called reward_rate to the input dataframe
    # the reward_rate column contains a rolling average of rewards/min
    # window sets the window in which a response is considered correct,
    # so a window of 1.0 means licks before 1.0 second are considered correct
    #
    # Reorganized into this unit-testable form by Nick Cain April 25 2019

    reward_rate = np.zeros(len(df))
    # make the initial reward rate infinite,
    # so that you include the first trials automatically.
    reward_rate[:initial_trials] = np.inf

    for trial_number in range(initial_trials, len(df)):

        min_index = np.max((0, trial_number - trial_window))
        max_index = np.min((trial_number + trial_window, len(df)))
        df_roll = df.iloc[min_index:max_index]

        # get a rolling number of correct trials
        correct = len(df_roll[df_roll.response_latency < window])

        # get the time elapsed over the trials
        time_elapsed = df_roll.starttime.iloc[-1] - df_roll.starttime.iloc[0]

        # calculate the reward rate, rewards/min
        reward_rate_on_this_lap = correct / time_elapsed * 60

        reward_rate[trial_number] = reward_rate_on_this_lap
    return reward_rate


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

    last_time = 0
    reward_rate_by_frame = []

    # need to calculate response latency because SDK doesnt include it for VBN
    if 'change_time_no_display_delay' in trials.keys(): # this means its from VBN
        trials['response_latency'] = trials.response_time-trials.change_time_no_display_delay
        trials['change_time'] = trials.change_time_no_display_delay

    # recalculate reward_rate for trials 
    trials['reward_rate'] = calculate_reward_rate(trials['response_latency'].values, trials['start_time'])

    trials = trials[trials['aborted'] == False]  # NOQA
    for change_time in trials.change_time.values:
        reward_rate = trials[trials.change_time ==  # NOQA
                             change_time].reward_rate.values[0]
        # add reward rate value from trial to all stim presentations belonging to that trial
        for start_time in stimulus_presentations.start_time: 
            if (start_time < change_time) and (start_time > last_time):
                reward_rate_by_frame.append(reward_rate)
                last_time = start_time
    # fill the last flashes with last value
    for i in range(len(stimulus_presentations) - len(reward_rate_by_frame)):
        reward_rate_by_frame.append(reward_rate_by_frame[-1])

    stimulus_presentations['reward_rate'] = reward_rate_by_frame
    return stimulus_presentations


def add_engagement_state_to_stimulus_presentations(
        stimulus_presentations, trials):
    """
    Add 'engaged' Boolean column and 'engagement_state' string ('engaged' or 'disengaged')
    using threshold of  2 rewards per minute, with reward_rate calculated as in the SDK by the
    function add_reward_rate_to_stimulus_presentations() in this repo, which is a copy of what is done in the SDK.
    Previously this function pulled directly from the SDK, but the funciton was added to a class and is no longer directly accessible.

    :param stimulus_presentations: stimulus_presentations attribute of BehaviorOphysExperiment
    :param trials: trials attribute of BehaviorOphysExperiment object
    :return: stimulus_presentations with columns added: 'reward_rate', 'engaged', 'engagement_state'
    """

    if 'reward_rate' not in stimulus_presentations.keys():
        stimulus_presentations = add_reward_rate_to_stimulus_presentations(stimulus_presentations, trials)
    
    reward_threshold = 2

    stimulus_presentations['engaged'] = [x > reward_threshold for x in stimulus_presentations['reward_rate'].values]
    stimulus_presentations['engagement_state'] = ['engaged' if engaged else 'disengaged' for engaged in stimulus_presentations['engaged'].values]

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
            indices = stimulus_presentations[(
                stimulus_presentations[time_column] >= epoch_times[i])].index.values
        stimulus_presentations.loc[indices, 'epoch'] = i
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
        query_string = 'change_time > @start_time - 0.5 and change_time < @start_time + 0.5'
        trials_id = (np.abs(start_time - trials.query(query_string)['change_time']))
        if len(trials_id) == 1:
            trials_id = trials_id.idxmin()
        else:
            trials_id = np.nan
        stimulus_presentations.loc[idx, 'trials_id'] = trials_id
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
    stimulus_presentations = stimulus_presentations.reset_index().merge(
        trials, on='trials_id', how='left')
    stimulus_presentations = stimulus_presentations.set_index(
        'stimulus_presentations_id')
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
    if 'change_time' not in trials.keys():
        trials['change_time'] = trials['change_time_no_display_delay']
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


def add_time_from_last_change_to_stimulus_presentations(stimulus_presentations):
    '''
    Adds a column to stimulus_presentations called 'time_from_last_change', which is the time, in seconds since the last image change

    ARGS: SDK session object
    MODIFIES: session.stimulus_presentations
    RETURNS: stimulus_presentations
    '''
    stimulus_times = stimulus_presentations["start_time"].values
    change_times = stimulus_presentations.query(
        'is_change')['start_time'].values
    time_from_last_change = general_utilities.time_from_last(stimulus_times, change_times)
    stimulus_presentations["time_from_last_change"] = time_from_last_change

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
    omission_times = stimulus_presentations[stimulus_presentations['omitted']==True]['start_time'].values
    time_from_last_omission = general_utilities.time_from_last(stimulus_times, omission_times)
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
    time_from_last_lick = general_utilities.time_from_last(stimulus_times, lick_times)
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
    time_from_last_reward = general_utilities.time_from_last(stimulus_times, reward_times)
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


def add_n_to_stimulus_presentations(stimulus_presentations):
    """
    Adds a column to stimulus_presentations called 'n_after_change',
    which is the number of stimulus presentations that have occurred since the last change.
    It will also add a column called 'n_after_omission',
    which is the number of stimulus presentations that have occurred since the last omission,
    before the next change.
    If there is no omission, this value will be -1.
    Presentations before the first change or omission will have a value of -1.
    It will also add a column called 'n_before_change',
    which is the number of stimulus presentations that have occurred before the next change.
    Presentations after the last change will have a value of -1.
    Presentations before the first change will also have a value of -1.
    0 for 'n_after_change' and 'n_before_change' indicates the change itself.
    0 for 'n_after_omission' indicates the omission itself.

    Parameters
    ----------
    stimulus_presentations : pd.DataFrame
        stimulus_presentations table from BehaviorOphysExperiment

    Returns
    -------
    stimulus_presentations : pd.DataFrame
        stimulus_presentations table with 'n_after_change', 'n_after_omission', and 'n_before_change' columns added
    """

    change_ind = stimulus_presentations[stimulus_presentations['is_change']].index.values

    # Adding n_after_change
    # -1 indicates before the first change
    n_after_change = np.zeros(len(stimulus_presentations)) - 1
    for i in range(1, len(change_ind)):
        n_after_change[change_ind[i - 1]: change_ind[i]
                       ] = np.arange(0, change_ind[i] - change_ind[i - 1]).astype(int)
    n_after_change[change_ind[i]:] = np.arange(
        0, len(stimulus_presentations) - change_ind[i]).astype(int)
    stimulus_presentations['n_after_change'] = n_after_change

    # Adding n_before_change
    # -1 indicates after the last and before the first change
    n_before_change = np.zeros(len(stimulus_presentations)) - 1
    for i in range(len(change_ind) - 1):
        n_before_change[change_ind[i] + 1: change_ind[i + 1] + 1] = np.arange(
            change_ind[i + 1] - change_ind[i] - 1, -1, -1).astype(int)
    stimulus_presentations['n_before_change'] = n_before_change

    # Adding n_after_omission
    # -1 indicates before the first omission or
    n_after_omission = np.zeros(len(stimulus_presentations)) - 1
    # from the next change till the next omission # noqa E114,E116
    # if there are no omissions, n_after_omission will be all -1
    # and 'omitted' will be added and assigned to False
    if 'omitted' in stimulus_presentations.columns:
        omission_ind = stimulus_presentations[stimulus_presentations['omitted']].index.values
        for i in range(len(omission_ind)):
            if change_ind[-1] > omission_ind[i]:  # if there is a change after the omission
                next_change_ind = change_ind[change_ind > omission_ind[i]][0]
                n_after_omission[omission_ind[i]: next_change_ind] = np.arange(
                    0, next_change_ind - omission_ind[i]).astype(int)
            else:
                n_after_omission[omission_ind[i]:] = np.arange(
                    0, len(stimulus_presentations) - omission_ind[i]).astype(int)
    else:
        stimulus_presentations['omitted'] = False
    stimulus_presentations['n_after_omission'] = n_after_omission

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


def annotate_stimuli(dataset, inplace=False):
    '''
    adds the following columns to the stimulus_presentations table, facilitating calculation
    of behavior performance based entirely on the stimulus_presentations table:

    'trials_id': the corresponding ID of the trial in the trials table in which the stimulus occurred
    'previous_image_name': the name of the stimulus on the last flash (will list 'omitted' if last stimulus is omitted)
    'next_start_time': The time of the next stimulus start (including the time of the omitted stimulus if the next stimulus is omitted)
    'auto_rewarded': True for trials where rewards were delivered regardless of animal response
    'trial_stimulus_index': index of the given stimulus on the current trial. For example, the first stimulus in a trial has index 0, the second stimulus in a trial has index 1, etc
    'response_lick': Boolean, True if a lick followed the stimulus
    'response_lick_times': list of all lick times following this stimulus
    'response_lick_latency': time difference between first lick and stimulus
    'previous_response_on_trial': Boolean, True if there has been a lick to a previous stimulus on this trial
    'could_change': Boolean, True if the stimulus met the conditions that would have allowed
                    to be chosen as the change stimulus by camstim:
                        * at least the fourth stimulus flash in the trial
                        * not preceded by any licks on that trial

    Parameters:
    -----------
    dataset : BehaviorSession or BehaviorOphysSession object
        an SDK session object
    inplace : Boolean
        If True, operates on the dataset.stimulus_presentations object directly and returns None
        If False (default), operates on a copy and returns the copy

    Returns:
    --------
    Pandas.DataFrame (if inplace == False)
    None (if inplace == True)
    '''

    if inplace:
        stimulus_presentations = dataset.stimulus_presentations
    else:
        stimulus_presentations = dataset.stimulus_presentations.copy()

    # limit to change detection block
    stimulus_presentations = limit_stimulus_presentations_to_change_detection(stimulus_presentations)

    # add previous_image_name
    stimulus_presentations['previous_image_name'] = stimulus_presentations['image_name'].shift()

    # add next_start_time
    stimulus_presentations['next_start_time'] = stimulus_presentations['start_time'].shift(-1)

    # add trials_id and trial_stimulus_index
    stimulus_presentations['trials_id'] = None
    stimulus_presentations['trial_stimulus_index'] = None
    last_trial_id = -1
    trial_stimulus_index = 0

    # add response_lick, response_lick_times, response_lick_latency
    stimulus_presentations['response_lick'] = False
    stimulus_presentations['response_lick_times'] = None
    stimulus_presentations['response_lick_latency'] = None

    # make a copy of trials with 'start_time' as index to speed lookup
    trials = dataset.trials.copy().reset_index().set_index('start_time')

    # make a copy of licks with 'timestamps' as index to speed lookup
    licks = dataset.licks.copy().reset_index().set_index('timestamps')

    # iterate over every stimulus
    for idx, row in stimulus_presentations.iterrows():
        # trials_id is last trials_id with start_time <= stimulus_time
        try:
            trials_id = trials.loc[:row['start_time']].iloc[-1]['trials_id']
        except IndexError:
            trials_id = -1
        stimulus_presentations.loc[idx, 'trials_id'] = trials_id

        if trials_id == last_trial_id:
            trial_stimulus_index += 1
        else:
            trial_stimulus_index = 0
            last_trial_id = trials_id
        stimulus_presentations.loc[idx,
                                  'trial_stimulus_index'] = trial_stimulus_index

        # note the `- 1e-9` acts as a <, as opposed to a <=
        stim_licks = licks.loc[row['start_time']:row['next_start_time'] - 1e-9].index.to_list()

        stimulus_presentations.loc[idx, 'response_lick_times'] = stim_licks
        if len(stim_licks) > 0:
            stimulus_presentations.loc[idx, 'response_lick'] = True
            stimulus_presentations.loc[idx,
                                      'response_lick_latency'] = stim_licks[0] - row['start_time']

    # merge in auto_rewarded column from trials table
    stimulus_presentations = stimulus_presentations.reset_index().merge(
        dataset.trials[['auto_rewarded']],
        on='trials_id',
        how='left',
    ).set_index('stimulus_presentations_id')

    # add previous_response_on_trial
    stimulus_presentations['previous_response_on_trial'] = False
    # set 'stimulus_presentations_id' and 'trials_id' as indices to speed
    # lookup
    stimulus_presentations = stimulus_presentations.reset_index(
    ).set_index(['stimulus_presentations_id', 'trials_id'])
    for idx, row in stimulus_presentations.iterrows():
        stim_id, trials_id = idx
        # get all stimuli before the current on the current trial
        mask = (stimulus_presentations.index.get_level_values(0) < stim_id) & (
            stimulus_presentations.index.get_level_values(1) == trials_id)
        # check to see if any previous stimuli have a response lick
        stimulus_presentations.at[idx, 'previous_response_on_trial'] = stimulus_presentations[mask]['response_lick'].any()
    # set the index back to being just 'stimulus_presentations_id'
    stimulus_presentations = stimulus_presentations.reset_index().set_index('stimulus_presentations_id')

    # add could_change
    stimulus_presentations['could_change'] = False
    for idx, row in stimulus_presentations.iterrows():
        # check if we meet conditions where a change could occur on this
        # stimulus (at least 4th flash of trial, no previous change on trial)
        if row['trial_stimulus_index'] >= 4 and row['previous_response_on_trial'] is False and row[
                'image_name'] != 'omitted' and row['previous_image_name'] != 'omitted':
            stimulus_presentations.loc[idx, 'could_change'] = True

    if inplace is False:
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

    trials = trials.copy()
    if 'change_time' not in trials.keys():
        trials['change_time'] = trials['change_time_no_display_delay']
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

    trials = trials.copy()
    if 'change_time' not in trials.keys():
        trials['change_time'] = trials['change_time_no_display_delay']
        
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


def get_annotated_stimulus_presentations(
        ophys_experiment, epoch_duration_mins=10):
    """
    Takes in an ophys_experiment dataset object and returns the stimulus_presentations table with additional columns.
    Adds several useful columns to the stimulus_presentations table, including the mean running speed and pupil diameter for each stimulus,
    the times of licks for each stimulus, the rolling reward rate, an identifier for 10 minute epochs within a session,
    whether or not a stimulus was a pre-change or pre or post omission, and whether change stimuli were hits or misses
    :param ophys_experiment: obj
        AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501
    :return: stimulus_presentations attribute of BehaviorOphysExperiment, with additional columns added
    """
    stimulus_presentations = ophys_experiment.stimulus_presentations
    # limit to change detection block
    stimulus_presentations = limit_stimulus_presentations_to_change_detection(stimulus_presentations)
    
    # add licks
    stimulus_presentations = add_licks_to_stimulus_presentations(
        stimulus_presentations, ophys_experiment.licks, time_window=[0, 0.75])
    # add running
    stimulus_presentations = add_mean_running_speed_to_stimulus_presentations(
        stimulus_presentations, ophys_experiment.running_speed, time_window=[0, 0.75])
    # if hasattr('ophys_experiment', 'eye_tracking'):
    try:
        stimulus_presentations = add_mean_pupil_to_stimulus_presentations(
            stimulus_presentations,
            ophys_experiment.eye_tracking,
            column_to_use='pupil_width',
            time_window=[0, 0.75])
    except Exception as e:
        print('could not add mean pupil to stimulus presentations, length of eye_tracking attribute is', len(
                ophys_experiment.eye_tracking))
        print(e)

    # add trials info
    try:  # not all session types have catch trials or omissions
        stimulus_presentations = add_trials_data_to_stimulus_presentations_table(
            stimulus_presentations, ophys_experiment.trials)
        # add time from last change
        stimulus_presentations = add_time_from_last_change_to_stimulus_presentations(stimulus_presentations)
        # add pre-change
        stimulus_presentations['pre_change'] = stimulus_presentations['is_change'].shift(-1)
        # add licked Boolean
        stimulus_presentations['licked'] = [True if len(
            licks) > 0 else False for licks in stimulus_presentations.licks.values]
        stimulus_presentations['lick_on_next_flash'] = stimulus_presentations['licked'].shift(-1)
        # add omission annotation
        stimulus_presentations['pre_omitted'] = stimulus_presentations['omitted'].shift(-1)
        stimulus_presentations['post_omitted'] = stimulus_presentations['omitted'].shift(1)
    # add repeat number
    except Exception as e:
        print(e)

    # add reward rate
    stimulus_presentations = add_reward_rate_to_stimulus_presentations(
        stimulus_presentations, ophys_experiment.trials)
    # add engagement state based on reward rate 
    stimulus_presentations = add_engagement_state_to_stimulus_presentations(
            stimulus_presentations, ophys_experiment.trials)
    # add epochs
    stimulus_presentations = add_epochs_to_stimulus_presentations(
        stimulus_presentations,
        time_column='start_time',
        epoch_duration_mins=epoch_duration_mins)

    return stimulus_presentations



def calculate_response_matrix(stimuli, aggfunc=np.mean, sort_by_column=True, engaged_only=True):
    '''
    calculates the response matrix for each individual image pair in the `stimulus` dataframe

    Parameters:
    -----------
    stimuli: Pandas.DataFrame
        From experiment.stimulus_presentations, after annotating as follows:
            annotate_stimuli(experiment, inplace = True)
    aggfunc: function
        function to apply to calculation. Default = np.mean
        other options include np.size (to get counts) or np.median
    sort_by_column: Boolean
        if True (default), sorts outputs by column means
    engaged_only: Boolean
        If True (default), calculates only on engaged trials
        Will throw an assertion error if True and 'engagement_state' column does not exist

    Returns:
    --------
    Pandas.DataFrame
        matrix of response probabilities for each image combination
        index = previous image
        column = current image
        catch trials are on diagonal

    '''
    stimuli_to_analyze = stimuli.query(
        'auto_rewarded == False and could_change == True and image_name != "omitted" and previous_image_name != "omitted"')
    if engaged_only:
        assert 'engagement_state' in stimuli_to_analyze.columns, 'stimuli must have column called "engagement_state" if passing engaged_only = True'
        stimuli_to_analyze = stimuli_to_analyze.query(
            'engagement_state == "engaged"')

    response_matrix = pd.pivot_table(
        stimuli_to_analyze,
        values='response_lick',
        index=['previous_image_name'],
        columns=['image_name'],
        aggfunc=aggfunc
    ).astype(float)

    if sort_by_column:
        sort_by = response_matrix.mean(axis=0).sort_values().index
        response_matrix = response_matrix.loc[sort_by][sort_by]

    response_matrix.index.name = 'previous_image_name'

    return response_matrix


def get_licks_df(ophys_experiment):
    '''
    Creates a dataframe containing columns for 'timestamps', 'licks', where values are from
    a binary array of the length of stimulus timestamps where frames with no lick are 0 and frames with a lick are 1,
    and a column called 'lick_rate' with values of 'licks' averaged over a 6 frame window to get licks per 100ms,
    Can be used to plot stim triggered average lick rate
    Parameters:
    -----------
    ophys_experiment: obj
        AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501

    Returns:
    --------
    Pandas.DataFrame with columns 'timestamps', 'licks', and 'lick_rate' in units of licks / 100ms

    '''
    timestamps = ophys_experiment.stimulus_timestamps.copy()
    licks = ophys_experiment.licks.copy()
    lick_array = np.zeros(timestamps.shape)
    lick_array[licks.frame.values] = 1
    licks_df = pd.DataFrame(data=timestamps, columns=['timestamps'])
    licks_df['licks'] = lick_array
    licks_df['lick_rate'] = licks_df['licks'].rolling(
        window=6, min_periods=1, win_type='triang').mean()

    return licks_df


def calculate_dprime_matrix(stimuli, sort_by_column=True, engaged_only=True):
    '''
    calculates the d' matrix for each individual image pair in the `stimulus` dataframe

    Parameters:
    -----------
    stimuli: Pandas.DataFrame
        From experiment.stimulus_presentations, after annotating as follows:
            annotate_stimuli(experiment, inplace = True)
    sort_by_column: Boolean
        if True (default), sorts outputs by column means
    engaged_only: Boolean
        If True (default), calculates only on engaged trials
        Will throw an assertion error if True and 'engagement_state' column does not exist

    Returns:
    --------
    Pandas.DataFrame
        matrix of d' for each image combination
        index = previous image
        column = current image
        catch trials are on diagonal

    '''
    if engaged_only:
        assert 'engagement_state' in stimuli.columns, 'stimuli must have column called "engagement_state" if passing engaged_only = True'

    response_matrix = calculate_response_matrix(
        stimuli,
        aggfunc=np.mean,
        sort_by_column=sort_by_column,
        engaged_only=engaged_only)

    d_prime_matrix = response_matrix.copy()
    for row in response_matrix.columns:
        for col in response_matrix.columns:
            d_prime_matrix.loc[row][col] = mindscope_utilities.dprime(
                hit_rate=response_matrix.loc[row][col],
                fa_rate=response_matrix[col][col],
                limits=False
            )
            if row == col:
                d_prime_matrix.loc[row][col] = np.nan

    return d_prime_matrix








