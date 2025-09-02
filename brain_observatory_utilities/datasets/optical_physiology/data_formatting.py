import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray

from brain_observatory_utilities.utilities import general_utilities
from brain_observatory_utilities.datasets.behavior import data_formatting as behavior


def build_tidy_cell_df(ophys_experiment, exclude_invalid_rois=True):
    '''
    Builds a tidy dataframe describing activity for every cell in ophys_experiment.
    Tidy format is defined as one row per observation.
    Thus, the output dataframe will be n_cells x n_timetpoints long

    Parameters:
    -----------
    ophys_experiment : AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501
    exclude_invalid_rois : bool
        If True (default), only includes ROIs that are listed as `valid_roi = True` in the ophys_experiment.cell_specimen_table.
        If False, include all ROIs.
        Note that invalid ROIs are only exposed for internal AllenInstitute users, so passing `False` will not change behavior for external users

    Returns:
    --------
    Pandas.DataFrame
        Tidy Format (one observation per row) with the following columns:
            * timestamps (float) : the ophys timestamps
            * cell_roi_id (int) : the cell roi id
            * cell_specimen_id (int) : the cell specimen id
            * dff (float) : measured deltaF/F for every timestep
            * events (float) : extracted events for every timestep
            * filtered events (float) : filtered (convolved with half-gaussian) events for every timestep
    '''

    # make an empty list to populate with dataframes for each cell
    list_of_cell_dfs = []

    # query on valid_roi if exclude_invalid_rois == True
    if exclude_invalid_rois:
        cell_specimen_table = ophys_experiment.cell_specimen_table.query('valid_roi').reset_index()  # noqa E501
    else:
        cell_specimen_table = ophys_experiment.cell_specimen_table.reset_index()  # noqa E501

    # iterate over each individual cell
    for idx, row in cell_specimen_table.iterrows():
        cell_specimen_id = row['cell_specimen_id']

        # build a tidy dataframe for this cell
        cell_df = pd.DataFrame({
            'timestamps': ophys_experiment.ophys_timestamps,
            'dff': ophys_experiment.dff_traces.loc[cell_specimen_id]['dff'] if cell_specimen_id in ophys_experiment.dff_traces.index else [np.nan] * len(ophys_experiment.ophys_timestamps),  # noqa E501
            'events': ophys_experiment.events.loc[cell_specimen_id]['events'] if cell_specimen_id in ophys_experiment.events.index else [np.nan] * len(ophys_experiment.ophys_timestamps),  # noqa E501
            'filtered_events': ophys_experiment.events.loc[cell_specimen_id]['filtered_events'] if cell_specimen_id in ophys_experiment.events.index else [np.nan] * len(ophys_experiment.ophys_timestamps),  # noqa E501
        })

        # Make the cell_roi_id and cell_specimen_id columns categorical.
        # This will reduce memory useage since the columns
        # consist of many repeated values.
        for cell_id in ['cell_roi_id', 'cell_specimen_id']:
            cell_df[cell_id] = np.int32(row[cell_id])
            cell_df[cell_id] = pd.Categorical(
                cell_df[cell_id],
                categories=cell_specimen_table[cell_id].unique()
            )

        # append the dataframe for this cell to the list of cell dataframes
        list_of_cell_dfs.append(cell_df)

    # concatenate all dataframes in the list
    tidy_df = pd.concat(list_of_cell_dfs)

    # return the tidy dataframe
    return tidy_df


def get_event_timestamps(
        stimulus_presentation,
        event_type='all',
        onset='start_time'):
    '''
    Gets timestamps of events of interest from the stimulus_presentations df.

    Parameters:
    ___________
    stimulus_presentation: Pandas.DataFrame
        Output of stimulus_presentations with stimulus trial metadata
    event_type: str
        Event of interest. Event_type can be any column in the stimulus_presentation,  # noqa E501
        including 'omissions' or 'change'. Default is 'all', gets all trials  # noqa E501
    onset: str
        optons: 'start_time' - onset of the stimulus, 'stop_time' - offset of the stimulus
        stimulus_presentationshas a multiple timestamps to align data to. Default = 'start_time'.

    Returns:
        event_times: array
        event_ids: array
    --------
    '''
    if event_type == 'all':
        event_times = stimulus_presentation[onset]
        event_ids = stimulus_presentation.index.values
    elif event_type == 'images':
        event_times = stimulus_presentation[stimulus_presentation['omitted'] == False][onset]  # noqa E501
        event_ids = stimulus_presentation[stimulus_presentation['omitted'] == False].index.values  # noqa E501
    elif event_type == 'omissions' or event_type == 'omitted':
        event_times = stimulus_presentation[stimulus_presentation['omitted']][onset]  # noqa E501
        event_ids = stimulus_presentation[stimulus_presentation['omitted']].index.values  # noqa E501
    elif event_type == 'changes' or event_type == 'is_change':
        event_times = stimulus_presentation[stimulus_presentation['is_change']][onset]  # noqa E501
        event_ids = stimulus_presentation[stimulus_presentation['is_change']].index.values  # noqa E501
    else:
        event_times = stimulus_presentation[stimulus_presentation[event_type]][onset]  # noqa E501
        event_ids = stimulus_presentation[stimulus_presentation[event_type]].index.values  # noqa E501

    return event_times, event_ids



def get_stimulus_response_xr(ophys_experiment,
                             data_type='dff',
                             event_type='all',
                             time_window=[-3, 3],
                             response_window_duration=0.5,
                             interpolate=True,
                             output_sampling_rate=None,
                             exclude_invalid_rois=True,
                             **kwargs):
    '''
    Parameters:
    ___________
    ophys_experiment: obj
        AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501
    data_type: str
        neural or behavioral data type to extract, options are: dff (default), events, filtered_events, running_speed, pupil_diameter, lick_rate
    event_type: str
        event type to align to, which can be found in columns of ophys_experiment.stimulus_presentations df.
        options are: 'all' (default) - gets all stimulus trials
                     'images' - gets only image presentations (changes and not changes)
                     'omissions' - gets only trials with omitted stimuli
                     'changes' - get only trials of image changes
    time_window: array
        array of two int or floats indicating the time window on sliced data, default = [-3, 3]
    response_window_duration: float
        time period, in seconds, relative to stimulus onset to compute the mean and baseline response
    interpolate: bool
        type of alignment. If True (default) - interpolates neural data to align timestamps
        with stimulus presentations. If False - shifts data to the nearest timestamps
    output_sampling_rate : float
        Desired sampling of output.
        Input data will be interpolated to this sampling rate if interpolate = True (default). # NOQA E501
        If passing interpolate = False, the sampling rate of the input timeseries will # NOQA E501
        be used and output_sampling_rate should not be specified.
    exclude_invalid_rois : bool
        If True, only ROIs deemed as 'valid' by the classifier will be returned. If False, 'invalid' ROIs will be returned
        This only works if the provided dataset was loaded using internal Allen Institute database, does not work for NWB files.
        In the case that dataset object is loaded through publicly released NWB files, only 'valid' ROIs will be returned


    kwargs: key, value mappings
        Other keyword arguments are passed down to general_utilities.event_triggered_response(),
        for interpolation method such as include_endpoint.

    Returns:
    __________
    stimulus_response_xr: xarray
        Xarray of aligned neural data with multiple dimentions: cell_specimen_id,
        'eventlocked_timestamps', and 'stimulus_presentations_id'

    '''

    import brain_observatory_utilities.datasets.behavior.data_access as behavior_data_access

    # load stimulus_presentations table
    stimulus_presentations = ophys_experiment.stimulus_presentations.copy()

    # limit to change detection block
    stimulus_presentations = behavior.limit_stimulus_presentations_to_change_detection(stimulus_presentations)

    # get event times and event ids (original order in the stimulus flow)
    event_times, event_ids = get_event_timestamps(
        stimulus_presentations, event_type)

    if ('running' in data_type) or ('pupil' in data_type) or ('lick' in data_type):
        # for behavioral datastreams
        # set up variables to handle only one timeseries per event instead of multiple cell_specimen_ids
        unique_id_string = 'trace_id'  # create a column to take the place of 'cell_specimen_id'
        unique_ids = [0]  # list to iterate over
    else:
        unique_id_string = 'cell_specimen_id'

    if 'running' in data_type:
        data = ophys_experiment.running_speed.copy() # running_speed attribute is already in tidy format
        data = data.rename(columns={'speed':'running_speed'}) # rename column so its consistent with data_type
        data[unique_id_string] = 0  # only one value because only one trace
    elif 'pupil' in data_type:
        data = ophys_experiment.eye_tracking.copy() # eye tracking attribute is in tidy format
        data = behavior_data_access.get_pupil_data(data, interpolate_likely_blinks=True, normalize_to_gray_screen=True, zscore=False,
                                interpolate_to_ophys=False, stimulus_presentations=stimulus_presentations, ophys_timestamps=None)
        # normalize to gray screen baseline
        data[unique_id_string] = 0  # only one value because only one trace
    elif 'lick' in data_type:
        data = behavior.get_licks_df(ophys_experiment) # create dataframe with info about licks for each stimulus timestamp
        data[unique_id_string] = 0  # only one value because only one trace
    else:
        # load neural data
        data = build_tidy_cell_df(ophys_experiment, exclude_invalid_rois=exclude_invalid_rois)
        # all cell specimen ids in an ophys_experiment
        unique_ids = np.unique(data['cell_specimen_id'].values)

    # get native sampling rate if one is not provided
    if output_sampling_rate is None:
        output_sampling_rate = 1 / np.diff(data['timestamps']).mean()

    # collect aligned data
    sliced_dataout = []

    # align data using interpolation method
    for unique_id in tqdm(unique_ids):
        etr = general_utilities.event_triggered_response(
            data=data[data[unique_id_string] == unique_id],
            t='timestamps',
            y=data_type,
            event_times=event_times,
            t_start=time_window[0],
            t_end=time_window[1],
            output_format='wide',
            interpolate=interpolate,
            output_sampling_rate=output_sampling_rate,
            **kwargs
        )

        # get timestamps array
        trace_timebase = etr.index.values

        # collect aligned data from all cell, all trials into one array
        sliced_dataout.append(etr.transpose().values)

    # convert to xarray
    sliced_dataout = np.array(sliced_dataout)
    stimulus_response_xr = xarray.DataArray(
        data=sliced_dataout,
        dims=(unique_id_string, 'stimulus_presentations_id',
              'eventlocked_timestamps'),
        coords={
            'eventlocked_timestamps': trace_timebase,
            'stimulus_presentations_id': event_ids,
            unique_id_string: unique_ids
        }
    )


    # get traces for significance computation
    if 'events' in data_type:
        traces_array = np.vstack(ophys_experiment.events[data_type].values)
    elif data_type == 'dff':
        traces_array = np.vstack(ophys_experiment.dff_traces['dff'].values)
    else:
        traces_array = data[data_type].values

    # compute mean activity following stimulus onset and during pre-stimulus baseline
    stimulus_response_xr = compute_means_xr(stimulus_response_xr, response_window_duration=response_window_duration)

    # get mean response for each trial
    mean_responses = stimulus_response_xr.mean_response.data.T  # input needs to be array of nConditions, nCells

    try:
        # compute significance of each trial, returns array of nConditions, nCells
        p_value_gray_screen = get_p_value_from_shuffled_spontaneous(mean_responses,
                                                                    stimulus_presentations,
                                                                    ophys_experiment.ophys_timestamps,
                                                                    traces_array,
                                                                    response_window_duration*output_sampling_rate,
                                                                    output_sampling_rate)
    except:
        p_value_gray_screen = np.zeros(mean_responses.shape)

    # put p_value_gray_screen back into same coordinates as xarray and make it an xarray data array
    p_value_gray_screen = xarray.DataArray(data=p_value_gray_screen.T, coords=stimulus_response_xr.mean_response.coords)

    # create new xarray with means and p-values
    stimulus_response_xr = xarray.Dataset({
        'eventlocked_traces': stimulus_response_xr.eventlocked_traces,
        'mean_response': stimulus_response_xr.mean_response,
        'mean_baseline': stimulus_response_xr.mean_baseline,
        'p_value_gray_screen': p_value_gray_screen
    })

    return stimulus_response_xr


def compute_means_xr(stimulus_response_xr, response_window_duration=0.5):
    '''
    Computes means of traces for stimulus response and pre-stimulus baseline.
    Response by default starts at 0, while baseline
    trace by default ends at 0.

    Parameters:
    ___________
    stimulus_response_xr: xarray
        stimulus_response_xr from get_stimulus_response_xr
        with three main dimentions: cell_specimen_id,
        trail_id, and eventlocked_timestamps
    response_window_duration:
        duration in seconds relative to stimulus onset to compute the mean and baseline responses
        in get_stimulus_response_xr

    Returns:
    _________
        stimulus_response_xr with additional
        dimentions: mean_response and mean_baseline
    '''
    response_range = [0, response_window_duration]
    baseline_range = [-response_window_duration, 0]

    mean_response = stimulus_response_xr.loc[
        {'eventlocked_timestamps': slice(*response_range)}
    ].mean(['eventlocked_timestamps'])

    mean_baseline = stimulus_response_xr.loc[
        {'eventlocked_timestamps': slice(*baseline_range)}
    ].mean(['eventlocked_timestamps'])

    stimulus_response_xr = xarray.Dataset({
        'eventlocked_traces': stimulus_response_xr,
        'mean_response': mean_response,
        'mean_baseline': mean_baseline,
    })

    return stimulus_response_xr


def get_spontaneous_frames(stimulus_presentations, ophys_timestamps, gray_screen_period_to_use='before'):
    '''
        Returns a list of the frames that occur during the before and after spontaneous windows. This is copied from VBA. Does not use the full spontaneous period because that is what VBA did. It only uses 4 minutes of the before and after spontaneous period.

    Args:
        stimulus_presentations_df (pandas.DataFrame): table of stimulus presentations, including start_time and stop_time
        ophys_timestamps (np.array): timestamps of each ophys frame
        gray_screen_period_to_use (str): 'before', 'after', or 'both'
                                        whether to use the gray screen period before the session, after the session, or across both
    Returns:
        spontaneous_inds (np.array): indices of ophys frames during the gray screen period before or after the session, or both
    '''
    # exclude the very first minute of the session because the monitor has just turned on and can cause artifacts
    # spont_duration_frames = 4 * 60 * 60  # 4 mins * * 60s/min * 60Hz
    spont_duration = 4 * 60  # 4mins * 60sec

    # for spontaneous at beginning of session, get 4 minutes of gray screen values prior to first stimulus
    if stimulus_presentations.iloc[0].image_name == 'omitted': # something weird happens when first stimulus is omitted, start time is at beginning of session
        first_index = 1
    else:
        first_index = 0
    behavior_start_time = stimulus_presentations.iloc[first_index].start_time
    spontaneous_start_time_pre = behavior_start_time - spont_duration
    spontaneous_end_time_pre = behavior_start_time
    spontaneous_start_frame_pre = general_utilities.index_of_nearest_value(ophys_timestamps, spontaneous_start_time_pre)
    spontaneous_end_frame_pre = general_utilities.index_of_nearest_value(ophys_timestamps, spontaneous_end_time_pre)
    spontaneous_frames_pre = np.arange(spontaneous_start_frame_pre, spontaneous_end_frame_pre, 1)

    # for spontaneous epoch at end of session, get 4 minutes of gray screen values after the last stimulus
    behavior_end_time = stimulus_presentations.iloc[-1].start_time
    spontaneous_start_time_post = behavior_end_time + 0.75
    spontaneous_end_time_post = spontaneous_start_time_post + spont_duration
    spontaneous_start_frame_post = general_utilities.index_of_nearest_value(ophys_timestamps, spontaneous_start_time_post)
    spontaneous_end_frame_post = general_utilities.index_of_nearest_value(ophys_timestamps, spontaneous_end_time_post)
    spontaneous_frames_post = np.arange(spontaneous_start_frame_post, spontaneous_end_frame_post, 1)

    if gray_screen_period_to_use == 'before':
        spontaneous_frames = spontaneous_frames_pre
    elif gray_screen_period_to_use == 'after':
        spontaneous_frames = spontaneous_frames_post
    elif gray_screen_period_to_use == 'both':
        spontaneous_frames = np.concatenate([spontaneous_frames_pre, spontaneous_frames_post])
    return spontaneous_frames



def get_p_value_from_shuffled_spontaneous(mean_responses,
                                      stimulus_presentations,
                                      ophys_timestamps,
                                      traces_array,
                                      response_window_duration,
                                      ophys_frame_rate=None,
                                      number_of_shuffles=10000):
    '''
    Args:
        mean_responses (array): Mean response values, shape (nConditions, nCells)
        stimulus_presentations_df (pandas.DataFrame): Table of stimulus presentations, including start_time and stop_time
        ophys_timestamps (np.array): Timestamps of each ophys frame
        traces_arr (np.array): trace values, shape (nSamples, nCells)
        response_window_duration (int): Number of frames averaged to produce mean response values
        number_of_shuffles (int): Number of shuffles of spontaneous activity used to produce the p-value
    Returns:
        p_values (array): p-value for each response mean, shape (nConditions, nCells)
    '''

    from brain_observatory_utilities.utilities.general_utilities import eventlocked_traces

    spontaneous_frames = get_spontaneous_frames(stimulus_presentations, ophys_timestamps, gray_screen_period_to_use='before')
    shuffled_spont_inds = np.random.choice(spontaneous_frames, number_of_shuffles)

    if ophys_frame_rate is None:
        ophys_frame_rate = 1 / np.diff(ophys_timestamps).mean()

    trace_len = np.round(response_window_duration * ophys_frame_rate).astype(int)
    start_ind_offset = 0
    end_ind_offset = trace_len
    # get an x frame segment of each cells trace after each shuffled spontaneous timepoint
    spont_traces = eventlocked_traces(traces_array, shuffled_spont_inds, start_ind_offset, end_ind_offset)
    # average over the response window (x frames) for each shuffle,
    spont_mean = spont_traces.mean(axis=0)  #Returns (nShuffles, nCells) - mean repsonse following each shuffled spont frame

    # Goal is to figure out how each response compares to the shuffled distribution, which is just
    # a searchsorted call if we first sort the shuffled.
    spont_mean_sorted = np.sort(spont_mean, axis=0) # for each cell, sort the spontaneous mean values (axis0 is shuffles, axis1 is cells)
    response_insertion_ind = np.empty(mean_responses.shape) # should be nConditions, nCells
    # in cases where there is only 1 unique ID (i.e. one neuron in FOV, or one running or pupil trace), duplicate dims so the code below works
    if spont_mean_sorted.ndim == 1:
        spont_mean_sorted = np.expand_dims(spont_mean_sorted, axis=1)
    # loop through indices and figure out how many times the mean response is greater than the spontaneous shuffles
    for ind_cell in range(mean_responses.shape[1]):
        response_insertion_ind[:, ind_cell] = np.searchsorted(spont_mean_sorted[:, ind_cell],
                                                              mean_responses[:, ind_cell])
    # p value is 1 over the fraction times that a given mean response is larger than the 10,000 shuffle means
    # response_insertion_ind tells the index that the mean response would have to be placed in to maintain the order of the shuffled spontaneous
    # if that number is 10k, the mean response is larger than all the shuffles
    # dividing response_insertion_index by 10k gives you the fraction of times that mean response was greater than the shuffles
    # then divide by 1 to get p-value
    proportion_spont_larger_than_sample = 1 - (response_insertion_ind / number_of_shuffles)
    p_values = proportion_spont_larger_than_sample
    # result = xarray.DataArray(data=proportion_spont_larger_than_sample,
    #                       coords=mean_responses.coords)
    return p_values


def get_stimulus_response_df(ophys_experiment,
                             data_type='dff',
                             event_type='all',
                             time_window=[-3, 3],
                             response_window_duration=0.5,
                             interpolate=True,
                             output_sampling_rate=None,
                             exclude_invalid_rois=True,
                             **kwargs):
    '''
    Get stimulus aligned responses from one ophys_experiment.

    Parameters:
    ___________
    ophys_experiment: obj
        AllenSDK BehaviorOphysExperiment object
        A BehaviorOphysExperiment instance
        See https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/behavior/behavior_ophys_ophys_experiment.py  # noqa E501
    data_type: str
        neural or behavioral data type to extract, options are: dff (default), events, filtered_events, running_speed, pupil_diameter, lick_rate
    event_type: str
        event type to align to, which can be found in columns of ophys_experiment.stimulus_presentations df.
        options are: 'all' (default) - gets all stimulus trials
                     'images' - gets only image presentations (changes and not changes)
                     'omissions' - gets only trials with omitted stimuli
                     'changes' - get only trials of image changes
    time_window: array
        array of two int or floats indicating the time window on sliced data, default = [-3, 3]
    response_window_duration: float
        time period, in seconds, relative to stimulus onset to compute the mean and baseline response
    interpolate: bool
        type of alignment. If True (default) - interpolates neural data to align timestamps
        with stimulus presentations. If False - shifts data to the nearest timestamps
    output_sampling_rate : float
        Desired sampling of output.
        Input data will be interpolated to this sampling rate if interpolate = True (default). # NOQA E501
        If passing interpolate = False, the sampling rate of the input timeseries will # NOQA E501
        be used and output_sampling_rate should not be specified.
     exclude_invalid_rois : bool
        If True, only ROIs deemed as 'valid' by the classifier will be returned. If False, 'invalid' ROIs will be returned
        This only works if the provided dataset was loaded using internal Allen Institute database, does not work for NWB files.
        In the case that dataset object is loaded through publicly released NWB files, only 'valid' ROIs will be returned

    kwargs: key, value mappings
        Other keyword arguments are passed down to general_utilities.event_triggered_response(),
        for interpolation method such as output_sampling_rate and include_endpoint.

    Returns:
    ___________
    stimulus_response_df: Pandas.DataFrame


    '''

    stimulus_response_xr = get_stimulus_response_xr(
        ophys_experiment=ophys_experiment,
        data_type=data_type,
        event_type=event_type,
        time_window=time_window,
        response_window_duration=response_window_duration,
        interpolate=interpolate,
        output_sampling_rate=output_sampling_rate,
        exclude_invalid_rois=exclude_invalid_rois,
        **kwargs)

    # set up identifier columns depending on whether behavioral or neural data is being used
    # if (data_type == 'running_speed') or (data_type == 'pupil_diameter') or (data_type == 'lick_rate'):
    if ('lick' in data_type) or ('pupil' in data_type) or ('running' in data_type):
        # set up variables to handle only one timeseries per event instead of multiple cell_specimen_ids
        unique_id_string = 'trace_id'  # create a column to take the place of 'cell_specimen_id'
    else:
        # all cell specimen ids in an ophys_experiment
        unique_id_string = 'cell_specimen_id'

    # get mean response after stimulus onset and during pre-stimulus baseline
    mean_response = stimulus_response_xr['mean_response']
    mean_baseline = stimulus_response_xr['mean_baseline']
    stacked_response = mean_response.stack(
        multi_index=('stimulus_presentations_id', unique_id_string)).transpose()  # noqa E501
    stacked_baseline = mean_baseline.stack(
        multi_index=('stimulus_presentations_id', unique_id_string)).transpose()  # noqa E501

    # get p_value for each stimulus response compared to a shuffled distribution of gray screen values
    p_vals_gray_screen = stimulus_response_xr['p_value_gray_screen']
    stacked_pval_gray_screen = p_vals_gray_screen.stack(
        multi_index=('stimulus_presentations_id', unique_id_string)).transpose()  # noqa E501

    # get event locked traces and timestamps from xarray
    traces = stimulus_response_xr['eventlocked_traces']
    stacked_traces = traces.stack(multi_index=(
        'stimulus_presentations_id', unique_id_string)).transpose()
    num_repeats = len(stacked_traces)
    trace_timestamps = np.repeat(
        stacked_traces.coords['eventlocked_timestamps'].data[np.newaxis, :],
        repeats=num_repeats, axis=0)

    # turn it all into a dataframe
    stimulus_response_df = pd.DataFrame({
        'stimulus_presentations_id': stacked_traces.coords['stimulus_presentations_id'],  # noqa E501
        unique_id_string: stacked_traces.coords[unique_id_string],
        'trace': list(stacked_traces.data),
        'trace_timestamps': list(trace_timestamps),
        'mean_response': stacked_response.data,
        'baseline_response': stacked_baseline.data,
        'p_value_gray_screen': stacked_pval_gray_screen,
    })

    # save frame rate, time window and other metadata for reference
    if output_sampling_rate is not None:
        stimulus_response_df['ophys_frame_rate'] = output_sampling_rate
    else:
        stimulus_response_df['ophys_frame_rate'] = ophys_experiment.metadata['ophys_frame_rate']
    stimulus_response_df['data_type'] = data_type
    stimulus_response_df['event_type'] = event_type
    stimulus_response_df['interpolate'] = interpolate
    stimulus_response_df['output_sampling_rate'] = output_sampling_rate
    stimulus_response_df['response_window_duration'] = response_window_duration

    return stimulus_response_df



