import numpy as np
import pandas as pd


def get_trace_average(trace, timestamps, start_time, stop_time):
    """
    takes average value of a trace within a window
    designated by start_time and stop_time
    """
    values_this_range = trace[(
        (timestamps >= start_time) & (timestamps < stop_time))]
    return values_this_range.mean()


def average_df_timeseries_values(dataframe, values_column):
    """calculates the mean timeseries from a dataframe
        column.

    Parameters
    ----------
    dataframe : pandas dataframe
        generic dataframe
    values_column : string
        name of the column that contains the
        timeseries arrays to average

    Returns
    -------
    array
        the averaged (mean) timeseries
    """
    values_array = np.vstack(dataframe[values_column].values)
    mean_trace = np.mean(values_array, axis=0)
    return mean_trace


# Precondition utilities
def validate_value_in_dict_keys(input_key, dictionary, dict_name):
    """validates that the input key is in fact a
        key in a given dictionary

    Parameters
    ----------
    input_key : generic
        key to check
    dictionary : dictionary
        dictionary to check
    dict_name : string
        name of dictionary
    """
    assert input_key in dictionary, "Error: input value is\
        not in {} keys.".format(dict_name)


def get_time_array(t_start, t_end, sampling_rate=None, step_size=None, include_endpoint=True):  # NOQA E501
    '''
    A function to get a time array between two specified timepoints at a defined sampling rate  # NOQA E501
    Deals with possibility of time range not being evenly divisible by desired sampling rate  # NOQA E501
    Uses np.linspace instead of np.arange given decimal precision issues with np.arange (see np.arange documentation for details)  # NOQA E501

    Parameters:
    -----------
    t_start : float
        start time for array
    t_end : float
        end time for array
    sampling_rate : float
        desired sampling of array
        Note: user must specify either sampling_rate or step_size, not both
    step_size : float
        desired step size of array
        Note: user must specify either sampling_rate or step_size, not both
    include_endpoint : Boolean
        Passed to np.linspace to calculate relative time
        If True, stop is the last sample. Otherwise, it is not included.
            Default is True

    Returns:
    --------
    numpy.array
        an array of timepoints at the desired sampling rate

    Examples:
    ---------
    get a time array exclusive of the endpoint
    >>> t_array = get_time_array(
        t_start=-1,
        t_end=1,
        step_size=0.5,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5])


    get a time array inclusive of the endpoint
    >>> t_array = get_time_array(
        t_start=-1,
        t_end=1,
        step_size=0.5,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5, 1.0])


    get a time array where the range can't be evenly divided by the desired step_size
    in this case, the time array includes the last timepoint before the desired endpoint
    >>> t_array = get_time_array(
        t_start=-1,
        t_end=0.75,
        step_size=0.5,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5])


    Instead of passing the step_size, we can pass the sampling rate
    >>> t_array = get_time_array(
        t_start=-1,
        t_end=1,
        sampling_rate=2,
        include_endpoint=False
    )

    np.array([-1., -0.5,  0.,  0.5])
    '''
    assert sampling_rate is not None or step_size is not None, 'must specify either sampling_rate or step_size'  # NOQA E501
    assert sampling_rate is None or step_size is None, 'cannot specify both sampling_rate and step_size'  # NOQA E501

    # value as a linearly spaced time array
    if not step_size:
        step_size = 1 / sampling_rate
    # define a time array
    n_steps = (t_end - t_start) / step_size
    if n_steps != int(n_steps):
        # if the number of steps isn't an int, that means it isn't possible
        # to end on the desired t_after using the defined sampling rate
        # we need to round down and include the endpoint
        n_steps = int(n_steps)
        t_end_adjusted = t_start + n_steps * step_size
        include_endpoint = True
    else:
        t_end_adjusted = t_end

    if include_endpoint:
        # add an extra step if including endpoint
        n_steps += 1

    t_array = np.linspace(
        t_start,
        t_end_adjusted,
        int(n_steps),
        endpoint=include_endpoint
    )

    return t_array


def slice_inds_and_offsets(data_timestamps, event_timestamps, time_window, sampling_rate=None, include_endpoint=False):  # NOQA E501
    '''
    Get nearest indices to event timestamps, plus ind offsets (start:stop)
    for slicing out a window around the event from the trace.
    Parameters:
    -----------
    data_timestamps : np.array
        Timestamps of the datatrace.
    event_timestamps : np.array
        Timestamps of events around which to slice windows.
    time_window : list
        [start_offset, end_offset] in seconds
    sampling_rate : float, optional, default=None
        Sampling rate of the datatrace.
        If left as None, samplng rate is inferred from data_timestamps.

    Returns:
    --------
    event_indices : np.array
        Indices of events from the timestamps provided.
    start_ind_offset : int
    end_ind_offset : int
    trace_timebase :  np.array
    '''
    if sampling_rate is None:
        sampling_rate = 1 / np.diff(data_timestamps).mean()

    event_indices = index_of_nearest_value(data_timestamps, event_timestamps)
    trace_len = (time_window[1] - time_window[0]) * sampling_rate
    start_ind_offset = int(time_window[0] * sampling_rate)
    end_ind_offset = int(start_ind_offset + trace_len) + int(include_endpoint)
    trace_timebase = np.arange(
        start_ind_offset, end_ind_offset) / sampling_rate

    return event_indices, start_ind_offset, end_ind_offset, trace_timebase


def index_of_nearest_value(data_timestamps, event_timestamps):
    '''
    The index of the nearest sample time for each event time.

    Parameters:
    -----------
    sample_timestamps : np.ndarray of floats
        sorted 1-d vector of data sample timestamps.
    event_timestamps : np.ndarray of floats
        1-d vector of event timestamps.

    Returns:
    --------
    event_aligned_ind : np.ndarray of int
        An array of nearest sample time index for each event times.
    '''
    insertion_ind = np.searchsorted(data_timestamps, event_timestamps)
    # is the value closer to data at insertion_ind or insertion_ind-1?
    ind_diff = data_timestamps[insertion_ind] - event_timestamps
    ind_minus_one_diff = np.abs(data_timestamps[np.clip(
        insertion_ind - 1, 0, np.inf).astype(int)] - event_timestamps)

    event_indices = insertion_ind - (ind_diff > ind_minus_one_diff).astype(int)
    return event_indices


def eventlocked_traces(traces_array, event_indices, start_ind_offset, end_ind_offset):
    '''
    Extract trace for each cell, for each event-relative window.
    Args:
        traces_array (np.ndarray): shape (nSamples, nCells) with timeseries for each cell
        event_indices (np.ndarray): 1-d array of shape (nEvents) with closest sample ind for each event
        start_ind_offset (int): Where to start the window relative to each event ind
        end_ind_offset (int): Where to end the window relative to each event ind
    Returns:
        sliced_dataout (np.ndarray): shape (nSamples, nEvents, nCells)
    '''
    all_inds = event_indices + np.arange(start_ind_offset, end_ind_offset)[
        :, None]  # takes a slice around all event_indices
    sliced_dataout = traces_array.T[all_inds]
    return sliced_dataout


def event_triggered_response(data, t, y, event_times, t_start=None, t_end=None, t_before=None, t_after=None,
                             output_sampling_rate=None, include_endpoint=True, output_format='tidy', interpolate=True):  # NOQA E501
    '''
    Slices a timeseries relative to a given set of event times
    to build an event-triggered response.

    For example, If we have data such as a measurement of neural activity
    over time and specific events in time that we want to align
    the neural activity to, this function will extract segments of the neural
    timeseries in a specified time window around each event.

    The times of the events need not align with the measured
    times of the neural data.
    Relative times will be calculated by linear interpolation.

    Parameters:
    -----------
    data: Pandas.DataFrame
        Input dataframe in tidy format
        Each row should be one observation
        Must contains columns representing `t` and `y` (see below)
    t : string
        Name of column in data to use as time data
    y : string
        Name of column to use as y data
    event_times: Panda.Series, numpy array or list of floats
        Times of events of interest. If pd.Series, the original index and index name will be preserved in the output
        Values in column specified by `y` will be sliced and interpolated
            relative to these times
    t_start : float
        start time relative to each event for desired time window
        e.g.:   t_start = -1 would start the window 1 second before each
                t_start = 1 would start the window 1 second after each event
        Note: cannot pass both t_start and t_before
    t_before : float
        time before each of event of interest to include in each slice
        e.g.:   t_before = 1 would start the window 1 second before each event
                t_before = -1 would start the window 1 second after each event
        Note: cannot pass both t_start and t_before
    t_end : float
        end time relative to each event for desired time window
        e.g.:   t_end = 1 would end the window 1 second after each event
                t_end = -1 would end the window 1 second before each event
        Note: cannot pass both t_end and t_after
    t_after : float
        time after each event of interest to include in each slice
        e.g.:   t_after = 1 would start the window 1 second after each event
                t_after = -1 would start the window 1 second before each event
        Note: cannot pass both t_end and t_after
    output_sampling_rate : float
        Desired sampling of output.
        Input data will be interpolated to this sampling rate if interpolate = True (default). # NOQA E501
        If passing interpolate = False, the sampling rate of the input timeseries will # NOQA E501
        be used and output_sampling_rate should not be specified.
    include_endpoint : Boolean
        Passed to np.linspace to calculate relative time
        If True, stop is the last sample. Otherwise, it is not included.
            Default is True
    output_format : string
        'wide' or 'tidy' (default = 'tidy')
        if 'tidy'
            One column representing time
            One column representing event_number
            One column representing event_time
            One row per observation (# rows = len(time) x len(event_times))
        if 'wide', output format will be:
            time as indices
            One row per interpolated timepoint
            One column per event,
                with column names titled event_{EVENT NUMBER}_t={EVENT TIME}
    interpolate : Boolean
        if True (default), interpolates each response onto a common timebase
        if False, shifts each response to align indices to a common timebase

    Returns:
    --------
    Pandas.DataFrame
        See description in `output_format` section above

    Examples:
    ---------
    An example use case, recover a sinousoid from noise:

    First, define a time vector
    >>> t = np.arange(-10,110,0.001)

    Now build a dataframe with one column for time,
    and another column that is a noise-corrupted sinuosoid with period of 1
    >>> data = pd.DataFrame({
            'time': t,
            'noisy_sinusoid': np.sin(2*np.pi*t) + np.random.randn(len(t))*3
        })

    Now use the event_triggered_response function to get a tidy
    dataframe of the signal around every event

    Events will simply be generated as every 1 second interval
    starting at 0, since our period here is 1
    >>> etr = event_triggered_response(
            data,
            x = 'time',
            y = 'noisy_sinusoid',
            event_times = np.arange(100),
            t_start = -1,
            t_end = 1,
            output_sampling_rate = 100
        )
    Then use seaborn to view the result
    We're able to recover the sinusoid through averaging
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> fig, ax = plt.subplots()
    >>> sns.lineplot(
            data = etr,
            x='time',
            y='noisy_sinusoid',
            ax=ax
        )
    '''
    # ensure that non-conflicting time values are passed
    assert t_before is not None or t_start is not None, 'must pass either t_start or t_before'  # noqa: E501
    assert t_after is not None or t_end is not None, 'must pass either t_start or t_before'  # noqa: E501

    assert t_before is None or t_start is None, 'cannot pass both t_start and t_before'  # noqa: E501
    assert t_after is None or t_end is None, 'cannot pass both t_after and t_end'  # noqa: E501

    if interpolate is False:
        output_sampling_rate = None
        # MG - commenting this out because it crashes my code when interpolate = False, even if output_sampling_rate = None
        # assert output_sampling_rate is None, 'if interpolation = False, the sampling rate of the input timeseries will be used. Do not specify output_sampling_rate'  # NOQA E501

    # assign time values to t_start and t_end
    if t_start is None:
        t_start = -1 * t_before
    if t_end is None:
        t_end = t_after

    # get original stimulus_presentation_ids, preserve the column for .join() method
    if type(event_times) is pd.Series:  # if pd.Series, preserve the name of index column
        original_index = event_times.index.values
        if type(event_times.index.name) is str:
            original_index_column_name = event_times.index.name
        else:  # if index column does not have a name, name is original_index
            event_times.index.name = 'original_index'
            original_index_column_name = event_times.index.name
    # is list or array, turn into pd.Series
    elif type(event_times) is list or type(event_times) is np.ndarray:
        event_times = pd.Series(data=event_times,
                                name='event_times'
                                )
        # name the index column "original_index"
        event_times.index.name = 'original_index'
        original_index_column_name = event_times.index.name
        original_index = event_times.index.values

    # ensure that t_end is greater than t_start
    assert t_end > t_start, 'must define t_end to be greater than t_start'

    if output_sampling_rate is None:
        # if sampling rate is None,
        # set it to be the mean sampling rate of the input data
        output_sampling_rate = 1 / np.diff(data[t]).mean()

    # if interpolate is set to True,
    # we will calculate a common timebase and
    # interpolate every response onto that timebase
    if interpolate:
        # set up a dictionary with key 'time' and
        t_array = get_time_array(
            t_start=t_start,
            t_end=t_end,
            sampling_rate=output_sampling_rate,
            include_endpoint=include_endpoint,
        )
        data_dict = {'time': t_array}

        # iterate over all event times
        data_reindexed = data.set_index(t, inplace=False)

        for event_number, event_time in enumerate(np.array(event_times)):

            # get a slice of the input data surrounding each event time
            data_slice = data_reindexed[y].loc[event_time + t_start: event_time + t_end]  # noqa: E501

            # update our dictionary to have a new key defined as
            # 'event_{EVENT NUMBER}_t={EVENT TIME}' and
            # a value that includes an array that represents the
            # sliced data around the current event, interpolated
            # on the linearly spaced time array
            data_dict.update({
                'event_{}_t={}'.format(event_number, event_time): np.interp(
                    data_dict['time'],
                    data_slice.index - event_time,
                    data_slice.values
                )
            })

        # define a wide dataframe as a dataframe of the above compiled dictionary  # NOQA E501
        wide_etr = pd.DataFrame(data_dict)

    # if interpolate is False,
    # we will calculate a common timebase and
    # shift every response onto that timebase
    else:
        event_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(  # NOQA E501
            np.array(data[t]),
            np.array(event_times),
            time_window=[t_start, t_end],
            sampling_rate=None,
            include_endpoint=True
        )
        all_inds = event_indices + \
            np.arange(start_ind_offset, end_ind_offset)[:, None]
        wide_etr = pd.DataFrame(
            data[y].values.T[all_inds],
            index=trace_timebase,
            columns=['event_{}_t={}'.format(event_index, event_time) for event_index, event_time in enumerate(event_times)]  # NOQA E501
        ).rename_axis(index='time').reset_index()

    if output_format == 'wide':
        # return the wide dataframe if output_format is 'wide'
        return wide_etr.set_index('time')
    elif output_format == 'tidy':
        # if output format is 'tidy',
        # transform the wide dataframe to tidy format
        # first, melt the dataframe with the 'id_vars' column as "time"
        tidy_etr = wide_etr.melt(id_vars='time')

        # add an "event_number" column that contains the event number
        tidy_etr['event_number'] = tidy_etr['variable'].map(
            lambda s: s.split('event_')[1].split('_')[0]
        ).astype(int)

        tidy_etr[original_index_column_name] = tidy_etr['event_number'].apply(
            lambda row: original_index[row])

        # add an "event_time" column that contains the event time ()
        tidy_etr['event_time'] = tidy_etr['variable'].map(
            lambda s: s.split('t=')[1]
        ).astype(float)

        # drop the "variable" column, rename the "value" column
        tidy_etr = (
            tidy_etr
            .drop(columns=['variable'])
            .rename(columns={'value': y})
        )
        # return the tidy event triggered responses
        return tidy_etr


def time_from_last(timestamps, event_times, side='right'):
    '''
    For each timestamp, returns the time from the most recent other time (in event_times)

    Args:
        timestamps (np.array): array of timestamps for which the 'time from last event' will be returned
        event_times (np.array): event timestamps
    Returns
        time_from_last_event (np.array): the time from the last event for each timestamp

    '''
    last_event_index = np.searchsorted(
        a=event_times, v=timestamps, side=side) - 1
    time_from_last_event = timestamps - event_times[last_event_index]
    # flashes that happened before the other thing happened should return nan
    time_from_last_event[last_event_index == -1] = np.nan

    return time_from_last_event
