import numpy as np
import pandas as pd


def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    '''
    Convenience function to compute a peri-stimulus-time histogram
    (see section 7.2.2 here: https://neuronaldynamics.epfl.ch/online/Ch7.S2.html)
    INPUTS:
        spikes: spike times in seconds for one unit
        startTimes: trial start times in seconds; the first spike count 
            bin will be aligned to these times
        windowDur: trial duration in seconds
        binSize: size of spike count bins in seconds
    OUTPUTS:
        Tuple of (PSTH, bins), where:
            PSTH gives the trial-averaged spike rate for 
                each time bin aligned to the start times;
            bins are the bin edges as defined by numpy histogram
    '''
    bins = np.arange(0, windowDur+binSize, binSize)
    counts = np.zeros(bins.size-1)
    for start in startTimes:
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/len(startTimes)
    return counts/binSize, bins[:-1]


def make_psth(spike_times, stim_times, pre_window=0.5, post_window=1.0, bin_size=0.05):
    """
    Generate a Peri-Stimulus Time Histogram (PSTH).
    
    Parameters:
    - spike_times: array-like, timestamps of all spikes (in seconds)
    - stim_times: array-like, timestamps of stimulus onsets (in seconds)
    - pre_window: float, time before stimulus to include in PSTH (seconds)
    - post_window: float, time after stimulus to include in PSTH (seconds)
    - bin_size: float, width of each time bin (seconds)
    
    Returns:
    - firing_rates: 2D numpy array of firing rates (trials x bins)
    - bin_centers: 1D numpy array of bin center times (relative to stimulus onset)
    """

    # Ensure inputs are numpy arrays 
    spike_times = np.array(spike_times)
    stim_times = np.array(stim_times)
    
    # Define bin edges from -pre_window to +post_window
    bins = np.arange(-pre_window, post_window + bin_size, bin_size)
    
    # Compute centers of bins (for plotting)
    bin_centers = bins[:-1] + bin_size / 2
    
    # Initialize a matrix to hold spike counts: rows = trials, columns = bins
    all_counts = np.zeros((len(stim_times), len(bins) - 1))
    
    # Loop through each stimulus time to compute trial-specific spike counts
    for i, stim_time in enumerate(stim_times):
        # Select spikes that fall within the time window around this stimulus
        mask = ((spike_times >= stim_time - pre_window) & 
                (spike_times < stim_time + post_window))
        
        # Align spike times to stimulus onset (0 = stimulus)
        trial_spikes = spike_times[mask] - stim_time
        
        # Bin the aligned spikes and count how many fall into each bin
        counts, _ = np.histogram(trial_spikes, bins=bins)
        
        # Store the result in the i-th row (trial)
        all_counts[i, :] = counts
    
    # Convert spike counts to firing rates (spikes per second)
    firing_rates = all_counts / bin_size
    
    # Return firing rates (trials x bins) and bin center positions
    return firing_rates, bin_centers


def make_neuron_time_trials_array(units, spike_times, stim_table, 
                                   time_before, trial_duration,
                                   bin_size=0.001):
    '''
    Function to make a 3D array with dimensions [neurons, time bins, trials] to store
    the spike counts for stimulus presentation trials. 
    INPUTS:
        units: dataframe with unit info (same form as session.units table)
        spike_times: dictionary with spike times for each unit (ie session.spike_times)
        stim_table: dataframe whose indices are trial ids and containing a
            'start_time' column indicating when each trial began
        time_before: seconds to take before each start_time in the stim_table
        trial_duration: total time in seconds to take for each trial
        bin_size: bin_size in seconds used to bin spike counts 
    OUTPUTS:
        unit_array: 3D array storing spike counts. The value in [i,j,k] 
            is the spike count for neuron i at time bin j in the kth trial.
        time_vector: vector storing the trial timestamps for the time bins
    '''
    # Get dimensions of output array
    neuron_number = len(units)
    trial_number = len(stim_table)
    num_time_bins = int(trial_duration/bin_size)
    
    # Initialize array
    unit_array = np.zeros((neuron_number, num_time_bins, trial_number))
    
    # Loop through units and trials and store spike counts for every time bin
    for u_counter, (iu, unit) in enumerate(units.iterrows()):
        
        # grab spike times for this unit
        unit_spike_times = spike_times[iu]
        
        # now loop through trials and make a PSTH for this unit for every trial
        for t_counter, (it, trial) in enumerate(stim_table.iterrows()):
            trial_start = trial.start_time - time_before
            unit_array[u_counter, :, t_counter] = makePSTH(unit_spike_times, 
                                                            [trial_start], 
                                                            trial_duration, 
                                                            binSize=bin_size)[0]
    
    # Make the time vector that will label the time axis
    time_vector = np.arange(num_time_bins)*bin_size - time_before
    
    return unit_array, time_vector


def get_good_units(session):
    # get units table
    this_session_units = session.get_units()

    # Apply QC criteria
    good_units = this_session_units[
        (this_session_units.isi_violations<.5) &
        (this_session_units.amplitude_cutoff<.1) &
        (this_session_units.presence_ratio>.95)]
    print(len(good_units), 'units passing QC criteria')

    return good_units


def get_continous_spike_rate_for_units(session, stimulus_presentations, spike_rate_bin_size=0.01):
    """
    Create dataframe containing continuous spike rates across a full stimulus block for all units

    session: SDK VBN session object
    stimulus_presentations: stimulus_presentations table limited to the stimuli or stimulus blocks of interest
    spike_rate_bin_size: bin size, in seconds, to use when computing spike rate over time
                0.001 = 1ms, 0.01 = 10ms, 1 = 1s (spikes / second)
    stimulus_block: stimulus block number indicating portion of VBN session to compute spike rates for
                stimulus block 0 = change detection active behavior, 1 = 10s gray screen, 2 = gabor RF mapping,
                3 = 5min gray screen, 4 = full field flashes, 5 = change detection passive replay
    """

    # get data
    units = get_good_units(session)
    spike_times = session.spike_times.copy()
    stim_table = stimulus_presentations.copy()
    # create timeframe from start to end of behavior block
    start_time = stim_table.start_time.values[0]
    end_time = stim_table.end_time.values[-1] + 10 # Add 10 seconds to make sure the last image flashes can be included
    behavior_duration = end_time - start_time

    # Get dimensions of output array
    unit_ids = units.index.values
    neuron_number = len(unit_ids)
    num_time_bins = int(behavior_duration / spike_rate_bin_size) + 1

    # Initialize array
    unit_array = np.zeros((neuron_number, num_time_bins))

    # Loop through units and trials and store spike counts for every unit
    for i, unit_id in enumerate(unit_ids):
        # grab spike times for this unit
        unit_spike_times = spike_times[unit_id]
        spike_rate, timestamps = make_psth(unit_spike_times, [start_time], pre_window=0, 
                                           post_window=behavior_duration, bin_size=spike_rate_bin_size)
        # spike_rate, timestamps = makePSTH(unit_spike_times, [start_time], behavior_duration, binSize=spike_rate_bin_size)
        # convert to spikes per second
        spike_rate = spike_rate * spike_rate_bin_size
        unit_array[i, :] = spike_rate
    
    # turn it into a df where each row is a uit and column contains entire spike rate trace
    # to match format of ophys dff_traces table
    spike_rate_df = pd.DataFrame(index=units.index, columns=['spike_rate'])
    for i, unit_id in enumerate(unit_ids):
        spike_rate_df.loc[unit_id, 'spike_rate'] = unit_array[i, :]
    spike_rate_df.index.name = 'unit_id'

    return spike_rate_df, timestamps+start_time


def build_tidy_cell_df(dataset, stimulus_presentations, spike_rate_bin_size=0.01):
    '''
    Builds a tidy dataframe describing activity for every unit in ephys session.
    Tidy format is defined as one row per observation.
    Thus, the output dataframe will be n_units x n_timepoints long

    Parameters:
    -----------
    dataset : AllenSDK BehaviorEcephysSession object
    stimulus_presentations: stimulus_presentations table limited to the stimuli or stimulus blocks of interest
    spike_rate_bin_size: bin size, in seconds, to use when computing spike rate over time
                0.001 = 1ms, 0.01 = 10ms, 1 = 1s (spikes / second)
    stimulus_block: stimulus block number indicating portion of VBN session to compute spike rates for
                stimulus block 0 = change detection active behavior, 1 = 10s gray screen, 2 = gabor RF mapping,
                3 = 5min gray screen, 4 = full field flashes, 5 = change detection passive replay

    Returns:
    --------
    Pandas.DataFrame
        Tidy Format (one observation per row) with the following columns:
            * timestamps (float) : the ephys timestamps for the session, computed based on bin_size
            * unit_id (int) : ecephys unit ID
            * spike_rate (float) : measured spike rate for every timestep
    '''

    spike_rate_df, timestamps = get_continous_spike_rate_for_units(dataset, stimulus_presentations, spike_rate_bin_size)
    print('spike rate df computed')
    # make an empty list to populate with dataframes for each cell
    list_of_cell_dfs = []

    # iterate over each individual unit
    for unit_id in spike_rate_df.index.values:
        # build a tidy dataframe for this unit
        cell_df = pd.DataFrame({'timestamps': timestamps,
                                'spike_rate': spike_rate_df.loc[unit_id]['spike_rate']})  # noqa E501

        # Make the unit_id column categorical
        # This will reduce memory useage since the columns
        # consist of many repeated values.
        cell_df['unit_id'] = np.int32(unit_id)
        cell_df['unit_id'] = pd.Categorical(cell_df['unit_id'], categories=spike_rate_df.index.unique())

        # append the dataframe for this cell to the list of cell dataframes
        list_of_cell_dfs.append(cell_df)

    # concatenate all dataframes in the list
    tidy_df = pd.concat(list_of_cell_dfs)
    print('tidy cell df computed')

    # return the tidy dataframe
    return tidy_df


def getImageNovelty(image_name, session_id, ecephys_sessions_table):
    '''
    Function to help annotate the stimulus_presentations table
    to indicate whether the image was novel to the mouse.      
    
    INPUT:
        image_name: str indicating which image to check for novelty (ie 'im024_r')
        session_id: the ecephys_session_id for session during which image was presented
        ecephys_sessions_table: the ecephys_sessions metadata table from the VBN cache
    
    OUTPUT:
        Returns one of the following:
        True: indicating that this image was novel for this session
        False: indicating that this image was familiar for this session
        np.nan: indicating that this stimulus wasn't one of the natural images (including omitted stimuli)

    '''
    is_novel_image_set = ecephys_sessions_table.loc[session_id]['experience_level'] == 'Novel'

    IMAGE_SET_KEY={
                'G' : ['im012_r', 'im036_r', 'im044_r', 
                    'im047_r', 'im078_r', 'im115_r'],
                'H' : ['im005_r', 'im024_r', 'im034_r', 
                    'im087_r', 'im104_r', 'im114_r'],
                'shared' : ['im083_r', 'im111_r'],
                'omitted' : 'omitted'
                }
    
    # First check that this image is one of the Natural Images used
    image_in_image_set = any([np.isin(image_name, imset) \
                              for _,imset in IMAGE_SET_KEY.items()]) 
    
    if not image_in_image_set:
        return np.nan
    
    #Get the image set for this image
    image_set_for_this_image = [name for name, image_set in IMAGE_SET_KEY.items()\
                                if image_name in image_set][0]
    
    #Get the image novelty for this image
    if image_set_for_this_image == 'omitted':
        novelty_for_this_image = np.nan
    else:
        novelty_for_this_image = is_novel_image_set and \
                            bool(np.isin(image_set_for_this_image, ['G', 'H']))

    return novelty_for_this_image