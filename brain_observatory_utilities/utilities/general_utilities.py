import numpy as np

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