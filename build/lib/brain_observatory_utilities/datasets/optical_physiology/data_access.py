
import numpy as np
from brain_observatory_utilities.datasets.behavior.data_access import *
import brain_observatory_utilities.utilities.general_utilities as gen_utils



def get_dff_trace_timeseries(ophysObject, cell_specimen_id=None):
    """ By default will return the average dff trace (mean
        of all cell dff traces) for an ophys experiment. If
        cell_specimen_id is specified then will return the
        dff trace for that single cell specimen id.

    Parameters
    ----------
    ophysObject : (BehaviorOphysExperiment)
        Object provided via allensdk.brain_observatory
        module
    cell_specimen_id : int
        unified id of segmented cell across experiments
        (assigned after cell matching). Will return dff
        trace for this single cell_specimen_id.

    Returns
    -------
    tuple
        dff_trace, timestamps (sec)
    """
    dff_traces_df = ophysObject.dff_traces.reset_index()
    if cell_specimen_id is None:
        dff = get_all_cells_mean_dff(dff_traces_df)
    else:
        dff = get_cell_dff(dff_traces_df, cell_specimen_id)
    timestamps = ophysObject.ophys_timestamps
    return dff, timestamps


def get_all_cells_dff(dff_traces_df):
    """gets the dff traces for all cells
    in a BehaviorOphysExperiment

    Parameters
    ----------
    dff_traces_df :pandas dataframe
        dff_traces attribute of a BehaviorOphysExperiment
        object.

    Returns
    -------
    array
        array of arrays with each second level array containing
        the dff timeseries values for a single cell
    """
    dff_trace_array = np.vstack(dff_traces_df['dff'].values)
    return dff_trace_array


def get_cell_dff(dff_traces_df, cell_specimen_id):
    """_summary_

    Parameters
    ----------
    dff_traces_df : pandas dataframe
        dff_traces attribute of a BehaviorOphysExperiment
        object.
    cell_specimen_id : int
       unified id of segmented cell across experiments
       (assigned after cell matching).

    Returns
    -------
    array
       dff timeseries values for the given specified cell
    """

    cell_dff = dff_traces_df.loc[dff_traces_df["cell_specimen_id"] == cell_specimen_id, "dff"].values[0]  # noqa: E501
    return cell_dff


def get_all_cells_mean_dff(dff_traces_df):
    """gets the mean dff trace for all cells
    for a given BehaviorOphysExperiment

    Parameters
    ----------
    dff_traces_df : pandas dataframe
        dff_traces attribute of a BehaviorOphysExperiment
        object.
    Returns
    -------
    array
        mean dff timeseries values for all cells
    """
    mean_dff = gen_utils.average_df_timeseries_values(dff_traces_df, 'dff')
    return mean_dff


def get_transparent_segmentation_mask(ophysObject):
    """transforms the segmentation mask image to
    make the background transparent

    Parameters
    ----------
    ophysObject : (BehaviorOphysExperiment)
        Object provided via allensdk.brain_observatory
        module
    Returns
    -------
    2D image
        1 where an ROI mask and NaN everywhere else
    """
    segmentation_mask = ophysObject.segmentation_mask_image
    transparent_mask = np.zeros(segmentation_mask[0].shape)
    transparent_mask[:] = np.nan
    transparent_mask[segmentation_mask[0] == 1] = 1
    return transparent_mask


