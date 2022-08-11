import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from .plot_utils import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_access as data  # noqa: 402


def plot_max_projection(ophysObject, ax=None):
    """plots the maximum intensity projection
    for the given dataset

    Parameters
    ----------
    ophysObject : BehaviorOphysExperiment
        Object provided via allensdk.brain_observatory
        module
    ax : matplotlib.axes, optional
        Figure axes

    Returns
    -------
    matplotlib.axes
        plotting axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(ophysObject.max_projection, cmap='gray')
    return ax


def plot_average_projection(ophysObject, ax=None):
    """plots the average intensity projection
    for the given dataset

    Parameters
    ----------
    ophysObject : BehaviorOphysExperiment
        Object provided via allensdk.brain_observatory
        module
    ax : matplotlib.axes, optional
        Figure axes

    Returns
    -------
    matplotlib.axes
        plotting axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(ophysObject.average_projection, cmap='gray')
    return ax


def plot_segmentation_masks(ophysObject, ax=None):
    """plots the masks of the segmented rois/cells

    Parameters
    ----------
    ophysObject : BehaviorOphysExperiment
        Object provided via allensdk.brain_observatory
        module
    ax : matplotlib.axes, optional
        Figure axes

    Returns
    -------
    matplotlib.axes
        plotting axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(ophysObject.segmentation_mask_image)
    return ax


def plot_transparent_segmentation_masks(ophysObject, ax=None):
    """plots the segmentation masks with a transparent
    background

    Parameters
    ----------
    ophysObject : BehaviorOphysExperiment
        Object provided via allensdk.brain_observatory
        module
    ax : matplotlib.axes, optional
        Figure axes

    Returns
    -------
    matplotlib.axes
        plotting axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    transparent_mask = data.get_transparent_segmentation_mask(ophysObject)  # noqa: E501
    ax.imshow(transparent_mask, cmap='hsv', vmax=1, alpha=0.5)
    return ax


def plot_segmentation_mask_overlay(ophysObject, projection_type="max",
                                   ax=None):
    """plots the segmentation masks on top of the one of the
    projections (average intensity projection,
    max intensity projection)

    Parameters
    ----------
    ophysObject : BehaviorOphysExperiment
        Object provided via allensdk.brain_observatory
        module
    projection_type : str, optional
        by default "max",  options:
            "max" : maximum intensity projection
            "average" : average intensity projection
    ax : matplotlib.axes, optional
        Figure axes

    Returns
    -------
    matplotlib.axes
        plotting axes
    """

    if ax is None:
        fig, ax = plt.subplots()
    if projection_type == "max":
        ax = plot_max_projection(ophysObject, ax)
    elif projection_type == "average":
        ax = plot_average_projection(ophysObject, ax)
    else:
        print("Please enter a valid projection_type, \
            'max' or 'average'.")

    ax = plot_transparent_segmentation_masks(ophysObject, ax)
    ax.axis('off')
    return ax


def plot_dff(ophysObject, cell_specimen_id=None, ax=None):
    """ plots the fluroescence trace (dF/F) trace.
    x axis is time (sec) and y axis is fluorescence
    values

    Parameters
    ----------
    ophysObject :BehaviorOphysExperiment
        Object provided via allensdk.brain_observatory
        module
    cell_specimen_id : int, optional
        _description_, by default None
    ax : matplotlib.axes, optional
        Figure axes

    Returns
    -------
    matplotlib.axes
        plotting axes
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel("time (sec)")
    dff_trace, timestamps = data.get_dff_trace_timeseries(ophysObject, cell_specimen_id)  # noqa: E501
    ax.plot(timestamps, dff_trace, color=DATASTREAM_STYLE_DICT['dff']['color'])  # noqa: E501
    ax.set_title("Fluorescence trace")
    ax.set_ylabel("df/f")
    return ax


def plot_dff_heatmap(ophysObject, ax=None):
    """plots a heatmap of fluorescence activity for
    all cells in a given ophys experiment.

    Parameters
    ----------
    ophysObject : BehaviorOphysExperiment
        Object provided via allensdk.brain_observatory
        module
    ax : matplotlib.axes, optional
        Figure axes
    """
    dff_traces = data.get_all_cells_dff(ophysObject.dff_traces)
    timestamps = ophysObject.ophys_timestamps

    if ax is None:
        fig, ax = plt.subplots()

    fig, ax = plt.subplots(figsize=(20, 5))
    color_ax = ax.pcolormesh(dff_traces,
                             vmin=0, vmax=np.percentile(dff_traces, 99),
                             cmap='magma')
    # label axes
    ax.set_ylabel('cells')
    ax.set_xlabel('time (sec)')

    # x ticks
    ax.set_yticks(np.arange(0, len(dff_traces), 10))
    ax.set_xticklabels(np.arange(0, timestamps[-1], 300))

    # creating a color bar
    cb = plt.colorbar(color_ax, pad=0.015, label='dF/F')
    return ax
