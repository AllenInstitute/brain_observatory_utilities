
from brain_observastory_utilities.datasets.behavior.data_formatting import calculate_response_matrix,\
    calculate_dprime_matrix
import os
import path
import sys
import numpy as np
import matplotlib.pyplot as plt

from .plot_utils import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import brain_observatory_utilities.datasets.behavior.data_access as data  # noqa: E402


def plot_running(dataObject, ax=None):
    """ plot running speed for trial on specified dataset
    x axis is time (sec), y is running speed (cm/sec)

    Parameters
    ----------
        dataObject : (BehaviorSesson, BehaviorOphysExperiment)
            Objects provided via allensdk.brain_observatory
            module
        ax : (matplotlib.axes), optional
            Figure axes
    Returns
    ----------
    None
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel("time (sec)")
    speed, timestamps = data.get_running_speed_timeseries(dataObject)
    ax.plot(timestamps, speed,
            color=DATASTREAM_STYLE_DICT['running_speed']['color'])  # noqa: F405, E501
    ax.set_title("running speed")
    ax.set_ylabel("speed (cm/s)")
    return ax


def plot_licks(dataObject, ax=None):
    """ plot licks as tick marks on specified dataset.
    x axis is time (sec) and y axis is lick ticks

    Parameters
    ----------
        dataObject : (BehaviorSesson, BehaviorOphysExperiment)
            Objects provided via allensdk.brain_observatory module
        ax : (matplotlib.axes), optional
            Figure axes
    Returns
    ----------
    None
    """
    if ax is None:
        fig, ax = plt.subplots()

    licks = data.get_lick_timestamps(dataObject)
    ax.plot(licks, np.zeros_like(licks), marker="|",
            linestyle="none",
            color=DATASTREAM_STYLE_DICT['licks']['color'])  # noqa: F405
    return ax


def plot_rewards(dataObject, ax=None, reward_type="all"):
    """ plot rewards as blue dots on specified axis
    x axis is time (sec) and y axis is rewards.

    Parameters
    ----------
        dataObject : (BehaviorSesson, BehaviorOphysExperiment)
            Objects provided via allensdk.brain_observatory module
        ax : (matplotlib.axes), optional
            Figure axes
        reward_type : string
            options:
                "all": all rewards (auto and earned)
                "auto": only free or unearned rewards
                "earned": only earned (hit on a go trial) rewards
    Returns
    ----------
    None
    """
    if ax is None:
        fig, ax = plt.subplots()

    reward_timesestamps = data.get_reward_timestamps(dataObject,
                                                     reward_type=reward_type)
    ax.plot(
        reward_timesestamps,
        np.zeros_like(reward_timesestamps),
        marker="o",
        linestyle="none",
        color='b',
        markersize=6,
        alpha=0.25,
    )
    return ax


def plot_stimuli(dataObject, ax=None):
    """ plot stimuli as colored bars on specified axis

    Parameters
    ----------
        dataObject : (BehaviorSesson, BehaviorOphysExperiment)
            Objects provided via allensdk.brain_observatory
            module
        ax : (matplotlib.axes), optional
            Figure axes
    Returns
    ----------
    None
    """
    if ax is None:
        fig, ax = plt.subplots()

    for _, stimulus in dataObject.stimulus_presentations.iterrows():
        ax.axvspan(stimulus["start_time"],
                   stimulus["stop_time"], alpha=0.5)
    return ax


def plot_pupil_area(dataObject, ax=None):
    """ plot pupil area timeseries.
    x axis is time (sec) and y axis is pupil area
    in pixels squared.

    Parameters
    ----------
        dataObject : (BehaviorOphysExperiment)
            Object provided via allensdk.brain_observatory
            module
        ax : (matplotlib.axes), optional
            Figure axes
    Returns
    ----------
    matplotlib.axes
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel("time (sec)")

    pupil_area, timestamps = data.get_pupil_area_timeseries(dataObject)  # noqa: E501

    ax.plot(
        timestamps, pupil_area,
        color=DATASTREAM_STYLE_DICT['pupil_area']['color'],  # noqa: F405
    )
    ax.set_title(DATASTREAM_STYLE_DICT['pupil_area']['label'])  # noqa: F405
    ax.set_ylabel("pupil area\n$(pixels^2)$")
    return ax


def plot_response_matrix(stimulus_presentations, ax=None, vmin=0, vmax=1, cmap='viridis', cbar_ax=None):
    '''
    makes a plot of the response matrix given a table of stimuli

    Parameters:
    -----------
    stimulus_presentations : pandas.dataframe
        From experiment.stimulus_presentations, after annotating as follows:
            annotate_stimuli(experiment, inplace = True)
    ax : matplotlib axis
        axis on which to plot.
        If not passed, will create a figure and axis and return both
    vmin : float
        minimum for plot, default = 0
    vmax : float
        maximum for plot, default = 1
    cmap : string
        cmap for plot, default = 'viridis'
    cbar_ax : matplotlib axis
        axis on which to plot colorbar
        colorbar will not be added if not passed

    Returns:
    --------
    fig, ax if ax is not passed
    None if ax is passed
    '''

    return_fig_ax = False
    if ax is None:
        return_fig_ax = True
        fig, ax = plt.subplots()

    response_matrix = calculate_response_matrix(stimulus_presentations)

    im = ax.imshow(
        response_matrix,
        cmap=cmap,
        aspect='equal',
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(range(len(response_matrix)))
    ax.set_xticklabels(response_matrix.columns)
    ax.set_yticks(range(len(response_matrix)))
    ax.set_yticklabels(response_matrix.index)
    ax.set_xlabel('image name')
    ax.set_ylabel('previous image name')

    if cbar_ax is not None:
        plt.colorbar(im, cax=cbar_ax, label='response probability')

    if return_fig_ax:
        return fig, ax


def plot_dprime_matrix(stimulus_presentations, ax=None, vmin=0, vmax=1.5, cmap='magma', cbar_ax=None):
    '''
    makes a plot of the response matrix given a table of stimuli
    Parameters:
    -----------
    stimulus_presentations : pandas.dataframe
        From experiment.stimulus_presentations, after annotating as follows:
            annotate_stimulus_presentations(experiment, inplace = True)
    ax : matplotlib axis
        axis on which to plot.
        If not passed, will create a figure and axis and return both
    vmin : float
        minimum for plot, default = 0
    vmax : float
        maximum for plot, default = 1.5
    cmap : string
        cmap for plot, default = 'magma'
    cbar_ax : matplotlib axis
        axis on which to plot colorbar
        colorbar will not be added if not passed

    Returns:
    --------
    fig, ax if ax is not passed
    None if ax is passed
    '''

    return_fig_ax = False
    if ax is None:
        return_fig_ax = True
        fig, ax = plt.subplots()

    dprime_matrix = calculate_dprime_matrix(stimulus_presentations)

    im = ax.imshow(
        dprime_matrix,
        cmap=cmap,
        aspect='equal',
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(range(len(dprime_matrix)))
    ax.set_xticklabels(dprime_matrix.columns)
    ax.set_yticks(range(len(dprime_matrix)))
    ax.set_yticklabels(dprime_matrix.index)
    ax.set_xlabel('image name')
    ax.set_ylabel('previous image name')

    if cbar_ax is not None:
        plt.colorbar(im, cax=cbar_ax, label="d'")

    if return_fig_ax:
        return fig, ax


if __name__ == "__main__":
    # make_plots(experiment_dataset) - test
    pass
