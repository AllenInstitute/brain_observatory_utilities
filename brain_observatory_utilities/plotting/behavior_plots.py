
import os
import path
import sys
import numpy as np
import matplotlib.pyplot as plt

from .plot_utils import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_access as data  # noqa: E402


def plot_behavioral_streams(dataObject):
    """ plot behavioral streams including running,
        licks, rewards, and df/f streams.

    Parameters
    ----------
        dataObject : (BehaviorSesson, BehaviorOphysExperiment)
            Objects provided via allensdk.brain_observatory
            module
    Returns
    ----------
        MatPlotLib: figure and axes
    """
    experiment = False
    fig, axes = None, None

    if "ophys_experiment_id" in dataObject.list_data_attributes_and_methods():
        print("experiment=True")
        fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
        experiment = True
    else:
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    for ax in axes:
        plot_stimuli(dataObject, ax)

    plot_running(dataObject, axes[0])
    plot_licks(dataObject, axes[1])
    plot_rewards(dataObject, axes[1])

    axes[1].set_title("licks and rewards")
    axes[1].set_yticks([])
    axes[1].legend(["licks", "rewards"])

    if experiment:
        plot_pupil_area(dataObject, axes[2])

    fig.tight_layout()
    return fig, axes


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


if __name__ == "__main__":
    # make_plots(experiment_dataset) - test
    pass
