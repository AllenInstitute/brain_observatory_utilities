import numpy as np
import pandas as pd
from scipy.stats import norm


def dprime(hit_rate=None, fa_rate=None, go_trials=None, catch_trials=None, limits=False):
    '''
    calculates the d-prime for a given hit rate and false alarm rate

    https://en.wikipedia.org/wiki/Sensitivity_index

    Parameters
    ----------
    hit_rate : float or vector of floats
        rate of hits in the True class
    fa_rate : float or vector of floats
        rate of false alarms in the False class
    go_trials: vector of booleans
        responses on all go trials (hit = True, miss = False)
    catch_trials: vector of booleans
        responses on all catch trials (false alarm = True, correct reject = False)
    limits : boolean or tuple, optional
        limits on extreme values, which can cause d' to overestimate on low trial counts.
        False (default) results in limits of (0.01,0.99) to avoid infinite calculations
        True results in limits being calculated based on trial count (only applicable if go_trials and catch_trials are passed)
        (limits[0], limits[1]) results in specified limits being applied

    Note: user must pass EITHER hit_rate and fa_rate OR go_trials and catch trials

    Returns
    -------
    d_prime

    Examples
    --------
    With hit and false alarm rates of 0 and 1, if we pass in limits of (0, 1) we are
    allowing the raw probabilities to be used in the dprime calculation.
    This will result in an infinite dprime:

    >> dprime(hit_rate = 1.0, fa_rate = 0.0, limits = (0, 1))
    np.inf

    If we do not pass limits, the default limits of 0.01 and 0.99 will be used
    which will convert the hit rate of 1 to 0.99 and the false alarm rate of 0 to 0.01.
    This will prevent the d' calcluation from being infinite:

    >>> dprime(hit_rate = 1.0, fa_rate = 0.0)
    4.6526957480816815

    If the hit and false alarm rates are already within the limits, the limits don't apply
    >>> dprime(hit_rate = 0.6, fa_rate = 0.4)
    0.5066942062715994

    Alternately, instead of passing in pre-computed hit and false alarm rates,
    we can pass in a vector of results on go-trials and catch-trials.
    Then, if we call `limits = True`, the limits will be calculated based
    on the number of trials in both the go and catch trial vectors
    using the `trial_number_limit` function.

    For example, for low trial counts, even perfect performance (hit rate = 1, false alarm rate = 0)
    leads to a lower estimate of d', given that we have low confidence in the hit and false alarm rates;

    >>> dprime(
            go_trials = [1, 1, 1],
            catch_trials = [0, 0],
            limits = True
            )
    1.6419113162977828

    At the limit, if we have only one sample of both trial types, the `trial_number_limit`
    pushes our estimated response probability to 0.5 for both the hit and false alarm rates,
    giving us a d' value of 0:

    >>> dprime(
            go_trials = [1],
            catch_trials = [0],
            limits = True
            )
    0.0

    And with higher trial counts, the `trial_number_limit` allows the hit and false alarm
    rates to get asymptotically closer to 0 and 1, leading to higher values of d'.
    For example, for 10 trials of each type:

    >>> dprime(
            go_trials = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            catch_trials = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            limits = True
        )
    3.289707253902945

    Or, if we had 20 hit trials and 10 false alarm trials:

    >>> dprime(
            go_trials = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            catch_trials = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            limits = True
        )
    3.604817611491527

    Note also that Boolean vectors can be passed:

    >>> dprime(
            go_trials = [True, False, True, True, False],
            catch_trials = [False, True, False, False, False],
            limits = True
        )
    1.094968336708714
    '''

    assert hit_rate is not None or go_trials is not None, 'must either a specify `hit_rate` or pass a boolean vector of `go_trials`'
    assert fa_rate is not None or catch_trials is not None, 'must either a specify `fa_rate` or pass a boolean vector of `catch_trials`'

    assert hit_rate is None or go_trials is None, 'do not pass both `hit_rate` and a boolean vector of `go_trials`'
    assert fa_rate is None or catch_trials is None, 'do not pass both `fa_rate` and a boolean vector of `catch_trials`'

    assert not (hit_rate is not None and limits is True), 'limits can only be calculated if a go_trials vector is passed, not a hit_rate'
    assert not (fa_rate is not None and limits is True), 'limits can only be calculated if a catch_trials vector is passed, not a fa_rate'

    # calculate hit and fa rates as mean of boolean vectors
    if hit_rate is None:
        hit_rate = np.mean(go_trials)
    if fa_rate is None:
        fa_rate = np.mean(catch_trials)

    Z = norm.ppf

    if limits is False:
        # if limits are False, apply default
        limits = (0.01, 0.99)
    elif limits is True:
        # clip the hit and fa rate based on trial count
        hit_rate = response_probabilities_trial_number_limit(
            hit_rate, len(go_trials))
        fa_rate = response_probabilities_trial_number_limit(
            fa_rate, len(catch_trials))

    if limits is not True:
        # Limit values in order to avoid d' infinity
        hit_rate = np.clip(hit_rate, limits[0], limits[1])
        fa_rate = np.clip(fa_rate, limits[0], limits[1])

    # keep track of nan locations
    hit_rate = pd.Series(hit_rate)
    fa_rate = pd.Series(fa_rate)
    hit_rate_nan_locs = list(hit_rate[pd.isnull(hit_rate)].index)
    fa_rate_nan_locs = list(fa_rate[pd.isnull(fa_rate)].index)

    # fill nans with 0.0 to avoid warning about nans
    d_prime = Z(hit_rate.fillna(0)) - Z(fa_rate.fillna(0))

    # for every location in hit_rate and fa_rate with a nan, fill d_prime with a nan
    for nan_locs in [hit_rate_nan_locs, fa_rate_nan_locs]:
        d_prime[nan_locs] = np.nan

    if len(d_prime) == 1:
        # if the result is a 1-length vector, return as a scalar
        return d_prime[0]
    else:
        return d_prime


def response_probabilities_trial_number_limit(P, N):
    '''
    Calculates limits on response probability estimate based on trial count.
    An important point to note about d' is that the metric will be infinite with
    perfect performance, given that Z(0) = -infinity and Z(1) = infinity.
    Low trial counts exacerbate this issue. For example, with only one sample of a go-trial,
    the hit rate will either be 1 or 0. Macmillan and Creelman [1] offer a correction on the hit and false
    alarm rates to avoid infinite values whereby the response probabilities (P) are bounded by
    functions of the trial count, N:

        1/(2N) < P < 1 - 1/(2N)

    Thus, for the example of just a single trial, the trial-corrected hit rate would be 0.5.
    Or after only two trials, the hit rate could take on the values of 0.25, 0.5, or 0.75.

    [1] Macmillan, Neil A., and C. Douglas Creelman. Detection theory: A user's guide. Psychology press, 2004.
    Parameters
    ----------
    P : float
        response probability, bounded by 0 and 1
    N : int
        number of trials used in the response probability calculation

    Returns
    -------
    P_corrected : float
        The response probability after adjusting for the trial count

    Examples
    --------
    Passing in a response probability of 1 and trial count of 10 will lead to a reduced
    estimate of the response probability:

    >>> trial_number_limit(1, 10)
    0.95

    A higher trial count increases the bounds on the estimate:

    >>> trial_number_limit(1, 50)
    0.99

    At the limit, the bounds are 0 and 1:

    >>> trial_number_limit(1, np.inf)
    1.0

    The bounds apply for estimates near 0 also:

    >>> trial_number_limit(0, 1)
    0.5

    >>> trial_number_limit(0, 2)
    0.25

    >>> trial_number_limit(0, 3)
    0.16666666666666666

    Note that passing a probality that is inside of the bounds
    results in no change to the passed probability:

    >>> trial_number_limit(0.25, 3)
    0.25
    '''
    if N == 0:
        return np.nan
    if not pd.isnull(P):
        P = np.max((P, 1. / (2 * N)))
        P = np.min((P, 1 - 1. / (2 * N)))
    return P
