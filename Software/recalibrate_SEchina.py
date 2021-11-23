"""
Recalibrate SE China data
"""

from importlib import reload
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import xarray
import numpy as np
import seaborn as sns
import collections
import scipy.stats
import cftime
import recalibrate



def read_data(var):
    """
    Read sim & obs data then make any fixes needed
    :param var:
    :return: obs & sim data as a named tuple.
    """
    data_dir = pathlib.Path('../data')
    obs = xarray.load_dataset(data_dir / f'{var}_obs_jja_1961-2020_ts.nc')[var]
    sim = xarray.load_dataset(data_dir / f'{var}_model_15runs_jja_1961-2020_ts.nc')[var]
    sim_2020 = pd.read_csv(data_dir / f'{var}_model_all_anom_2020.txt', header=None, squeeze=True)
    sim_nat_2020 = pd.read_csv(data_dir / f'{var}_model_nat_anom_2020.txt', header=None, squeeze=True)
    # convert 2020 data to xarray with the right time
    time_fn = cftime.Datetime360Day  # shoudl eventually be cftime.datatime
    t = time_fn(2020, 7, 16, calendar='360_day')  # this should eventually go to cftime.datatime(...,calendar='360_day')
    sim_2020 = xarray.DataArray(sim_2020).rename(dim_0='model').expand_dims(time=1).assign_coords(time=[t])
    sim_nat_2020 = xarray.DataArray(sim_nat_2020).rename(dim_0='model').expand_dims(time=1).assign_coords(time=[t])
    # fix obs
    obs_2020 = float(obs[-1])
    for c in ['ncl0', 'ncl3']:
        try:
            obs = obs.sel({c: slice(0, 53)}).rename({c: 'time'})
            break  # worked so stop trying to rename
        except KeyError:  # don't have c so go on!
            pass

    # fix sim,
    for c in ['ncl0', 'ncl1']:
        try:
            sim = sim.rename({c: 'time'})
            # managed to rename so need to sort times...
            time = [time_fn(yr, 7, 16) for yr in range(1961, 2014)]
            sim = sim.assign_coords(time=time)
            break  # worked so stop trying to rename
        except ValueError:  # don't have c so go on!
            pass

    obs = obs.assign_coords(time=sim.time)

    named_result = collections.namedtuple('data', ['obs', 'sim', 'obs_2020', 'sim_2020', 'sim_nat_2020'])
    return named_result(obs, sim, obs_2020, sim_2020, sim_nat_2020)


def indep_calib(sim, sim_nat, trend, params):
    """
    Apply the calibration to the 2020 data and return it
    :param sim: hist data for 2020
    :param sim_nat: Nat data for 2020
    :param trend: trend info
    :param params: params
    :return: calib2020, calib2020_notrend, nat_calib2020
    """

    # now apply the calibrations to the 2020 data.
    mn = float(sim.mean())
    sim_trend_yr = xarray.polyval(sim.time, trend.simulated.polyfit_coefficients)  # component of trend for this year
    delta_trend = mn - sim_trend_yr
    #
    calib = (sim - mn) * params.beta + delta_trend * params.alpha + sim_trend_yr * params.gamma
    calib_notrend = (sim - mn) * params.beta + delta_trend * params.alpha + sim_trend_yr
    mn_nat = sim_nat.mean()
    nat_calib = (sim_nat - mn_nat) * params.beta + mn_nat  # no scaling on the deviation from the mn

    return calib, calib_notrend, nat_calib


# noinspection SpellCheckingInspection
def comp_dist(obs, sim, hist_yr, nat_yr, dist_to_use=scipy.stats.norm, first_guess=None):
    """

    Fit dist_to_use to raw, calibrated and calibrated without trend hist_yr  as well as to calibrated

    :param obs: Observations for calibration period
    :param sim: Simulations for calibration period
    :param hist_yr: Hist Simulation in tgt year
    :param nat_yr: Nat Simulations in tgt year
    :param yr: Defaualt 202.5) -- tgt year
    :return:  dists (0 = hist_yr, 1=nat_yr, 2=calib_hist, 3= calib_hist with no trend removal, 4= calib_nat)
    """
    calib, params, trends = recalibrate.calibrate(obs, sim)
    calib_hist, calib_notrend_hist, calib_nat = indep_calib(hist_yr, nat_yr, trends, params)
    dlist = [hist_yr, nat_yr, calib_hist, calib_notrend_hist, calib_nat]
    dists = []
    if first_guess is None:
        start_args = [[None] * 2] * len(dlist)
    else:
        start_args = first_guess
    # breakpoint()
    for data, start_arg in zip(dlist, start_args):
        params = dist_to_use.fit(data, loc=np.mean(data), scale=np.std(data))
        dist = dist_to_use(*params)
        dists.append(dist)
    return dists


def prob_ratio(dists, obs, useCDF=False, all=False):
    """
    Compute the probability ratios. Dists are assumed to be as comp_dists
    :param dists: input distributions (0 = hist_yr, 1=nat_yr, 2=calib_hist, 3= calib_hist with no trend removal, 4= calib_nat)
    :param obs: Obs value
    :param useCDF: use CDF rather than SF
    :param all: return probabilities and ratios
    :return: returns probability ratios 0/1 2/4 3/4
    """
    probs = []
    for d in dists:
        if useCDF:
            pv = d.cdf(obs)
        else:
            pv = d.sf(obs)
        probs.append(pv)
    probs = np.array(probs)  # convert to numpy array
    pr = np.array([probs[0] / probs[1], probs[2] / probs[4], probs[3] / probs[4]])
    if all:
        return pr, probs
    else:
        return pr


def resample(var):
    """
    Resample var -- an xarray dataset
    :param var: variable to be resampled
    :return:
    """
    if var.ndim == 1:
        lvar = len(var)
    elif var.ndim == 2:
        lvar = var.shape[1]
    else:
        raise Exception("Only deal with 1 or 2D data")

    ind = np.sort(rng.choice(lvar, size=lvar))  # index.
    if var.ndim == 1:
        sample = var[ind]
    else:
        sample = var[:, ind]

    return sample


nhd = read_data('nhd')
tas = read_data('tas')
pap = read_data('pap')

reliab = dict()
bins = 3
pbin = np.linspace(0, 1.0, bins, endpoint=False)
delta_p = 1 / (2 * bins)  # offset for centre of bin
for title, data in zip(['TAS', 'NHD', 'PAP'], [tas, pap, nhd]):
    obs_quant = recalibrate.quantile(data.obs)
    sim_quant = recalibrate.quantile(data.sim)
    # let's get probability of each simulated/forecast value.
    forecast_p = recalibrate.forecast_prob(sim_quant)
    forecast_pq = (xarray.apply_ufunc(np.digitize, forecast_p, kwargs=dict(bins=pbin)))
    # and then the forecast reliability
    reliab[title] = recalibrate.reliability(forecast_pq, obs_quant)

## now to plot
fig, axes = plt.subplots(nrows=3, ncols=2, clear=True, sharex=True, sharey=True,
                         figsize=[8, 9], num='reliability')

for ax, ax2, (name, r) in zip(axes[:, 0], axes[:, 1], reliab.items()):
    sns.heatmap(r.reliab.to_pandas(), ax=ax, annot=True, robust=True, fmt='4.2f', center=0.5, cbar=False)
    sns.heatmap(r.fCount.to_pandas().astype(int), ax=ax2, annot=True, robust=True, fmt='2d', center=0.5, cbar=False)
    ax.set_title(f"{name} Reliability")
    ax2.set_title(f"{name} Fcount")
    for a in [ax, ax2]:
        a.set_ylabel("Quantile")
        a.set_xlabel("P Quantile")
    ax.invert_yaxis()
fig.tight_layout()
fig.show()

recalibrate.saveFig(fig)

## next do the recalibration.
# use the 1960-2013 data to calibrate.

# now need to fit a dist to 2020 and see how risk changes (and to the raw data),
nboot = 1000
rng = np.random.default_rng(seed=123456) # being explicit about the seed so rng is reproducible..
## compute prob ratio and dist fits
prob_ratios = dict()
fit_dists = dict()
obs_2020 = dict()
prob_ratios_bs = dict()
fit_dists_bs = dict()
for variab, dist_to_use, title in zip(
        [tas, nhd, pap],  # data
        [scipy.stats.norm, scipy.stats.norm, scipy.stats.genextreme],  # distributions to fit
        ['TAS', 'NHD', 'PAP']  # titles
):
    obs_2020[title] = variab.obs_2020
    dists = comp_dist(variab.obs, variab.sim, variab.sim_2020, variab.sim_nat_2020,
                      dist_to_use=dist_to_use)

    prob_ratios[title] = prob_ratio(dists, variab.obs_2020, useCDF=(title == 'PAP'))
    fit_dists[title] = dists
    # extract the first guess for initialising the bootstrap..
    start_args = [d.args for d in dists]

    bs_dists = []
    bs_pratio = []
    # compute obs & sim trend
    obs_trend = recalibrate.xarray_detrend(variab.obs, 'time', 1)
    sim_trend = recalibrate.xarray_detrend(variab.sim, 'time', 1)
    for n in range(nboot):
        if (n % (nboot//10)) == 0:
            print(f"{n}..", end='')

        obs = rng.choice(obs_trend.detrend, obs_trend.detrend.shape[0]) + obs_trend.trend
        sim = rng.choice(sim_trend.detrend, sim_trend.detrend.shape[1], axis=1) + sim_trend.trend
        hist_2020 = rng.choice(variab.sim_2020, variab.sim_2020.shape[1], axis=1) + (variab.sim_2020 * 0.)
        nat_2020 = rng.choice(variab.sim_nat_2020, variab.sim_nat_2020.shape[1], axis=1) + (variab.sim_nat_2020 * 0.0)
        dists = comp_dist(obs, sim, hist_2020, nat_2020, dist_to_use=dist_to_use, first_guess=start_args)
        bs_pratio.append(prob_ratio(dists, variab.obs_2020, useCDF=(title == 'PAP')))
        bs_dists.append(dists)
    prob_ratios_bs[title] = np.array(bs_pratio)
    fit_dists_bs[title] = bs_dists
    print("")
## plot the data
bs_uncert_method = 'distribution'
# options are '
fig, axes = plt.subplots(nrows=3, ncols=1, num='Dists', figsize=[9, 9], clear=True)

for ax, key in zip(axes, fit_dists.keys()):
    pr = prob_ratios[key]
    uncerts, ks  = recalibrate.prob_rat_uncert(prob_ratios_bs[key],method=bs_uncert_method,pr_est=pr)
    # cases where ks is 0 then use the std bootstrap...
    if (ks is not None) and np.any(ks < 0.1):
        dd, ks2 = recalibrate.prob_rat_uncert(prob_ratios_bs[key],method='percentile',pr_est=pr)
        uncerts[:,ks < 0.1] = dd[:,ks < 0.1]

    titles = ['', 'Calib', 'Calib (no detrend)']
    for indx, p_ratio in enumerate(pr):
        print(f"{key} PR {titles[indx]}: {p_ratio:7.2g} ({uncerts[0, indx]:7.2g} - {uncerts[1, indx]:7.2g})",end=' ')
        if ks is not None:
            print(f"ks:{ks[indx]:3.2f}")
        else:
            print("") # flush the output linek

    if key == 'PAP':  # interested in drought...
        ylabel = 'p < threshold'
    else:
        ylabel = 'p > threshold'

    for dist, label, color, linestyle in zip(fit_dists[key],
                                             ['Hist 2020', 'Nat 2020', 'Calib Hist 2020',
                                              'Calib (no trend) Hist 2020', 'Calib Nat 2020'],
                                             ['red', 'blue', 'red', 'purple', 'blue'],
                                             ['solid', 'solid', 'dashed', 'dashed', 'dashed']):
        v_range = dist.ppf([1e-5, 1 - 1e-5])
        p = np.linspace(v_range[0], v_range[1], 500)
        if key == 'PAP':  # interested in drought...
            fn = dist.cdf
        else:
            fn = dist.sf
        ax.plot(p, fn(p), color=color, linewidth=2, linestyle=linestyle, label=label)
    ax.axvline(obs_2020[key], linewidth=3, color='gray', label='2020 Obs')
    ax.axhline(1e-2,linewidth=2,color='grey',linestyle='dashed')
    ax.set_yscale('log')
    ax.set_title(key)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Threshold")

axes[0].legend()
fig.tight_layout()
fig.show()

recalibrate.saveFig(fig)
