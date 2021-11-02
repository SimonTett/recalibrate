"""
Recalibrate SE China data
"""

import recalibrate
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import xarray
import numpy as np
import seaborn as sns
import collections
import scipy.stats
import cftime


def read_data(var):
    """
    Read sim & obs data then make any fixes needed
    :param var:
    :return: obs & sim data
    """
    data_dir = pathlib.Path('../data')
    obs = xarray.load_dataset(data_dir / f'{var}_obs_jja_1961-2020_ts.nc')[var]
    sim = xarray.load_dataset(data_dir / f'{var}_model_15runs_jja_1961-2020_ts.nc')[var]
    sim_2020 = pd.read_csv(data_dir / f'{var}_model_all_anom_2020.txt', header=None)
    sim_nat_2020 = pd.read_csv(data_dir / f'{var}_model_nat_anom_2020.txt', header=None)
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
            time = [cftime.datetime(yr, 7, 16, calendar='360_day') for yr in range(1961, 2014)]
            sim = sim.assign_coords(time=time)
            break  # worked so stop trying to rename
        except ValueError:  # don't have c so go on!
            pass

    obs = obs.assign_coords(time=sim.time)

    named_result = collections.namedtuple('data', ['obs', 'sim', 'obs_2020', 'sim_2020', 'sim_nat_2020'])
    return named_result(obs, sim, obs_2020, sim_2020, sim_nat_2020)





nhd = read_data('nhd')
tas = read_data('tas')
pap = read_data('pap')

reliab = dict()
bins = 3
pbin = np.linspace(0, 1.0, bins, endpoint=False)
delta_p = 1 / (2 * bins)  # offset for centre of bin
for title, data in zip(['TAS', 'PAP', 'NHD'], [tas, pap, nhd]):
    obs_quant = recalibrate.quantile(data.obs)
    sim_quant = recalibrate.quantile(data.sim)
    # let's get probability of each simulated/forecast value.
    forecast_p = recalibrate.forecast_prob(sim_quant)
    forecast_pq = (xarray.apply_ufunc(np.digitize, forecast_p, kwargs=dict(bins=pbin)))
    # and then the forecast reliability
    reliab[title] = recalibrate.reliability(forecast_pq, obs_quant)

## now to plot
fig, axes = plt.subplots(nrows=3, ncols=1, clear=True, figsize=[8, 9], num='reliability')

for ax, (name, r) in zip(axes, reliab.items()):
    sns.heatmap(r.to_pandas(), ax=ax, annot=True, robust=True, fmt='4.2f', center=0.5, cbar=False)
    ax.set_ylabel("Quantile")
    ax.set_xlabel("Forecast Probability Quantile")
    ax.set_title(f"{name} Reliability")
    ax.invert_yaxis()
fig.tight_layout()
fig.show()

## next do the recalibration. Will switch of the trend recalibration.
# use the 1960-2013 data to calibrate.

# now need to fit a dist to 2020 and see how risk changes (and to the raw data),

fig, axes = plt.subplots(nrows=3, ncols=1, num='Dists', figsize=[9, 9], clear=True)

for variab, dist_to_use, title, ax in zip([tas, pap, nhd],
                                          [scipy.stats.norm, scipy.stats.genextreme, scipy.stats.norm],
                                          ['TAS', 'PAP', 'NHD'], axes):
    calib, params, trend = calibrate(variab.obs, variab.sim)
    # now apply the calibrations to the 2020 data. Will not apply linear trend scaling
    mn = float(variab.sim_2020.mean())
    yr = 2020 + (7 - 1) / 12.
    trend2020 = trend.simulated.intercept + trend.simulated.slope * yr
    delta_trend = mn - trend2020
    calib2020 = (variab.sim_2020 - mn) * params.beta + delta_trend * params.alpha + trend2020*params.gamma
    calib2020_notrend = (variab.sim_2020 - mn) * params.beta + delta_trend * params.alpha + trend2020
    mn_nat = variab.sim_nat_2020.mean()
    nat_calib2020 = (variab.sim_nat_2020 - mn_nat) * params.beta + mn_nat * params.alpha # no scaling on the deviation from the mn
    prob_obs = []
    if title == 'PAP':  # interested in drought...
        ylabel = 'p < threshold'
    else:
        ylabel = 'p > threshold'
    for label, data, color, linestyle in zip(['Hist 2020', 'Nat 2020', 'Calib Hist 2020',
                                              'Calib (no trend) Hist 2020','Calib Nat 2020'],
                                             [variab.sim_2020, variab.sim_nat_2020, calib2020, calib2020_notrend,nat_calib2020],
                                             ['red', 'blue', 'red', 'purple', 'blue'], ['solid', 'solid', 'dashed', 'dashed','dashed']):
        dist = dist_to_use(*dist_to_use.fit(data))
        v_range = dist.ppf([1e-5, 1 - 1e-5])
        p = np.linspace(v_range[0], v_range[1], 500)
        if title == 'PAP':  # interested in drought...
            fn = dist.cdf

        else:
            fn = dist.sf

        ax.plot(p, fn(p), color=color, linewidth=2, linestyle=linestyle, label=label)
        prob_obs.append(fn(variab.obs_2020))
    ax.axvline(variab.obs_2020, linewidth=3, color='gray', label='2020 Obs')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Threshold")
    print(f"{title} PR: {prob_obs[0] / prob_obs[1]:8.3g}")
    print(f"{title} Calib PR: {prob_obs[2] / prob_obs[4]:8.3g}")
    print(f"{title} Calib (notrend) PR: {prob_obs[3] / prob_obs[4]:8.3g}")
axes[0].legend()
fig.tight_layout()
fig.show()
