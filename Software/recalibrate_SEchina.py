"""
Recalibrate SE China data
"""

from importlib import reload
import matplotlib.pyplot as plt
import xarray
import numpy as np
import seaborn as sns
import scipy.stats
import recalibrate
import pandas as pd
import pickle

nboot = 1000 # how many boot strap samples we want
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
    :param yr: Defaualt 2020.5) -- tgt year
    :return:  dists (0 = hist_yr, 1=nat_yr, 2=calib_hist, 3= calib_hist with no trend removal, 4= calib_nat)
    """
    calib, params, trends = recalibrate.calibrate(obs, sim)
    calib_hist, calib_notrend_hist, calib_nat = indep_calib(hist_yr, nat_yr, trends, params)
    dlist = [hist_yr, nat_yr, calib_hist, calib_notrend_hist, calib_nat]
    dists = dict()
    if first_guess is None:
        start_args = [[None] * 2] * len(dlist)
    else:
        start_args = first_guess
    # breakpoint()
    for data, start_arg,key in zip(dlist, start_args,['Hist', 'Natural', 'Hist_calib', 'Hist_calib_notrend', 'Nat_calib']):
        params = dist_to_use.fit(data.squeeze(), loc=np.mean(data), scale=np.std(data))
        dist = dist_to_use(*params)
        dists[key]=dist
    dists = pd.Series(dists)
    return dists


def prob_ratio(dists, obs, useCDF=False, all=False):
    """
    Compute the probability ratios. Dists are assumed to be as comp_dists
    :param dists: input distributions (0 = hist_yr, 1=nat_yr, 2=calib_hist, 3= calib_hist with no trend removal, 4= calib_nat)
    :param obs: Obs value
    :param useCDF: use CDF rather than SF
    :param all: return probabilities and ratios as pandas series
    :return: returns probability ratios as pandas series
    """
    probs = {}
    for key,d in dists.items():
        if useCDF:
            pv = d.cdf(obs)
        else:
            pv = d.sf(obs)
        probs[key]=pv

    probs = pd.Series(probs)  # convert to numpy array
    pr = pd.Series(dict(raw=probs.loc['Hist'] / probs.loc['Natural'],
                        calib=probs.loc['Hist_calib'] / probs.loc['Nat_calib'],
                        calib_notrend=probs.loc['Hist_calib_notrend'] / probs.loc['Nat_calib']))
    if all:
        return pr, probs
    else:
        return pr


nhd = recalibrate.read_data('nhd')
tas = recalibrate.read_data('tas')
pap = recalibrate.read_data('pap')

## do the recalibration.
# use the 1960-2013 data to calibrate.

# now need to fit a dist to 2020 and see how risk changes (and to the raw data),

rng = np.random.default_rng(seed=123456)  # being explicit about the seed so rng is reproducible..
## compute prob ratio and dist fits
prob_ratios = dict()
probs = dict()  # probabilities of event...
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
                      dist_to_use=dist_to_use).rename(title)

    prob_ratios[title], probs[title] = prob_ratio(dists, variab.obs_2020, useCDF=(title == 'PAP'), all=True)
    fit_dists[title] = dists
    # extract the first guess for initialising the bootstrap..
    start_args = [d.args for d in dists]

    bs_dists = []
    bs_pratio = []
    # compute obs & sim trend
    obs_trend = recalibrate.xarray_detrend(variab.obs, 'time', 1)
    sim_trend = recalibrate.xarray_detrend(variab.sim, 'time', 1)
    for n in range(nboot):
        if (n % (nboot // 10)) == 0:
            print(f"{n}..", end='')

        obs = rng.choice(obs_trend.detrend, obs_trend.detrend.shape[0]) + obs_trend.trend
        sim = rng.choice(sim_trend.detrend, sim_trend.detrend.shape[1], axis=1) + sim_trend.trend
        hist_2020 = rng.choice(variab.sim_2020, variab.sim_2020.shape[1], axis=1) + (variab.sim_2020 * 0.)
        nat_2020 = rng.choice(variab.sim_nat_2020, variab.sim_nat_2020.shape[1], axis=1) + (variab.sim_nat_2020 * 0.0)
        dists = comp_dist(obs, sim, hist_2020, nat_2020, dist_to_use=dist_to_use, first_guess=start_args)
        bs_pratio.append(prob_ratio(dists, variab.obs_2020, useCDF=(title == 'PAP')))
        bs_dists.append(dists)
    prob_ratios_bs[title] = pd.DataFrame(bs_pratio)
    fit_dists_bs[title] = pd.DataFrame(bs_dists)
    print("")

# save the data so plotting happens separately.
prob_ratios = pd.DataFrame(prob_ratios)
probs = pd.DataFrame(probs)
obs_2020 = pd.Series(obs_2020)
prob_ratios_bs = pd.concat(prob_ratios_bs)
fit_dists= pd.DataFrame(fit_dists)
fit_dists_bs = pd.concat(fit_dists_bs)

for var,filename in zip([prob_ratios,probs,obs_2020,prob_ratios_bs],['prob_ratios','probs','obs_2020','prob_ratios_bs']):
    var.to_csv(recalibrate.gen_data/(filename+'.csv'))


# Need to use pickle for scipy.stats distributions...
for var, filename in zip([fit_dists,fit_dists_bs],['fit_dists','fit_dists_bs']):
    with open(recalibrate.gen_data/(filename+'.pickle'),mode='wb') as fp:
        pickle.dump(var,fp)


