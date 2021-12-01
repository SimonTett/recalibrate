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
        params = dist_to_use.fit(data.squeeze(), loc=np.mean(data), scale=np.std(data))
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


nhd = recalibrate.read_data('nhd')
tas = recalibrate.read_data('tas')
pap = recalibrate.read_data('pap')



## do the recalibration.
# use the 1960-2013 data to calibrate.

# now need to fit a dist to 2020 and see how risk changes (and to the raw data),
nboot = 1000
rng = np.random.default_rng(seed=123456) # being explicit about the seed so rng is reproducible..
## compute prob ratio and dist fits
prob_ratios = dict()
probs = dict() # probabilities of event...
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

    prob_ratios[title], probs[title] = prob_ratio(dists, variab.obs_2020, useCDF=(title == 'PAP'),all=True)
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
## print out PRs, liklihoods and the plot the data
bs_uncert_method = 'distribution'

for key,pr in prob_ratios.items():
    uncerts, ks  = recalibrate.prob_rat_uncert(prob_ratios_bs[key],method=bs_uncert_method,pr_est=pr)
    # cases where ks is 0 then use the std bootstrap...
    if (ks is not None) and np.any(ks < 0.1):
        dd, ks2 = recalibrate.prob_rat_uncert(prob_ratios_bs[key],method='percentile',pr_est=pr)
        uncerts[:,ks < 0.1] = dd[:,ks < 0.1]

    titles = ['', 'Calib', 'Calib (no detrend)']
    for indx, p_ratio in enumerate(pr):
        print(f"{key} PR {titles[indx]}: {p_ratio:7.2g} ({uncerts[0, indx]:7.2g} - {uncerts[1, indx]:7.2g})",end=' ')
        print(f"({uncerts[0, indx]/p_ratio:5.2g} - {uncerts[1, indx]/p_ratio:5.2g})", end=' ')
    if ks is not None:
        print(f"ks:{ks[indx]:3.2f}")
    else:
        print("") # flush the output line
for key, prob in probs.items():
    # now print out the likelihood ratios

    print(f"{key} Raw: {prob[0]:7.2g} Likelihood ratio  Calib: {prob[0]/prob[2]:3.1f}  Calib -- no trend:{prob[0]/prob[3]:3.1f} )")


fig, axes = plt.subplots(nrows=3, ncols=1, num='Dists', figsize=[9, 9], clear=True)
for ax, key in zip(axes, fit_dists.keys()):
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
