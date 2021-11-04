# module to provide some functions for recalibration work
import numpy as np
import scipy.stats
import scipy.signal
import collections
import xarray
import pathlib


def quantile(dataArray, binsize=None, dist=None):
    """

    :param dataArray: Data as a xarray dataArray.
    :param binsize: binsize (default is None). Number of bins wanted. If None the default value of 5 used.
    :param dist: Distribution used to fit simulated data. Default is normal. Not actually used.
    :return: reliability score.


    """

    if binsize is None:
        binsize = 3  # tercile forecasts by default

    pvalues = np.linspace(0, 1, binsize, endpoint=False)  # terciles.

    # compute the simulated quantiles
    quantiles = dataArray.quantile(pvalues)
    # want to workout quantile for obs in each time-bin.
    quant = xarray.apply_ufunc(np.digitize, dataArray, kwargs=dict(bins=quantiles.values))

    return quant


def forecast_prob(sim_quant, ensemble_dim='model'):
    """
    Compute the (empirical) probability of forecast for each quantile
    :param sim_quant: data of for each ensemble/time the simulated quantile that was forecase
    :param ensemble_dim (default model). The name of the ensemble dimension.
    :return: dataArray of probability of forecast for each quantile/
    """

    da = []
    qv = np.unique(sim_quant)
    for v in qv:
        d = (sim_quant == v).mean(ensemble_dim).assign_coords(quant=v)
        da.append(d)

    forecast_p = xarray.concat(da, dim='quant')  # this is the forecast probability for each quantile.

    return forecast_p


def reliability(forecast_pq, obs_q):
    """
    Compute reliability dataArray (fraction of obs in each forecast bin)
    :param forecast_pq: xarray of forecast quantiles for each quantile
    :param obs_q: which quantised bin obs were in.
    :return: a reliability  dataSet with  two variables each quant x prob_quant
      reliab: each bin is  the fraction of obs that are in the same bin as the forecast.
      fCount: each bin is the number of forecasts in each bin.
    """

    # now want to see how many times each quantile occurs for each forecast value.
    pquant = np.unique(forecast_pq)
    reliab = xarray.DataArray(np.zeros([len(forecast_pq.quant), len(pquant)]),
                              coords=dict(quant=forecast_pq.quant, prob_quant=pquant),
                              name='reliab')
    fCount = reliab.copy().rename('fCount') # where we will put the forecast count
    for quant in reliab.quant:
        f = forecast_pq.sel(quant=quant)
        omsk = obs_q == quant  # obs in tgt quantile
        for qp in pquant:
            pos = dict(quant=quant, prob_quant=qp)
            L = f == qp  # how many cases for this forecast quantile.
            reliab.loc[pos] = (omsk & L).sum() / L.sum()
            fCount.loc[pos] = L.sum()

    return xarray.Dataset(data_vars=dict(reliab=reliab,fCount=fCount))


def calibrate(observed, simulated, alpha=None, beta=None, gamma=None,
              ensemble_dim='model', time_dim='time'):
    """
    (re)Calibrate simulated data against observations.
    Based on Bellprat, 2019 paper doi:10.1038/s41467-019-09729-2.
    Original code written by Nico Freychet.
    Converted to use xarray and documented by Simon Tett

    :param observed -- observed data.
    :param simulated -- simulated ensemble data.
       Both observed and simulated should be xarray DataArrays.


    :param alpha -- scaling on the de-trended ensemble mean (set alpha =1 to not adjust this term)
    :param beta -- scaling on the deviation from the ensemble mean (set beta = 1 to not adjust this term)
    :param gamma -- scaling on the linear trend (set gamma = 1 to not adjust the linear trend).
    For the parameters alpha, beta & gamma if they are not specified (None) then they will be computed.
      If they are provided  then they will be used.

    :param ensemble_dim -- name of the ensemble dimension. Default is 'model'
    :param time_dim -- name of the time dimension. Default is 'time'
    Returns:
      calibrated model ensemble, named tuple of alpha, beta & gamma params with obvious names,
        & named tuple of trends (observed & simulated)
    """
    simulated_mean = simulated.mean(ensemble_dim)  # mean over ensemble = dim#1
    years = simulated_mean[time_dim].dt.year + (simulated_mean[time_dim].dt.month - 1) / 12.
    # think this should really be days since the start but we would then need to take account of multiple calendars..
    obs_trend = scipy.stats.linregress(years, observed)

    observed_dt = observed - (obs_trend.intercept + years * obs_trend.slope)
    simulated_trend = scipy.stats.linregress(years, simulated_mean)
    simulated_mean_dt = simulated_mean - (simulated_trend.intercept + years * simulated_trend.slope)
    simulated_ens_diff = simulated - simulated_mean  # remove the ensemble mean.

    # compute coefficients (if not provided)
    if alpha is None:
        corr, pv = scipy.stats.pearsonr(simulated_mean_dt, observed_dt)  # correlation of *detrended* obs vs ens mean
        alpha = float(np.abs(corr) * observed.std() / simulated_mean.std())

    if beta is None:
        corr, pv = scipy.stats.pearsonr(simulated_mean_dt, observed_dt)  # correlation of obs vs ens mean
        ens_var = simulated_ens_diff.var(ensemble_dim).mean(time_dim)  # mean intra-ens var
        beta = float(np.sqrt(1 - corr * observed.std() / np.sqrt(ens_var)))
    if gamma is None:
        gamma = float(obs_trend.slope / simulated_trend.slope)

    mod_calib = alpha * simulated_mean_dt + (beta * simulated_ens_diff) + (gamma * (simulated_trend.slope * years))
    # rescale the detrended model-mean, inflate the variances and correct to use observed trend (correcting to use the observed trend is a
    # particularly bad idea...)

    param_named = collections.namedtuple('params', ['alpha', 'beta', 'gamma'])
    params = param_named(alpha=alpha, beta=beta, gamma=gamma)
    trend_named = collections.namedtuple('trends', ['observed', 'simulated'])
    trends = trend_named(observed=obs_trend, simulated=simulated_trend)
    # wrap parameters and trends in named tuples to make it easier for callee to do things
    # with them.
    return mod_calib, params, trends

fig_dir = pathlib.Path("figures")

def saveFig(fig, name=None, savedir=None, figtype=None, dpi=None, verbose=False):
    """


    :param fig -- figure to save
    :param name (optional) set to None if undefined
    :param savedir (optional) directory as a pathlib.Path. Path to save figure to. Default is fig_dir
    :param figtype (optional) type of figure. (If not specified then png will be used)
    :param dpi: dots per inch to save at. Default is none which uses matplotlib default.
    :param verbose:  If True (default False) printout name of file being written to
    """

    defFigType = '.png'
    if dpi is None:
        dpi = 300
    # set up defaults
    if figtype is None:
        figtype = defFigType
    # work out sub_plot_name.
    if name is None:
        fig_name = fig.get_label()
    else:
        fig_name = name

    if savedir is None:
        savedir = fig_dir

    # try and create savedir
    # possibly create the fig_dir.
    savedir.mkdir(parents=True, exist_ok=True)  # create the directory

    outFileName = savedir / (fig_name + figtype)
    if verbose:
        print(f"Saving to {outFileName}")
    fig.savefig(outFileName, dpi=dpi)