"""

Plot reliability diagrams

"""
from importlib import reload
import matplotlib.pyplot as plt
import xarray
import numpy as np
import seaborn as sns
import recalibrate


def comp_reliability(sim,obs, pbin, binsize=None):

    if binsize is None:
        binsize=3
    obs_quant = recalibrate.quantile(obs, binsize=binsize)
    sim_quant = recalibrate.quantile(sim, binsize=binsize)
    # let's get probability of each simulated/forecast value.
    forecast_p = recalibrate.forecast_prob(sim_quant)
    forecast_pq = (xarray.apply_ufunc(np.digitize, forecast_p, kwargs=dict(bins=pbin)))
    # and then the forecast reliability
    return recalibrate.reliability(forecast_pq, obs_quant)


nhd = recalibrate.read_data('nhd')
tas = recalibrate.read_data('tas')
pap = recalibrate.read_data('pap')

reliability = dict()
bins = 4
pbin = np.linspace(0, 1.0, bins, endpoint=False)
delta_p = 1 / (2 * bins)  # offset for centre of bin
for title, data in zip(['TAS', 'NHD', 'PAP'], [tas, nhd, pap]):
    reliability[title] = comp_reliability(data.sim,data.obs, pbin)
    # lets do the bootstrap

## now to plot
fig, axes = plt.subplots(nrows=3, ncols=2, clear=True, sharex='all', sharey='none',
                         figsize=[8, 9], num='reliability')

colors_quant = ['blue', 'grey', 'red']
p_bnds=np.linspace(0,1,bins+1,endpoint=True)
yticks=[]
for indx in range(0,len(p_bnds)-1):
    yticks.append(f'{p_bnds[indx]:3.2f}-{p_bnds[indx+1]:3.2f}')
for ax, ax2, (name, r) in zip(axes[:, 0], axes[:, 1], reliability.items()):
    # noinspection SpellCheckingInspection
    reliab = recalibrate.quantile(r.reliab,binsize=bins)
    # noinspection SpellCheckingInspection
    fcount = r.fCount
    mxfcount = fcount.max()
    for q, col in zip(reliab.quant, colors_quant):
        offset = (q-2)
        (reliab.sel(quant=q)+offset*0.1).plot(linewidth=2, color=col, marker='o', ax=ax, yincrease=True,label=f'Quant: {int(q)}')
        (fcount.sel(quant=q)+offset*mxfcount*0.1/3).plot(linewidth=2, color=col, marker='o', ax=ax2, yincrease=True)
    ax.set_title(f"{name} Reliability")
    ax.set_yticks(range(1,bins+1))
    ax.set_yticklabels(yticks)
    ax.set_ylabel("Fraction Obs")
    ax.plot(ax.get_xlim(),ax.get_xlim(),color='black',linestyle='dashed')
    ax2.set_title(f"{name} Forecast cases")
    ax2.set_ylabel("Forecast Count")
    ax2.set_yticks(np.linspace(0, 35,8,endpoint=True))

    for a in [ax, ax2]:
        a.set_xlabel("Forecast Prob.")
        a.set_xticks(np.arange(1,bins+1))
        a.set_xticklabels(yticks)

axes[0,0].legend()
fig.tight_layout()
fig.show()

recalibrate.saveFig(fig)
