from importlib import reload
import matplotlib.pyplot as plt
import xarray
import numpy as np
import seaborn as sns
import scipy.stats
import recalibrate
import pandas as pd
import pickle

# load up the data.

prob_ratios = pd.read_csv(recalibrate.gen_data / 'prob_ratios.csv', index_col=0, header=0)
probs = pd.read_csv(recalibrate.gen_data / 'probs.csv', index_col=0, header=0)
obs_2020 = pd.read_csv(recalibrate.gen_data / 'obs_2020.csv', index_col=0, header=0, squeeze=True)
prob_ratios_bs = pd.read_csv(recalibrate.gen_data / 'prob_ratios_bs.csv', index_col=[0, 1], header=0)

with open(recalibrate.gen_data / 'fit_dists.pickle', mode='rb') as fp:
    fit_dists = pickle.load(fp)

with open(recalibrate.gen_data / 'fit_dists_bs.pickle', mode='rb') as fp:
    fit_dists_bs = pickle.load(fp)

# breakpoint()
## print out PRs, liklihoods and the plot the data
bs_uncert_method = 'distribution'
data_pr = dict()
for key, pr in prob_ratios.items():
    data = prob_ratios_bs.loc[pd.IndexSlice[key, :], :]
    titles = data.columns
    uncerts, ks = recalibrate.prob_rat_uncert(data.values, method=bs_uncert_method, pr_est=pr)

    # cases where ks < 0.1 then use the std bootstrap...
    if (ks is not None) and np.any(ks < 0.1):
        dd, ks2 = recalibrate.prob_rat_uncert(np.log(data.values), method='percentile', pr_est=pr)
        uncerts[:, ks < 0.1] = np.exp(dd[:, ks < 0.1])

    ks = pd.Series(ks, index=titles)

    summary = pd.DataFrame(uncerts, index=['5%', '95%'], columns=titles)
    summary = summary.append(pr.rename('Best Estimate'))
    data_pr[key] = summary
    for name, value in pr.items():
        print(f"{key} PR {name}: {value:7.2g} ({summary.loc['5%', name]:7.2g} - {summary.loc['95%', name]:7.2g})",
              end=' ')
        print(f"({summary.loc['5%', name] / value:5.2g} - {summary.loc['95%', name] / value:5.2g})", end=' ')
        if ks is not None:
            print(f"ks:{ks.loc[name]:3.2f}")
        else:
            print("")  # flush the output line
#
data_pr = pd.concat(data_pr)
for key, prob in probs.items():
    # now print out the likelihood ratios

    print(
        f"{key} Raw: {prob.loc['Hist']:7.2g} Likelihood ratio  Calib: {prob.loc['Hist'] / prob.loc['Hist_calib']:3.1f}  Calib -- no trend:{prob.loc['Hist'] / prob.loc['Hist_calib_notrend']:3.1f} )")

fig, axes = plt.subplots(nrows=3, ncols=1, num='Dists', figsize=[9, 9], clear=True)
for ax, (key, fit_dist) in zip(axes, fit_dists.items()):
    if key == 'PAP':  # interested in drought...
        ylabel = 'p < threshold'
    else:
        ylabel = 'p > threshold'

    for (label, dist), color, linestyle in zip(fit_dist.items(),
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
    ax.axhline(1e-2, linewidth=2, color='grey', linestyle='dashed')
    ax.set_yscale('log')
    ax.set_title(key)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Threshold")

axes[0].legend()
fig.tight_layout()
fig.show()

recalibrate.saveFig(fig)

## plot the PRs
# code from Sarah Sparrow witha  bti of editing to make it work with bootstrapped data

# Get variables and plot colours
variables = data_pr.index.get_level_values(0).unique()
cats = data_pr.columns
cols = ['RoyalBlue', 'SpringGreen', 'Gold']
fig = plt.figure(num='PR_plot', figsize=[8, 6])
ax = plt.subplot2grid((1, 1), (0, 0))
plt.title('PR summary')

ax.set_ylabel('Probability Ratio')

# Plot the best estimate and 5-95% range for each variable for all  categories
for iv, v in enumerate(variables):
    for c, cat in enumerate(cats):
        vals = data_pr.loc[pd.IndexSlice[v, :], cat].droplevel(0)
        ax.semilogy([(iv) * 5 + c + 1.25, (iv) * 5 + c + 0.75], [vals['Best Estimate'], vals['Best Estimate']],
                    base=10, color=cols[c], lw=2, zorder=2)
        ax.fill_between([(iv) * 5 + c + 1.25, (iv) * 5 + c + 0.75], [vals['5%'], vals['5%']],
                        [vals['95%'], vals['95%']], color=cols[c], alpha=0.3, zorder=1)

# Plot the PR=1 line
ax.semilogy([0, 15], [1, 1], base=10, color='silver', ls='--', zorder=0)

# Set the x axis range and tick labels
ax.set_xlim(0, 15)
plt.xticks([2, 7, 12], variables)  # Set text labels.

# Plot the legend and save the figure
ax.legend(data_pr.columns)
plt.tight_layout()
recalibrate.saveFig(fig)