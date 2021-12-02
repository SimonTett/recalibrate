###############################################################
# Name: summary_plot.py
# Description: Summary of effects of Bellprat calibration from Simon's analysis
# Author: Sarah Sparrow
# Date: 2 Dec 2021
################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


data_cat=['raw','calibrated','calibrated no trend']
variables=['TAS','NHD','PAP']
cols=['RoyalBlue','SpringGreen','Gold']

data = pd.read_csv('Simon_data.csv',index_col=0)

data_cat=['raw','calibrated','calibrated no trend']
variables=['TAS','NHD','PAP']

print(data)
print(data.columns)
 # Set up the plot
font = {'family' : 'sans-serif',
        'size'   : 20}

matplotlib.rc('font', **font)
fig = plt.figure()
fig.set_size_inches(10,6)

ax = plt.subplot2grid((1,1),(0,0))
plt.title('PR summary')

ax.set_ylabel('Probability Ratio',fontsize=16)
plt.setp(ax.get_xticklabels(),fontsize=16)
plt.setp(ax.get_yticklabels(),fontsize=16)


#l1=ax.semilogy(, basey=2

for iv,v in enumerate(variables):
    for c,cat in enumerate(data_cat):
        vals=data.loc[[v+' '+cat]]
        print(vals['Best estimate'][0])
        print(cols[c])
        ax.semilogy([(iv)*5+c+1.25,(iv)*5+c+0.75],[vals['Best estimate'][0],vals['Best estimate'][0]],basey=10,color=cols[c],lw=2,zorder=2)
        ax.fill_between([(iv)*5+c+1.25,(iv)*5+c+0.75],[vals['5%'][0],vals['5%'][0]], [vals['95%'][0],vals['95%'][0]],color=cols[c],alpha=0.3,zorder=1)
ax.semilogy([0,15],[1,1],basey=10,color='silver',ls='--',zorder=0)
ax.set_xlim(0,15)
plt.xticks([2,7,12], variables)  # Set text labels.
ax.legend(data_cat,fontsize="small")
plt.tight_layout()
fig.savefig('summary_plot.png')
print('Finished!')
