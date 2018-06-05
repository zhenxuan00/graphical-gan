import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


def scatter(data, label, dir, file_name, mus=None, mark_size=2):
    if label.ndim == 2:
        label = np.argmax(label, axis=1)

    df = pd.DataFrame(data={'x':data[:,0], 'y':data[:,1], 'class':label})
    sns_plot = sns.lmplot('x', 'y', data=df, hue='class', fit_reg=False, scatter_kws={'s':mark_size})
    sns_plot.savefig(os.path.join(dir, file_name))
    if mus is not None:
        df_mus = pd.DataFrame(data={'x':mus[:,0], 'y':mus[:,1], 'class':np.asarray(xrange(mus.shape[0])).astype(np.int32)})
        sns_plot_mus = sns.lmplot('x', 'y', data=df_mus, hue='class', fit_reg=False, scatter_kws={'s':mark_size*20})
        sns_plot_mus.savefig(os.path.join(dir, 'mus_'+file_name))
        # data = np.vstack((data, mus))
        # label = np.hstack((label, (np.ones(mus.shape[0])*(label.max()+1)).astype(np.int32)))
        # df = pd.DataFrame(data={'x':data[:,0], 'y':data[:,1], 'class':label})
        # sns_plot = sns.lmplot('x', 'y', data=df, hue='class', fit_reg=False, scatter_kws={'s':mark_size})
