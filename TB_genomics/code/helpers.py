__author__ = "Vy Bui, Ph.D."
__email__ = "01bui@cua.edu / vy.bui@nih.gov"

from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

"""Imshow utility"""
import pylab
from matplotlib import colors
import pandas as pd

# error messages
INVALID_FILETYPE_MSG = "Error: Invalid file format. %s must be a .csv file."
INVALID_PATH_MSG = "Error: Invalid file path/name. Path %s does not exist."
INVALID_INPUT_MSG = "Error: No file path/name provided. Please provide csv path for --rad, --gen, --tbp"

def valid_filetype(file_name):
    """
    Validate file type
    Args:
        path (string): path to csv files.
    Returns:
        file_name.endswith('.csv') (string): check if file is csv
    """
    print(file_name)
    return file_name.endswith('.csv')

def valid_path(path):
    """
    Validate file path
    Args:
        path (string): path to csv files.
    Returns:
        os.path.exists(path) (string): check if path exists
    """
    return os.path.exists(path)

def validate_file(file_name):
    """
    Validate file name and path.
    Args:
        file_name (string): path to csv files.
    Returns:
        None
    """
    if not valid_path(file_name):
        print(INVALID_PATH_MSG % (file_name))
        quit()
    elif not valid_filetype(file_name):
        print(INVALID_FILETYPE_MSG % (file_name))
        quit()
    return

def correlation(df, out_folder):
    """
    Compute pairwise correlation of columns in pandas frame, excluding NA/null values.
    Args:
        df (pandas.DataFrame): pandas dataframe that includes
                        the features to calculate correlation
        out_folder (string): path of folder to output the results
    Returns:
        None
    """
    plt.rcParams['font.size'] = 9
    corr = df.corr(method='pearson')#, min_periods=10)
    corr = corr.where(np.tril(np.ones(corr.shape)).astype(bool))
    im = Imshow(corr)
    im.plot(cmap='Dark2')
    plt.savefig(out_folder + 'corr_pearson.jpg', dpi=500)
    corr = corr.round(2).astype(str)
    corr.to_csv(out_folder + '/corr_pearson.csv', index=True)

def chi2(df, TOR, out_folder):
    """
    This function computes the chi-square statistic and p-value for the hypothesis test
    of independence of the observed frequencies in the contingency table observed.
    Args:
        df (pandas.DataFrame): pandas dataframe that includes
                        the features to calculate chi2 test
        TOR (string): a variable contain the type of resistance column name
        out_folder (string): path of folder to output the results
    Returns:
        None
    """
    pval_chi2 = []
    for feature in df.columns:
        contigency = pd.crosstab(df[TOR], df[feature])
        c, p, dof, expected = chi2_contingency(contigency)
        pval_chi2.append(p)
    pval_chi2_df = pd.DataFrame({'features': df.columns, 'p-value': pval_chi2})
    pval_chi2_df.loc[(pval_chi2_df['p-value'] < 0.05), 'Sig Test'] = 'Y'
    pval_chi2_df.to_csv(out_folder + '/pval_chi2' + '.csv', index=True)

__all__ = ['imshow', 'Imshow']

class VizInputSquare(object):
    def __init__(self, x, verbose=False):
        self.verbose = verbose
        self.df = pd.DataFrame(x)

class Imshow(VizInputSquare):
    """Wrapper around the matplotlib.imshow function

    Very similar to matplotlib but set interpolation to None, and aspect
    to automatic and accepts input as a dataframe, in whic case
    x and y labels are set automatically.


    .. plot::
        :width: 80%
        :include-source:

        import pandas as pd
        data = dict([ (letter,np.random.randn(10)) for letter in 'ABCDEFGHIJK'])
        df = pd.DataFrame(data)

        from biokit import Imshow
        im = Imshow(df)
        im.plot()

    """

    def __init__(self, x, verbose=True):
        """.. rubric:: constructor

        :param x: input dataframe (or numpy matrix/array).
            Must be squared.

        """
        super(Imshow, self).__init__(x, verbose=verbose)

    def plot(self, interpolation='None', aspect='auto', cmap='hot', tight_layout=True,
             colorbar=True, fontsize_x=None, fontsize_y=None, rotation_x=90,
             xticks_on=True, yticks_on=True, **kargs):
        """wrapper around imshow to plot a dataframe

        :param interpolation: set to None
        :param aspect: set to 'auto'
        :param cmap: colormap to be used.
        :param tight_layout:
        :param colorbar: add a colobar (default to True)
        :param fontsize_x: fontsize on xlabels
        :param fontsize_y: fontsize on ylabels
        :param rotation_x: rotate labels on xaxis
        :param xticks_on: switch off the xticks and labels
        :param yticks_on: switch off the yticks and labels

        """
        fontsize = 4
        data = self.df
        pylab.clf()
        cmap = colors.ListedColormap(
            ['#1b9e77', '#d95f02', '#7570b3', 'white', 'white', '#e7298a', '#e6ab02', '#66a61e'])
        bounds = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        pylab.imshow(data, interpolation='nearest', aspect=aspect, cmap=cmap, norm=norm, **kargs)

        if fontsize_x == None:
            fontsize_x = fontsize
        if fontsize_y == None:
            fontsize_y = fontsize

        if yticks_on is True:
            pylab.yticks(range(0, len(data.index)), data.index,
                         fontsize=fontsize_y)
        else:
            pylab.yticks([])
        if xticks_on is True:
            pylab.xticks(range(0, len(data.columns[:])), data.columns, fontsize=fontsize_x, rotation=90)
        else:
            pylab.xticks([])

        if colorbar is True:
            pylab.colorbar()

        if tight_layout:
            pylab.tight_layout()


def imshow(x, interpolation='None', aspect='auto', cmap='hot', tight_layout=True,
           colorbar=True, fontsize_x=None, fontsize_y=None, rotation_x=90,
           xticks_on=True, yticks_on=True, **kargs):
    """Alias to the class :class:`~biokit.viz.imshow.Imshow`"""

    print('Deprecated. Use Imshow instead')
    i = Imshow(x)
    i.plot(interpolation=interpolation, aspect=aspect, cmap=cmap,
           colorbar=colorbar, fontsize_x=fontsize_x, fontsize_y=fontsize_y,
           xticks_on=xticks_on, yticks_on=yticks_on, **kargs)
