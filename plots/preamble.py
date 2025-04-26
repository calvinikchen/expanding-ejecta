import numpy as np
# import pandas as pd
import seaborn as sns

import operator
from functools import reduce
from math import factorial

import scipy as sp
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
from scipy.spatial.transform import Rotation as R
from scipy.optimize import fmin


import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm, tqdm_notebook
import random
from time import time as tictoc
from matplotlib.animation import FuncAnimation


# from IPython.core.display import display, HTML
# from IPython.display import display, clear_output
# display(HTML("<style>.container { width:100% !important; }</style>"))
# np.set_printoptions(edgeitems=3, linewidth=200) 
#pd.set_option('display.max_columns', None)
# pd.set_option('max_rows',200) and pandas.set_option('max_columns',20)

plt.rcdefaults()
fontsize = 12
from matplotlib import font_manager
from matplotlib import rcParams
from matplotlib import rc
rcParams['font.family'] = 'serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':fontsize})
rc('text', usetex=True)
#rc('text.latex', preamble=r'\usepackage{boldsymbol}')