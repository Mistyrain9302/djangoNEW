import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fontPath = "./static/Fonts/NIAGENG.TTF"
fontName = fm.FontProperties(fname=fontPath, size=12).get_name()
plt.rc("font", family=fontName)
mpl.rcParams["axes.unicode_minus"] = False
from django_plotly_dash import DjangoDash
import warnings; warnings.filterwarnings('ignore')

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost
import lightgbm
import os 

datafile_name = "final_pivot_df48.xlsx"
PATH = os.path.dirname(os.path.abspath(__file__)) + '/' +datafile_name
data = pd.read_excel(PATH)
print(f"scores.py 갱신중")