import plotly
import plotly.figure_factory as ff

import numpy as np
import pandas as pd

dataframe = pd.read_csv("scores.csv")

fig = ff.create_scatterplotmatrix(dataframe, height=800, width=800)
plotly.offline.plot(fig, filename='scores-scatter')
