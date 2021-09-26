import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc

# latex text in plots
usetex = True
if usetex:
    rc('text', usetex=True)

plt.rcParams.update({'font.size': 18,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16})


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"$\scriptstyle{{{height}}}$",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def bar_hist(data):
    bincount = np.bincount(data)
    x = np.arange(len(bincount))
    fig, ax = plt.subplots()
    rects = ax.bar(x, bincount)
    ax.set_xticks(x)
    autolabel(ax, rects)
    ax.set_ylim((0, np.max(bincount) * 1.1))
    for direction in ['top', 'right']:
        ax.spines[direction].set_visible(False)
    fig.tight_layout()


with open("artifacts/dicom_keywords.json") as f:
    keywords = json.load(f)


stat_table = pd.read_csv("artifacts/_dicom_meta.csv")
stat_table.columns
stat_table.describe()

bar_hist(stat_table.n_boxes)
plt.hist(stat_table.pixel_max - stat_table.pixel_min, bins=24)
plt.pie(stat_table.PatientSex)
