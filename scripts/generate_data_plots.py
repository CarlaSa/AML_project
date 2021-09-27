import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc, cm, colors

# latex text in plots
usetex = True
if usetex:
    rc('text', usetex=True)
    autopct = "%.1f\,\%%"
else:
    autopct = "%.1f%%"
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


def bar_hist(data, save=None):
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
    if save is not None:
        plt.savefig(os.path.join("images/data", save), bbox_inches='tight')


def bar(data, save=None, to_int=False):
    ax = data.plot.bar()
    rects = ax.containers[0]
    autolabel(ax, rects)
    for direction in ['top', 'right']:
        ax.spines[direction].set_visible(False)
    if to_int is True:
        labels = ax.get_xticklabels()
        for text in labels:
            text.set_text(f"${int(float(text.get_text()))}$")
        ax.set_xticklabels(labels)
    if save is not None:
        plt.savefig(os.path.join("images/data", save), bbox_inches='tight')
    return ax


def pie(data, save=None, **kwargs):
    ax = data.plot.pie(ylabel="", autopct=autopct, labeldistance=None,
                       startangle=90, radius=1.2, legend=True, **kwargs)
    ax.legend(loc="center left", bbox_to_anchor=(1.25, 0, 0.5, 1))
    if save is not None:
        plt.savefig(os.path.join("images/data", save), bbox_inches='tight')


with open("artifacts/dicom_keywords.json") as f:
    keywords = json.load(f)


stat_table = pd.read_csv("artifacts/dicom_meta.csv")
stat_table.columns
stat_table.describe()

bar_hist(stat_table.n_boxes, save="n_boxes.pdf")
plt.hist((stat_table.pixel_max - stat_table.pixel_min), bins=24)
cmap = colors.ListedColormap(cm.Paired(range(12))[:8])
pie(stat_table[["study_label", "PatientSex"]].value_counts(), "balance.pdf",
    colormap=cmap, pctdistance=1.25)
ax = pie(stat_table.study_label.value_counts())
ax = bar(stat_table.BitsStored.value_counts().sort_index(), to_int=True,
         save="bit_depth.pdf")
text = ax.get_xticklabels()
text.set_text()
