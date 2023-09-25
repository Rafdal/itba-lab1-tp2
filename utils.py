import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plotBodeMC(frec, loglog=[], semilog=[]):
    # check if plot_data is a list
    if not isinstance(loglog, list):
        loglog = [loglog]
    if not isinstance(semilog, list):
        semilog = [semilog]

    # Create a figure and two axes
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    cmap =  plt.cm.viridis(np.linspace(0, 1, len(frec)))
    ax1.set_prop_cycle('color', cmap)
    ax2.set_prop_cycle('color', cmap)

    for y1 in loglog:
        ax1.loglog(frec, y1, lw=1.5, linestyle='-', alpha=0.8, label='LogLog')
    for y2 in semilog:
        ax2.semilogx(frec, y2, lw=1.5, linestyle='--', alpha=0.5, label='Semilog')

    ax1.set_ylabel('LogLog', color='black')
    ax2.set_ylabel('Semilog', color='black')

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, which="both", ls="-", axis="x")
    ax1.grid(True, which="both", ls="-", axis="y")

    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(True, which="major", ls="-")

    ax1.xaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=400))
    ax1.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=100))

    # set ticks for ax2 y axis
    ax2.yaxis.set_major_locator(plt.MultipleLocator(15))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(5))
    # ax1.yaxis.set_major_locator(plt.MultipleLocator(20))

    ax1.set_xlabel('Frecuencia $[Hz]$')

    # set legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')


    # set axis limits
    ax1.set_xlim(left=min(frec), right=max(frec))
    plt.show()




def readPd(filename):
    df = pd.read_csv(filename)
    # Read as scientific notation
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


# define the true objective function
def parallelZ(A, B):
	return (A*B) / (A + B)

def parallelABC(A, B, C):
    return (A*B*C) / (A*B + A*C + B*C)