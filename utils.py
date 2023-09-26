import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PlotBode:
    def __init__(self, figsize=(10, 6)):
        self.fig, self.ax1 = plt.subplots(figsize=figsize)
        self.ax2 = self.ax1.twinx()

        self.min_f = None
        self.max_f = None

    def __set_f_lim__(self, f):
        self.min_f = np.min([self.min_f, np.min(f)]) if self.min_f else np.min(f)
        self.max_f = np.max([self.max_f, np.max(f)]) if self.max_f else np.max(f)

    def plotSemilog1(self, f, y, **kwargs):
        self.ax1.semilogx(f, y, **kwargs)
        self.__set_f_lim__(f)

    def plotSemilog2(self, f, y, **kwargs):
        self.ax2.semilogx(f, y, **kwargs)
        self.__set_f_lim__(f)
        
    def plotLoglog1(self, f, y, **kwargs):
        self.ax1.loglog(f, y, **kwargs)
        self.__set_f_lim__(f)

    def plotLoglog2(self, f, y, **kwargs):
        self.ax2.loglog(f, y, **kwargs)
        self.__set_f_lim__(f)

    def show(self, loc='best', min2loc=2, maj2loc=15):
        self.ax1.tick_params(axis='y', labelcolor='black')
        self.ax1.grid(True, which="both", ls="-", axis="x")
        self.ax1.grid(True, which="both", ls="-", axis="y")

        self.ax2.tick_params(axis='y', labelcolor='black')
        self.ax2.grid(True, which="major", ls="-")

        self.ax1.xaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=400))
        self.ax1.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=100))

        # set ticks for ax2 y axis
        self.ax2.yaxis.set_major_locator(plt.MultipleLocator(maj2loc))
        self.ax2.yaxis.set_minor_locator(plt.MultipleLocator(min2loc))
        # self.ax1.yaxis.set_major_locator(plt.MultipleLocator(20))

        self.ax1.set_ylabel('Modulo $|Z|$')
        self.ax2.set_ylabel('Fase $Z \degree$')

        # set legend
        lines, labels = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax2.legend(lines + lines2, labels + labels2, loc=loc)

        self.ax1.set_xlabel('Frecuencia $[Hz]$')

        # set axis limits
        self.ax1.set_xlim(left=self.min_f, right=self.max_f)
        plt.tight_layout()
        plt.show()

def calcR2(y, y_pred):
    y_mean = np.mean(y)
    SS_tot = np.sum((y - y_mean)**2)
    SS_res = np.sum((y - y_pred)**2)
    return 1 - (SS_res / SS_tot)

def getAxes(frec=[], loglog=[], semilog=[]):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for y1 in loglog:
        ax1.loglog(frec, y1, lw=1.5, linestyle='-', alpha=0.8, label='LogLog')
    for y2 in semilog:
        ax2.semilogx(frec, y2, lw=1.5, linestyle='--', alpha=0.5, label='Semilog')

    ax1.set_ylabel('LogLog', color='black')
    ax2.set_ylabel('Semilog', color='black')

    


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