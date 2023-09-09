import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def readPd(filename):
    df = pd.read_csv(filename)
    # Read as scientific notation
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

# # Get files starting with L
# files = [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith('C')]
# print(files)

files = ['C_10_2.7K.csv', 'C_3.8K_1M.csv', 'C_37_25K.csv']
cutFs = [[10, 2.7e3],[800, 1e6],[37, 25e3],]

# Read all files
dfList = []
for f,f0f1 in zip(files, cutFs):
    df = readPd(f)
    df = df[(df['Frec[Hz]'] >= f0f1[0]) & (df['Frec[Hz]'] <= f0f1[1])]
    dfList.append(df)

# colName = "Cs[F]"
colName = "Rp[Ohm]"
x_axis = "Frec[Hz]"

# plot each file |Z|
for df, file in zip(dfList, files):
    y = df[colName]

    if file == 'C_10_2.7K.csv':
        y = y * 35
        pass
    if file == 'C_37_25K.csv':
        y = y / 35
        pass
    if file == 'C_3.8K_1M.csv':
        # apply moving average
        y = y.rolling(2).mean()
        y = y / 50
        pass
    
    plt.loglog(df[x_axis], y)

plt.gca().xaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=400))
plt.gca().xaxis.set_major_locator(plt.LogLocator(base=10, numticks=100))
plt.legend(files)
plt.grid(True, which="both")
plt.show()