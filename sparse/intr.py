# %%
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.sparse as sps

# %%
x = np.linspace(0, 1e6, 10)
plt.plot(x, 8.0 * (x **2)/1e6, lw=5)
plt.xlabel('size n')
plt.ylabel('memory [MB]')