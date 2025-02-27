#%%
from modules import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import convolve1d, gaussian_filter1d

def kernel_smoothing_uniform(series, kernel_size):
    """
    Applies kernel smoothing to a given time series.
    
    Parameters:
    series (array-like): Input time series.
    kernel_size (int): Size of the smoothing kernel.
    
    Returns:
    np.ndarray: Smoothed time series.
    """
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = convolve1d(series, kernel, mode='nearest')
    #correction factor um summe nicht zu verfälschen ist auf der 3-4 kommastelle != 1
    return smoothed * (series.sum() / smoothed.sum())

def kernel_smoothing_gaussian(series, kernel_size, sigma=None):
    if sigma is None:
        sigma = kernel_size / 3  # Rule of thumb: sigma ~ kernel_size / 3
    smoothed = gaussian_filter1d(series, sigma, mode='nearest')
    return smoothed * (series.sum() / smoothed.sum())  # Correction factor
#%%
# Load input data
input_data = InputData("ANSPRACHE_MARKETING_IMPUTED", 10)
test_dataframe = input_data.TwinData[735286989]
#716679764, 735286989, 796628327, 803656013, 804245827, 807444298, 811224241, 811350832, 811405342, 813036796, 815378026, 821340660, 821851364, 824008632, 824014002, 824093461, 829628090, 830443285, 831771058, 831990789, 83279335

#%%
# Plot original time series
plt.figure(figsize=(15, 5))
for column in test_dataframe.columns:
    plt.plot(test_dataframe[column], label=column)

plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Limit number of x-ticks
plt.title("Original Time Series")
plt.show()

# Apply smoothing and plot
plt.figure(figsize=(15, 5))
for column in test_dataframe.columns:
    smoothed_series = kernel_smoothing_gaussian(test_dataframe[column], kernel_size=7)
    plt.plot(smoothed_series, label=column)

plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
plt.title("Smoothed Time Series")
plt.show()
# %%
