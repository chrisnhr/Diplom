#%% 
import numpy as np
from scipy.spatial.distance import cosine

#%%
# Define the vectors
v1 = np.array([1, 3, 5, 7])
v2 = np.array([2, 4, 6, 8])

v3 = np.array([5, 7, 1, 3])
v4 = np.array([6, 8, 2, 4])

# Compute cosine dist =  1 - cosine similarity
# cosine distance e [0,2]
cos_dist_1 = 1 - cosine(v1, v2)
cos_dist_2 = 1 - cosine(v3, v4)

# Compare the results
cos_dist_1, cos_dist_2, np.isclose(cos_dist_1, cos_dist_2)

# %%
#wegen curse of dim will ich nicht alle param combis, but die ersten 10 nehmen