import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

arr0 = np.zeros((100, 100))
arr100 = np.full((100, 100), 100)
arr200 = np.full((100, 100), 200)

stacked = np.hstack((arr0, arr100, arr200))

image_with_noise = stacked + 50 * (np.random.rand(100, 300) - 0.5)

image_filtered = gaussian_filter(image_with_noise, 5)

fig, axs = plt.subplots(3)
axs[0].imshow(stacked, cmap="gray")
axs[1].imshow(image_with_noise, cmap="gray")
axs[2].imshow(image_filtered, cmap="gray")
plt.show()

