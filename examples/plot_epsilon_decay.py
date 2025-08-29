import matplotlib.pyplot as plt
import numpy as np

start_e = 1.0
end_e = 0.05
decay_e = 60000
steps_done = np.arange(0, 1000000)
eps_threshold = end_e + (start_e - end_e) * np.exp(-1.0 * steps_done / decay_e)

plt.plot(steps_done, eps_threshold)
plt.show()
