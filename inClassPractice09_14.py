import numpy as np
p = np.array([0.25, 0.25, 0.25, 0.25])
entropy = sum(-p * np.log2(p))
print(entropy)