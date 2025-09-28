import numpy as np

n = 3
vec = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
psi_nq = vec / np.linalg.norm(vec)

print(psi_nq)
