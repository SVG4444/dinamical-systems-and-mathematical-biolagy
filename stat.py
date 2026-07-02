import numpy as np
import matplotlib.pyplot as plt

def coupled_system_batch(x0, y0, K, n_iter=400, skip=50):
    n = len(x0)
    x = np.zeros((n, n_iter))
    y = np.zeros((n, n_iter))
    x[:, 0] = x0
    y[:, 0] = y0
    
    for i in range(1, n_iter):
        x[:, i] = r1*x[:, i-1] * (1 - x[:, i-1]) + K * (y[:, i-1] - x[:, i-1])
        y[:, i] = r2*y[:, i-1] * (1 - y[:, i-1]) + K * (x[:, i-1] - y[:, i-1])       
    return x[:, skip:].flatten(), y[:, skip:].flatten()

r1=3.6
r2=3.5
K = 0.13
n_trajectories = 500                  
n_iterations = 500
skip = 50

np.random.seed(500)
x0 = np.random.uniform(0, 1.5, n_trajectories)
y0 = np.random.uniform(0, 1.5, n_trajectories)

X, Y = coupled_system_batch(x0, y0, K, n_iterations, skip)

valid = np.isfinite(X) & np.isfinite(Y) & (X >= 0) & (X <= 1.5) & (Y >=0) & (Y <= 1.5)
X, Y = X[valid], Y[valid]

fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
ax.set_xlim(0, 1)  
ax.set_ylim(0, 1)
ax.set_xlabel('u', fontsize=12)
ax.set_ylabel('v', fontsize=12)
ax.set_title(f'Coupled Logistic Map, $r_1$={r1}, $r_2$={r2}', fontsize=14)
ax.set_xticks(np.arange(0, 1, 0.1))
ax.set_yticks(np.arange(0, 1, 0.1))
ax.grid(True, alpha=0.3)
ax.scatter(X, Y, s=1.2, alpha=0.5, c='navy', edgecolors='none', rasterized=True)
ax.text(0.02, 0.95, f'$\mu$ = {K:.4f}', transform=ax.transAxes,
        fontsize=14, fontweight='bold', color='darkred',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

plt.tight_layout()
plt.show()