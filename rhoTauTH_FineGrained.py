import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import expm, sinm, cosm
from sympy import Symbol, Integer
from sympy.physics.quantum import (Dagger,qapply,represent,InnerProduct,Commutator)
from sympy.physics.quantum.sho1d import (RaisingOp,LoweringOp,NumberOp,Hamiltonian,SHOKet,SHOBra)

hbar = 1
omega = 2
f = 4
k_B = 1
T = 10
beta = 1/(k_B*T)

ad = RaisingOp('a')
a = LoweringOp('a')
N = NumberOp('N')

creat = represent(a, ndim=12, format='numpy')
annih = represent(ad, ndim=12, format='numpy')
N = represent(N, ndim=12, format='numpy')

H_tau = hbar*omega*(N+np.identity(12)/2) + f*(creat + annih)/np.sqrt(2)

eigenvalues, eigenvectors = np.linalg.eig(H_tau)
H_tau_diag = np.diag(eigenvalues)

rho_tauTH = expm(-beta*H_tau_diag)/np.trace(expm(-beta*H_tau_diag))

plt.style.use("classic")
fig = plt.figure(facecolor="white", figsize=(10,8))

df = pd.DataFrame(rho_tauTH)

sns.heatmap(df, cmap="YlGnBu", linewidths=0.5, linecolor='gray', annot=True, annot_kws={"size": 12})

plt.show()
fig.savefig("rhoTauTH_FineGrained.pdf")