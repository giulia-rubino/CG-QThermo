import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import expm, sinm, cosm
from scipy.sparse.linalg import cg
from sympy import Symbol, Integer
from sympy.physics.quantum import (Dagger,qapply,represent,InnerProduct,Commutator)
from sympy.physics.quantum.sho1d import (RaisingOp,LoweringOp,NumberOp,Hamiltonian,SHOKet,SHOBra)

hbar = 1
omega = 2
k_B = 1
T = 10
Dim = 12
beta = 1/(k_B*T)
f = 4
chi = 0.1

ad = RaisingOp('a')
a = LoweringOp('a')
N = NumberOp('N')

creat = represent(a, ndim=12, format='numpy')
annih = represent(ad, ndim=12, format='numpy')
N = represent(N, ndim=12, format='numpy')

H_0 = hbar*omega*((N+np.identity(12)/2)-chi*(N+np.identity(12)/2)**2-chi/4)
rho_0 = expm(-beta*H_0)/np.trace(expm(-beta*H_0))

H_tau = hbar*omega*(N+np.identity(12)/2) + f*(creat + annih)/np.sqrt(2)

U = expm(-1j*H_tau)
Udag = expm(1j*H_tau)
rho_tau = np.real(U@rho_0@Udag)

plt.style.use("classic")
fig = plt.figure(facecolor="white", figsize=(10,8))

rho_tau[abs(rho_tau) < 5*10**(-3)] = 0

df = pd.DataFrame(rho_tau)

sns.heatmap(df, cmap="YlGnBu", linewidths=0.5, linecolor='gray', annot=True, annot_kws={"size": 12})

plt.show()
fig.savefig("rhoTau_FineGrained.pdf")