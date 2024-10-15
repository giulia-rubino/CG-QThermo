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
f = 6
k_B = 1
T = 10
beta = 1/(k_B*T)
chi = 0.1

b = SHOBra('b')
b0 = SHOBra(0)
b1 = SHOBra(1)

k = SHOKet('k')
k0 = SHOKet(0)
k1 = SHOKet(1)

ad = RaisingOp('a')
a = LoweringOp('a')
N = NumberOp('N')
H = Hamiltonian('H')

creat = represent(a, ndim=12, format='numpy')
annih = represent(ad, ndim=12, format='numpy')
N = represent(N, ndim=12, format='numpy')

#H_0 = hbar*omega*(N+np.identity(12)/2)
H_0 = hbar*omega*((N+np.identity(12)/2)-chi*(N+np.identity(12)/2)**2-chi/4)
rho_0 = expm(-beta*H_0)/np.trace(expm(-beta*H_0))

rho_0[abs(rho_0) < 9*10**(-4)] = 0

plt.style.use("classic")
fig = plt.figure(facecolor="white", figsize=(10,8))

df = pd.DataFrame(rho_0)

sns.heatmap(df, cmap="YlGnBu", linewidths=0.5, linecolor='gray', annot=True, annot_kws={"size": 12})

plt.show()
fig.savefig("rho0_FineGrained_Morse.pdf")