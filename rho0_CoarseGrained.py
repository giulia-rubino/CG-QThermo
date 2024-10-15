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
Dim = 12
beta = 1/(k_B*T)
chi = 0.1

ad = RaisingOp('a')
a = LoweringOp('a')
N = NumberOp('N')

creat = represent(a, ndim=Dim, format='numpy')
annih = represent(ad, ndim=Dim, format='numpy')
N = represent(N, ndim=Dim, format='numpy')


#H_0 = hbar*omega*(N+np.identity(12)/2)
H_0 = hbar*omega*((N+np.identity(12)/2)-chi*(N+np.identity(12)/2)**2-chi/4)
rho_0 = expm(-beta*H_0)/np.trace(expm(-beta*H_0))

CGfactor = 3
Dim1 = int(Dim/CGfactor)

P = np.zeros((Dim1,Dim,Dim))

for i in range(Dim1):
	for j in range(CGfactor):
		P[i,j+i*CGfactor,j+i*CGfactor] = 1


rho_0_CG = np.zeros((Dim,Dim))
for i in range(Dim1):
	rho_0_CG = rho_0_CG + np.trace(rho_0*P[i,:,:])/CGfactor*P[i,:,:]

plt.style.use("classic")
fig = plt.figure(facecolor="white", figsize=(10,8))

df = pd.DataFrame(rho_0_CG)

sns.heatmap(df, cmap="YlGnBu", linewidths=0.5, linecolor='gray', annot=True, annot_kws={"size": 12})

plt.show()
fig.savefig("rho0_CoarseGrained_Morse.pdf")