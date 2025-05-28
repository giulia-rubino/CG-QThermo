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
chi = 0.02

Switch = 0

ad = RaisingOp('a')
a = LoweringOp('a')
N = NumberOp('N')

creat = represent(a, ndim=Dim, format='numpy')
annih = represent(ad, ndim=Dim, format='numpy')
N = represent(N, ndim=Dim, format='numpy')

if Switch==0:
	H_0_diag = hbar*omega*(N+np.identity(12)/2)
else:
	H_0 = hbar*omega*((N+np.identity(12)/2)-chi*(N+np.identity(12)/2)**2-chi/4)
	eigenvalues0, eigenvectors0 = np.linalg.eig(H_0)
	R0 = eigenvectors0
	idx0 = eigenvalues0.argsort()
	eigenvalues0 = eigenvalues0[idx0]
	eigenvectors0 = eigenvectors0[:,idx0]
	H_0_diag = np.diag(eigenvalues0)

rho_0 = expm(-beta*H_0_diag)/np.trace(expm(-beta*H_0_diag))

rho_tauTH_CG = np.zeros((Dim,Dim))
rho_0_CG = np.zeros((Dim,Dim))
H_0_CG = np.zeros((Dim,Dim))
H_tau_CG = np.zeros((Dim,Dim))

CGfactor = 3
energies0=np.diag(H_0_diag)
#delta= CGfactor*abs(energies0[1]-energies0[0])
delta= (CGfactor)*abs(energies0[1]-energies0[0])

NSteps0 = int((energies0[Dim-1]-energies0[0])/delta)+1
P0 = np.zeros((NSteps0,Dim,Dim))

for j in range(0,NSteps0):
	for i in range(0,Dim):
		if H_0_diag[i,i] >= energies0[0]+j*delta and H_0_diag[i,i] < energies0[0]+(j+1)*delta:
			P0[j,i,i] = 1


for i in range(NSteps0):
	if int(np.trace(P0[i,:,:]))>0:
		rho_0_CG = rho_0_CG + np.trace(rho_0@P0[i,:,:])/int(np.trace(P0[i,:,:]))*P0[i,:,:]
		E_J0 = - k_B*T*np.log(np.trace(expm(-beta*H_0_diag)@P0[i,:,:])/int(np.trace(P0[i,:,:])))
		H_0_CG = H_0_CG + E_J0*P0[i,:,:]

plt.style.use("classic")
fig = plt.figure(facecolor="white", figsize=(10,8))

df = pd.DataFrame(rho_0_CG)

sns.heatmap(df, cmap="YlGnBu", linewidths=0.5, linecolor='gray', annot=True, annot_kws={"size": 12})

plt.show()
fig.savefig("rho0_CoarseGrained_NEW_Morse.pdf")
