import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import pylab

from scipy.linalg import logm, expm, sinm, cosm
from numpy.linalg import inv
from sympy import Symbol, Integer
from sympy.physics.quantum import (Dagger,qapply,represent,InnerProduct,Commutator)
from sympy.physics.quantum.sho1d import (RaisingOp,LoweringOp,NumberOp,Hamiltonian,SHOKet,SHOBra)

def S(rho, sigma):
    return np.trace(rho*(logm(rho)-logm(sigma)))

hbar = 1
omega = 2
k_B = 1
T = 10
Dim = 12
beta = 1/(k_B*T)
Force = 6
chi = 0.1

ad = RaisingOp('a')
a = LoweringOp('a')
N = NumberOp('N')

creat = represent(a, ndim=Dim, format='numpy')
annih = represent(ad, ndim=Dim, format='numpy')
N = represent(N, ndim=Dim, format='numpy')

Iter = 50
RelEnt = np.zeros(Iter)
RelEnt1 = np.zeros((Iter,Iter))
F = np.zeros(Iter)

for k in range(Iter):
	f = k*Force/Iter
	F[k] = k*Force/Iter

	H_0 = hbar*omega*(N+np.identity(Dim)/2)
	#H_0 = hbar*omega*((N+np.identity(12)/2)-chi*(N+np.identity(12)/2)**2-chi/4)
	rho_0 = expm(-beta*H_0)/np.trace(expm(-beta*H_0))

	H_tau = hbar*omega*(N+np.identity(12)/2) + f*(creat + annih)/np.sqrt(2)

	eigenvalues, eigenvectors = np.linalg.eig(H_tau)
	R = eigenvectors
	H_tau_diag = np.diag(eigenvalues)

	rho_tauTH = expm(-beta*H_tau_diag)/np.trace(expm(-beta*H_tau_diag))

	U = expm(-1j*H_tau)
	Udag = expm(1j*H_tau)
	rho_tau = np.real(U@rho_0@Udag)
	rho_tau_Rot = inv(R)@rho_tau@R

	rho_tauTH_CG = np.zeros((Dim,Dim))
	rho_tau_CG = np.zeros((Dim,Dim))

	for l in range(3):
		CGfactor = l + 2
		Dim1 = int(Dim/CGfactor)
		P = np.zeros((Dim1,Dim,Dim))

		for i in range(Dim1):
			for j in range(CGfactor):
				P[i,j+i*CGfactor,j+i*CGfactor] = 1

		for i in range(Dim1):
			rho_tauTH_CG = rho_tauTH_CG + np.trace(rho_tauTH*P[i,:,:])/CGfactor*P[i,:,:]
			rho_tau_CG = rho_tau_CG + np.trace(rho_tau_Rot*P[i,:,:])/CGfactor*P[i,:,:]

		RelEnt1[l,k] = S(rho_tau_CG, rho_tauTH_CG)

	RelEnt[k] = S(rho_tau_Rot, rho_tauTH)

	#print(rho_tau_CG3)
	#print(rho_tauTH_CG3)

fig = plt.figure(facecolor="white", figsize=(12,9))

pylab.plot(F,RelEnt, color="#7695FF", alpha = 0.7, label=r'$S\left(\rho_\tau \vert \vert \rho_\tau^{th} \right)$', linewidth=5)
pylab.plot(F,RelEnt1[0,:], color="#9DBDFF", alpha = 0.7, label=r'$S\left(\breve{\rho}_\tau \vert \vert \breve{\rho}_\tau^{th} \right)$, Tr($P_J$=2)', linewidth=5)
pylab.plot(F,RelEnt1[1,:], color="#FFD7C4", alpha = 0.7, label=r'$S\left(\breve{\rho}_\tau \vert \vert \breve{\rho}_\tau^{th} \right)$, Tr($P_J$=3)', linewidth=5)
pylab.plot(F,RelEnt1[2,:], color="#FF9874", alpha = 0.7, label=r'$S\left(\breve{\rho}_\tau \vert \vert \breve{\rho}_\tau^{th} \right)$, Tr($P_J$=4)', linewidth=5)

#plt.xscale('log')

#plt.xlim([0.1, 40])
#plt.ylim([0, 80])

plt.xlabel(r'Driving force $f$', size=32)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(prop={"size":32})
plt.show()
fig.savefig("RelEntropy.pdf")