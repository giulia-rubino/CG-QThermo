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
    return np.trace(rho@(logm(rho)-logm(sigma)))

hbar = 1
omega = 2
k_B = 1
T = 10
Dim = 12
beta = 1/(k_B*T)
Force = 5
chi = 0.09

Switch = 0

ad = RaisingOp('a')
a = LoweringOp('a')
N = NumberOp('N')

creat = represent(a, ndim=Dim, format='numpy')
annih = represent(ad, ndim=Dim, format='numpy')
N = represent(N, ndim=Dim, format='numpy')

Iter = 50
#Iter = 1
RelEnt = np.zeros(Iter)
RelEnt1 = np.zeros((Iter,Iter))
F = np.zeros(Iter)

for k in range(Iter):
	f = k*Force/Iter
	F[k] = k*Force/Iter

	if Switch==0:
		H_0_diag = hbar*omega*(N+np.identity(12)/2)
		H_tau = hbar*omega*(N+np.identity(12)/2) + f*(creat + annih)/np.sqrt(2)
	else:
		H_0 = hbar*omega*((N+np.identity(12)/2)-chi*(N+np.identity(12)/2)**2-chi/4)
		eigenvalues0, eigenvectors0 = np.linalg.eig(H_0)
		R0 = eigenvectors0
		idx0 = eigenvalues0.argsort()
		eigenvalues0 = eigenvalues0[idx0]
		eigenvectors0 = eigenvectors0[:,idx0]
		H_0_diag = np.diag(eigenvalues0)

		H_tau = hbar*omega*((N+np.identity(12)/2)-chi*(N+np.identity(12)/2)**2-chi/4) + f*(creat + annih)/np.sqrt(2)

	rho_0 = expm(-beta*H_0_diag)/np.trace(expm(-beta*H_0_diag))

	eigenvaluesTAU, eigenvectorsTAU = np.linalg.eig(H_tau)
	RTAU = eigenvectorsTAU
	idxTAU = eigenvaluesTAU.argsort()
	eigenvaluesTAU = eigenvaluesTAU[idxTAU]
	eigenvectorsTAU = eigenvectorsTAU[:,idxTAU]
	H_tau_diag = np.diag(eigenvaluesTAU)

	rho_tauTH = expm(-beta*H_tau)/np.trace(expm(-beta*H_tau))

	U = expm(-1j*H_tau)
	Udag = expm(1j*H_tau)
	rho_tau = U@rho_0@Udag

	for l in range(3):
#	for l in range(1):
		rho_tauTH_CG = np.zeros((Dim,Dim))
		rho_0_CG = np.zeros((Dim,Dim))
		H_0_CG = np.zeros((Dim,Dim))
		H_tau_CG = np.zeros((Dim,Dim))

		CGfactor = l + 2
		energies0=np.diag(H_0_diag)
		energiesTAU=np.diag(H_tau_diag)
		#delta= CGfactor*abs(energies0[1]-energies0[0])
		delta= (CGfactor+0.3)*abs(energies0[1]-energies0[0])

		NSteps0 = int((energies0[Dim-1]-energies0[0])/delta)+1
		NStepsTAU = int((energiesTAU[Dim-1]-energiesTAU[0])/delta)+1
		P0 = np.zeros((NSteps0,Dim,Dim))
		PTAU = np.zeros((NStepsTAU,Dim,Dim))

		for j in range(0,NSteps0):
			for i in range(0,Dim):
				if H_0_diag[i,i] >= energies0[0]+j*delta and H_0_diag[i,i] < energies0[0]+(j+1)*delta:
					P0[j,i,i] = 1

		for j in range(0,NStepsTAU):
			for i in range(0,Dim):
				if H_tau_diag[i,i] >= energiesTAU[0]+j*delta and H_tau_diag[i,i] < energiesTAU[0]+(j+1)*delta:
					PTAU[j,i,i] = 1

		for i in range(NSteps0):
			if int(np.trace(P0[i,:,:]))>0:
				rho_0_CG = rho_0_CG + np.trace(rho_0@P0[i,:,:])/int(np.trace(P0[i,:,:]))*P0[i,:,:]
				E_J0 = - k_B*T*np.log(np.trace(expm(-beta*H_0_diag)@P0[i,:,:])/int(np.trace(P0[i,:,:])))
				H_0_CG = H_0_CG + E_J0*P0[i,:,:]

		for i in range(NStepsTAU):
			if int(np.trace(PTAU[i,:,:]))>0:
				rho_tauTH_CG = rho_tauTH_CG + np.trace(rho_tauTH@RTAU@PTAU[i,:,:]@inv(RTAU))/int(np.trace(PTAU[i,:,:]))*RTAU@PTAU[i,:,:]@inv(RTAU)
				E_JTAU = - k_B*T*np.log(np.trace(expm(-beta*H_tau)@RTAU@PTAU[i,:,:]@inv(RTAU))/int(np.trace(PTAU[i,:,:])))
				H_tau_CG = H_tau_CG + E_JTAU*RTAU@PTAU[i,:,:]@inv(RTAU)
				rho_tau_CG = U@rho_0_CG@Udag

		RelEnt1[l,k] = S(rho_tau_CG, rho_tauTH_CG)

	RelEnt[k] = S(rho_tau, rho_tauTH)

fig = plt.figure(facecolor="white", figsize=(12,9))

pylab.plot(F,RelEnt, color="#7695FF", alpha = 0.7, label=r'$S\left(\rho_\tau \vert \vert \rho_\tau^{th} \right)$', linewidth=5)
pylab.plot(F,RelEnt1[0,:], color="#9DBDFF", alpha = 0.7, label=r'$S\left(\breve{\rho}_\tau \vert \vert \breve{\rho}_\tau^{th} \right)$, Tr($P_J$=2)', linewidth=5, marker = 'o')
pylab.plot(F,RelEnt1[1,:], color="#FFD7C4", alpha = 0.7, label=r'$S\left(\breve{\rho}_\tau \vert \vert \breve{\rho}_\tau^{th} \right)$, Tr($P_J$=3)', linewidth=5, marker = 'o')
pylab.plot(F,RelEnt1[2,:], color="#FF9874", alpha = 0.7, label=r'$S\left(\breve{\rho}_\tau \vert \vert \breve{\rho}_\tau^{th} \right)$, Tr($P_J$=4)', linewidth=5, marker = 'o')

plt.xlabel(r'Driving force $f$', size=32)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.legend(prop={"size":32})
plt.show()
fig.savefig("RelEntropy.pdf")