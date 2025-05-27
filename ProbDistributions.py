import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pylab

from scipy.linalg import logm, expm, sinm, cosm
from numpy.linalg import inv
from sympy import Symbol, Integer
from sympy.physics.quantum import (Dagger,qapply,represent,InnerProduct,Commutator)
from sympy.physics.quantum.sho1d import (RaisingOp,LoweringOp,NumberOp,Hamiltonian,SHOKet,SHOBra)
from scipy import optimize

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

hbar = 1
omega = 2
k_B = 1
T = 4
Dim = 24
beta = 1/(k_B*T)
f = 3.8

ad = RaisingOp('a')
a = LoweringOp('a')
N = NumberOp('N')

creat = represent(a, ndim=Dim, format='numpy')
annih = represent(ad, ndim=Dim, format='numpy')
N = represent(N, ndim=Dim, format='numpy')

Iter = 50

H_0_diag = hbar*omega*(N+np.identity(Dim)/2)
rho_0 = expm(-beta*H_0_diag)/np.trace(expm(-beta*H_0_diag))

H_tau = hbar*omega*(N+np.identity(Dim)/2) + f*(creat + annih)/np.sqrt(2)

eigenvaluesTAU, eigenvectorsTAU = np.linalg.eig(H_tau)
RTAU = eigenvectorsTAU
idxTAU = eigenvaluesTAU.argsort()
eigenvaluesTAU = eigenvaluesTAU[idxTAU]
eigenvectorsTAU = eigenvectorsTAU[:,idxTAU]
H_tau_diag = np.diag(eigenvaluesTAU)

rho_tauTH = expm(-beta*H_tau_diag)/np.trace(expm(-beta*H_tau_diag))

FE = -np.log(np.trace(expm(-beta*H_0_diag)))/beta
FE1 = -np.log(np.trace(expm(-beta*H_tau)))/beta
DeltaF = FE1 - FE

U = expm(-1j*H_tau)
Udag = expm(1j*H_tau)
rho_tau = U@rho_0@Udag

prob = np.zeros((Dim,Dim))
probTilde = np.zeros((Dim,Dim))
probTilde1 = np.zeros((Dim,Dim))
M = np.zeros((Dim,Dim))
M1 = np.zeros((Dim,Dim))
W = np.zeros((Dim,Dim))
WTilde = np.zeros((Dim,Dim))
S = np.zeros((Dim,Dim))
STilde = np.zeros((Dim,Dim))

PTemp = np.zeros((Dim,Dim,Dim))
for i in range(Dim):
	PTemp[i,i,i] = 1

for i in range(Dim):
	for j in range(Dim):
		M = RTAU@PTemp[j,:,:]@inv(RTAU)@U@PTemp[i,:,:]@Udag@RTAU@PTemp[j,:,:]@inv(RTAU)
		prob[i,j] = rho_0[i,i]*np.trace(M)
		W[i,j] = H_tau_diag[j,j] - H_0_diag[i,i]
		M1 = PTemp[i,:,:]@Udag@RTAU@PTemp[j,:,:]@inv(RTAU)@U@PTemp[i,:,:]
		probTilde[j,i] = rho_tauTH[j,j]*np.trace(M)
#		probTilde1[j,i] = prob[i,j]*np.exp(-beta*(W[i,j]-DeltaF))
		WTilde[j,i] = -W[i,j]
		STilde[j,i] = -S[i,j]

a = np.array(W).flatten()
a.sort(axis=0)
Val = np.unique(a.round(decimals=0))

b = np.zeros(Val.size)
bTilde = np.zeros(Val.size)
for i in range(Dim):
	for j in range(Dim):
		x = np.where(Val == W[i,j].round(decimals=0))
		b[x] = b[x] + prob[i,j]
		y = np.where(Val == WTilde[i,j].round(decimals=0))
		bTilde[y] = bTilde[y] + probTilde[i,j]
#bTilde = np.flip(bTilde)

#for i in range(ValTilde.size):
#	bTilde1[i] = b[i]*np.exp(-beta*(Val[i]-DeltaF))
#bTilde1 = np.flip(bTilde1)

barWidth = 1.2
fig = plt.figure(facecolor="white", figsize=(12,9))
plt.bar(Val,b, color="#740938", alpha = 0.7, label=r'$p(W)$', width=barWidth)
plt.plot(Val,b, color="#740938", marker='o', alpha = 0.7, linestyle='none', markersize=10)
plt.bar(-Val,bTilde, color="#727D73", alpha = 0.8, label=r'$\tilde{p}(-W)$', width=barWidth)
plt.plot(-Val,bTilde, color="#727D73", alpha = 0.7, marker='o', linestyle='none', markersize=10)
plt.axvline(x = DeltaF, color="#3D3D3D", alpha = 0.8, label = r'$\Delta F$', linewidth=5, linestyle='dashed')

colorForward = ['#DE7C7D', '#CC2B52', '#DE7C7D']
colorBackward = ['#AAB99A', '#D0DDD0', '#F0F0D7']

#for l in range(3):
for l in range(1):
	rho_tauTH_CG = np.zeros((Dim,Dim))
	rho_0_CG = np.zeros((Dim,Dim))
	rho_0_CG1 = np.zeros((Dim,Dim))
	H_0_CG = np.zeros((Dim,Dim))
	H_tau_CG = np.zeros((Dim,Dim))
	H_tau_CG_diag = np.zeros((Dim,Dim))

	CGfactor = l + 4
	energies0=np.diag(H_0_diag)
	energiesTAU=np.diag(H_tau_diag)
#	delta= CGfactor*abs(energies0[1]-energies0[0])
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

	non_zero_PTAU = []

	for j in range(0,NStepsTAU):
	    array_sum = sum(sum(PTAU[j,:,:])) # Sum all elements of the matrix
	    if array_sum != 0: # If the matrix is not all zeros, keep it
	    	non_zero_PTAU.append(PTAU[j, :, :])

	# Convert the list back to a NumPy array
	PTAU = np.array(non_zero_PTAU)

	NStepsTAU = int(PTAU.size/Dim**2)

	for i in range(NSteps0):
		if int(np.trace(P0[i,:,:]))>0:
			rho_0_CG = rho_0_CG + np.trace(rho_0@P0[i,:,:])/int(np.trace(P0[i,:,:]))*P0[i,:,:]
			E_J0 = - k_B*T*np.log(np.trace(expm(-beta*H_0_diag)@P0[i,:,:])/int(np.trace(P0[i,:,:])))
			H_0_CG = H_0_CG + E_J0*P0[i,:,:]
	rho_0_CG1 =expm(-beta*H_0_CG)/np.trace(expm(-beta*H_0_CG))
#	print('rho_0_CG = ', rho_0_CG)
#	print('rho_0_CG1 = ', rho_0_CG1)            
#Gonzalo: These two up are ok, so I comment

	for i in range(NStepsTAU):
		if int(np.trace(PTAU[i,:,:]))>0:
#			rho_tauTH_CG = rho_tauTH_CG + np.trace(rho_tauTH@RTAU@PTAU[i,:,:]@inv(RTAU))/int(np.trace(PTAU[i,:,:]))*RTAU@PTAU[i,:,:]@inv(RTAU)
			rho_tauTH_CG = rho_tauTH_CG + np.trace(rho_tauTH@PTAU[i,:,:])/int(np.trace(PTAU[i,:,:]))*PTAU[i,:,:]
			E_JTAU = - k_B*T*np.log(np.trace(expm(-beta*H_tau)@RTAU@PTAU[i,:,:]@inv(RTAU))/int(np.trace(PTAU[i,:,:])))
			H_tau_CG = H_tau_CG + E_JTAU*RTAU@PTAU[i,:,:]@inv(RTAU)
			rho_tau_CG = U@rho_0_CG@Udag
			E_JTAU_diag = - k_B*T*np.log(np.trace(expm(-beta*H_tau_diag)@PTAU[i,:,:])/int(np.trace(PTAU[i,:,:])))
			H_tau_CG_diag = H_tau_CG_diag + E_JTAU_diag*PTAU[i,:,:]
	rho_tauTH_CG1 = expm(-beta*H_tau_CG_diag)/np.trace(expm(-beta*H_tau_CG_diag))
#	print('rho_tauTH_CG = ', rho_tauTH_CG)
#	print('rho_tauTH_CG1 = ', rho_tauTH_CG1)            
#Gonzalo: These two up are ok, so I comment            
        
	W_CG = np.zeros((NSteps0,NStepsTAU))
	S_CG = np.zeros((NSteps0,NStepsTAU))
	prob_CG = np.zeros((NSteps0,NStepsTAU))
	probTilde_CG = np.zeros((NStepsTAU,NSteps0))
	probTilde_CG1 = np.zeros((NStepsTAU,NSteps0))
	WTilde_CG = np.zeros((NStepsTAU,NSteps0))
	for i in range(NSteps0):
		for j in range(NStepsTAU):
			W_CG[i,j] = np.trace(H_tau_CG_diag@PTAU[j,:,:])/int(np.trace(PTAU[j,:,:])) - np.trace(H_0_CG@P0[i,:,:])/int(np.trace(P0[i,:,:]))
			P_0 = np.trace(rho_0_CG@P0[i,:,:])
			M = RTAU@PTAU[j,:,:]@inv(RTAU)@U@P0[i,:,:]@Udag@RTAU@PTAU[j,:,:]@inv(RTAU)/int(np.trace(P0[i,:,:]))
			prob_CG[i,j] = P_0*np.trace(M)
			M1 = P0[i,:,:]@Udag@RTAU@PTAU[j,:,:]@inv(RTAU)@U@P0[i,:,:]/int(np.trace(PTAU[j,:,:]))
			P_TAU = np.trace(rho_tauTH_CG@PTAU[j,:,:])
			probTilde_CG[j,i] = P_TAU*np.trace(M1)
			probTilde_CG1[j,i] = prob_CG[i,j]*np.exp(-beta*(W_CG[i,j]-DeltaF))   
			print('CROOKs TEST',probTilde_CG[j,i]/probTilde_CG1[j,i])
			WTilde_CG[j,i] = -W_CG[i,j]
    
#TESTING CONDITIONAL PROBABILITIES 
#			test = np.trace(M)/np.trace(M1)
#			print('test = ', test)
#This is 1 as it should!
#			test = (int(np.trace(P0[i,:,:]))*np.trace(M))/(int(np.trace(PTAU[j,:,:]))*np.trace(M1))            
#			print('test = ', test) 

#TESTING INITAL STATES NORMALIZATION
	print('Tr[rho_0_CG] = ', np.trace(rho_0_CG))
	print('Tr[rho_tauTH_CG] = ', np.trace(rho_tauTH_CG))

#TESTING PROBS
#	print('probTilde_CG1 = ', probTilde_CG1)
#	print('probTilde_CG = ', probTilde_CG)
	print('NSteps0 = ', NSteps0)
	print('NStepsTAU = ', NStepsTAU)
	print('H_tau_CG_diag.size = ', H_tau_CG_diag.size)
	print('sum(prob_CG) = ', sum(sum(prob_CG)))
	print('sum(probTilde_CG1) = ', sum(sum(probTilde_CG1)))
	print('sum(probTilde_CG) = ',sum(sum(probTilde_CG)))
    

	a_CG = np.array(W_CG).flatten()
	a_CG.sort(axis=0)
	Val_CG = np.unique(a_CG.round(decimals=0))

	b_CG = np.zeros(Val_CG.size)
	bTilde_CG1 = np.zeros(Val_CG.size)
	for i in range(NSteps0):
		for j in range(NStepsTAU):
			x = np.where(Val_CG == W_CG[i,j].round(decimals=0))
			b_CG[x] = b_CG[x] + prob_CG[i,j]
			bTilde_CG1[x] = bTilde_CG1[x] + probTilde_CG[j,i]

	bTilde_CG = np.zeros(Val_CG.size)
	for i in range(NStepsTAU):
		for j in range(NSteps0):
			x = np.where(Val_CG == WTilde_CG[i,j].round(decimals=0))
			bTilde_CG[x] = bTilde_CG[x] + probTilde_CG[i,j]
#
	plt.bar(Val_CG,b_CG, color=colorForward[l], alpha = 0.7, label=r'$p(\breve{W})$, $\alpha$ = '+str(CGfactor), width=barWidth)
	plt.bar(-Val_CG,bTilde_CG, color=colorBackward[l], alpha = 0.7, label=r'$\tilde{p}(-\breve{W})$, $\alpha$ = '+str(CGfactor), width=barWidth)
	plt.plot(-Val_CG,bTilde_CG, color=colorBackward[l], marker='o', alpha = 0.8, linestyle='none', markersize=10)
	plt.plot(Val_CG,b_CG, color=colorForward[l], marker='o', alpha = 0.8, linestyle='none', markersize=10)

	#print('Val_CG = ', Val_CG)
	#print('b_CG = ', b_CG)

plt.xlabel(r'W', size=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(prop={"size":28})
plt.show()
fig.savefig("GaussianFB.pdf")

fig1 = plt.figure(facecolor="white", figsize=(12,9))
plt.plot(Val_CG, np.log(b_CG/bTilde_CG1), label=r'$\mathrm{log}\left[\dfrac{p(\breve{W})}{\tilde{p}(-\breve{W})}\right]$', color='#A6CDC6', linewidth=5, alpha = 0.6)
plt.plot(Val_CG, beta*(Val_CG - DeltaF), color='#DDA853', alpha = 0.7, label=r'$\beta (\breve{W} - \Delta F)$', marker='o', linestyle='none', markersize=15)

matplotlib.rcParams['legend.fancybox'] = True
matplotlib.rcParams['legend.loc'] = 'upper left'
matplotlib.rcParams['legend.fontsize'] = 'small'
matplotlib.rcParams['legend.framealpha'] = 0.3
matplotlib.rcParams['legend.edgecolor'] = 'grey'
plt.xlabel(r'W', size=28)
#plt.ylabel(r'$\mathrm{log}\left[\dfrac{p(\breve{W})}{\tilde{p}(-\breve{W})}\right]$', size=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(prop={"size":28})
plt.show()
fig1.savefig("DetailedFT.pdf")
