from numpy.linalg import det
import numpy as np
from math import factorial 
from scipy.linalg import null_space

#def my_nullspace(A):
#	Q, R = torch.linalg.qr(A.T)
#	d=torch.diagonal(R)
#	tol=torch.finfo(R.dtype).eps*R.shape[1]
#	rnum=torch.sum(torch.abs(d)>tol,dtype=int)
#	return Q[:,rnum:]
# a = torch.rand(10, 2,dtype=torch.float64)
# A = torch.mm(a, a.t()) # rank 2 input to have a nullspace of size 8
# #---- OR ----
# A[:2,:]=a.T
# A[2:,:]=0
# A.requires_grad_()
# out = my_nullspace(A)
# 
# print("out size, should be 10x8: ", out.size())
# 
# (out**2).sum().backward()
# print("A.grad", A.grad)
#
# out[:,3].dot(A[0,:])
# out = my_nullspace(A)
# out[:,3].dot(A[4,:])
# out[:,3].dot(A[0,:])
# out[:,3].dot(A[1,:])
# out[:,0].dot(A[1,:])

def signed_volume(coo,Z):
	ndim=coo.shape[0]
	Mij=np.zeros((ndim+1,ndim+1), dtype=np.float64)
	Mij[:,0]=1.0
	Mij[:-1,1:]=coo.astype(np.float64).copy()
	Mij[-1,1:]=Z
	return det(Mij)/np.float64(factorial(ndim))

def volume(L2ij):
	n=L2ij.shape[0]
	ndim=n-1
	Mij=np.zeros((n+1,n+1), dtype=np.float64)
	Mij[0,1:]=1
	Mij[1:,0]=1
	Mij[1:,1:]=L2ij
	d=det(Mij)
	if (-np.sign(d)*(-1)**ndim)<0:
		raise Exception("Triangle inequalities violated.")
	volume=np.sqrt(np.abs(d)/np.float64(2**(ndim)))/np.float64(factorial(ndim))
	return volume

def length2(coo,ls):
	n=coo.shape[0]
	if n!=len(ls):
		raise Exception("coo and ls should be same length.")
	L2ij=np.zeros((n+1,n+1), dtype=np.float64)
	for i in range(n):
		for j in range(n):
			L2ij[i,j]=np.linalg.norm(coo[i]-coo[j])**2
	L2ij[n,:n]=ls**2
	L2ij[:n,n]=ls**2
	for i in range(n):
		for j in range(n):
			a=L2ij[i,j]**0.5
			b=ls[i]
			c=ls[j]
			if (a+b<c) or (np.abs(a-b)>c):
				raise Exception("Triangle inequalities violated.")
	return L2ij

def height(coo,ls):
	L2ij=length2(coo,ls)
	ndim=len(coo) # dim of simplex
	volumeN=volume(L2ij)
	volumeNm1=volume(L2ij[:-1,:-1])
	return volumeN/volumeNm1*np.float64(ndim)

def pop_vertex(coo,ls):
	#consider pyramid ABCD.
	h=height(coo,ls) # height from ABC plane to D
	cooNew=coo[:-1,:] # pop C
	lsNew=np.sqrt(ls[:-1]**2-h**2) # find distance from A and B to to projection of D on (ABC) plane (call that dd). Pop C.

	return cooNew,lsNew,h


def find_height_vector(coo):
	ndim=coo.shape[1]
	Mcoo=np.zeros((ndim,ndim), dtype=np.float64)
	for i in range(len(coo)-2):
		Mcoo[i,:]=coo[i+1,:]-coo[0,:]
	vs=null_space(Mcoo)
	hv=np.zeros(ndim, dtype=np.float64)
	for i in range(vs.shape[1]):
		hv+=vs[:,i].dot(coo[-1,:]-coo[0,:])*vs[:,i]
	hv/=np.linalg.norm(hv)
	return hv

def find_coordinates(coo,ls):
	cooNew=coo.astype(np.float64).copy()
	lsNew=ls.astype(np.float64).copy()
	H=[]
	while (cooNew.shape[0]>=1):
		if (cooNew.shape[0]>=2):
			v=find_height_vector(cooNew)
			try:
				V=np.concatenate(([v],V),axis=0)
			except:
				V=[v]
		cooNew,lsNew,h=pop_vertex(cooNew,lsNew)
		H=[h]+H
	V=np.array(V).T
	Vnull=np.zeros((V.shape[0],V.shape[0]), dtype=np.float64)
	Vnull[:,:V.shape[1]]=V
	v0=null_space(Vnull.T) # Vectors in null_space() are arranged in rows; output in columns
	V=np.concatenate((V,v0),axis=1)
	#print(H)
	#print(V)
	ZZ=[]
	ss=0
	for j in range(2**V.shape[1]):
		Z=coo[0,:].astype(np.float64).copy()
		for i in range(V.shape[1]):
			sign=np.floor((j % 2**(i+1))/2**i)
			sign=(sign-0.5)*2
			#print(i,j,j % 2**(i+1),sign)
			#print(H[i]*V[:,i]*(sign.astype(np.float64)))
			#print(Z)
			#print(sign)
			Z+=H[i]*V[:,i]*(np.float64(sign))
		#print('---')
		s=0
		for i in range(coo.shape[0]):
			s+=((np.linalg.norm(coo[i,:]-Z[:])-ls[i])**2)
		#print(s)
		if s<1.e-20:
			ss+=s
			if signed_volume(coo,Z)>0 and len(ZZ)==1:
				ZZ=[Z]+ZZ
			else:
				ZZ.append(Z)
	return ZZ,ss 
	# ZZ contains the coordinates of the vertex that is `ls` away from `coo`.
	# There are two solutions, corresponding to two signs of the volume determinant. 
	# The first solution in ZZ corresponds to a positive determinant. The second -- to a negative.

# coo=np.random.randn(3,3)
# lds=np.random.rand(3)*2
# Z=find_coordinates(coo,lds)
