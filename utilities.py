#**Copyright:** Svetlin Tassev (2022-2023)
#**License:** GPLv3

import numpy as np


def FindC(A,B,l1,l2):
	# given joints A and B, find location of joint Z that is 
	# a distance l1 from A and l2 from B. A,B,Z follow the right hand rule if l2>0 and the left-hand rule if l2<0.
	# INPUT: A,B,l1,l2 (could be arrays)
	# OUTPUT: Z, error err if triangle inequality was not satisied
	
	D=B-A
	Nl=A.shape[0]
	Na=A.shape[1]
	
	s2=(D[...,0]**2+D[...,1]**2) #.reshape(Nl,Na)
	#s=np.sqrt(s2)
	#d=np.zeros_like(l2)+1.0
	#ind=np.where(l2<0)
	#d[ind]=-1.0
	d=np.sign(l2)
	#T2=(l1 + l2 - s)*(l1 - l2 + s)*(-l1 + l2 + s)*(l1 + l2 + s)
	
	eps=1.e-4
	
	T2=((l1+l2)**2-s2)*(s2-(l1-l2)**2)
	
	check=T2/4/((np.sqrt(s2)+np.abs(l1)+np.abs(l2))/3.0)**4 # area of (rhomboid/average side^2)^2
	
	ind=np.where(check>=eps)
	T=np.zeros_like(T2)
	T[ind]= np.sqrt(T2[ind])/(2*s2[ind])
	
	Ut=np.zeros_like(T2)
	Ut[ind] = ((l1**2 - l2**2)/(2*s2))[ind]
	#print(T2,d,T,Ut)
	err=-(check-eps)
	err[ind]=0
	
	#Ut = (l1**2 - l2**2)/(2*s2);
	# coordinates of point Z, that is l1 from A and l2 from B.
	Z=(A+B)/2
	Z[...,0] += D[...,0]*Ut - D[...,1]*T*d
	Z[...,1] += D[...,1]*Ut + D[...,0]*T*d
	
	#Z=A+Matrix[[U,-T],[T,U]].(B-A)
	#U = (l1**2 - l2**2 + s**2)/(2*s**2);
	#Z1 = Ax + (-Ax + Bx)*U + (Ay - By)*T*d;
	#Z2 = Ay + (-Ay + By)*U + (-Ax + Bx)*T*d;
	return Z,err

def joints(ang,lds,FL,npts):
	
	# returns the coordinates of the first npts joints.
	# INPUT: angle ang (could be an array of angles), strut lengths lds, connection matrix FL, npts
	# OUTPUT: coortinates P, error err if triangle inequality not satisfied
	
	
	# lds has dims: realization, joint, [x,y].
	# Nl= number of realizations of l.
	# if only one realization, then Nl=1.
	F,[nl1,nl2]=FL
	if ((nl1 != 3) or (nl2 != 1)):
		raise ValueError
	Npts=F.shape[0]+nl2 # number of joints
	Na=ang.shape[0] # number of samples in angle
	Nls=lds.shape
	if (len(Nls)==3): # Nl>1
		Nl=Nls[0]
	else:
		lds=lds.reshape(1,Nls[0],Nls[1])
		Nl=1
	P=np.zeros([Nl,Na,Npts,2]) # the dims are: realizations, angle samples, joint,[x,y] coo
	#Fix the first point to coo=(0,1)
	P[:,:,1,0]=lds[:,0,0].reshape(-1,1)
	P[:,:,1,1]=lds[:,0,1].reshape(-1,1)###
	# send 2nd point to be r=lds[...,1,0] away from pt. 0
	P[...,2,0] = P[...,0,0]+lds[...,1,0].reshape(-1,1)*np.cos(ang)
	P[...,2,1] = P[...,0,1]+lds[...,1,0].reshape(-1,1)*np.sin(ang)
	err=np.zeros([Nl,Na]) # will contain the total error for each realization, sample
	for i in range(nl1,npts):
		ind=np.where(F[:,i-nl1])[0]
		l1=lds[...,i-1,0].reshape(-1,1)
		l2=lds[...,i-1,1].reshape(-1,1)
		fc = FindC(P[...,ind[0],:],P[...,ind[1],:],l1,l2) # Coo of pt l1,l2 away from ind[0],ind[1]
		P[...,i,:]=fc[0][...,:]
		err+=fc[1]
	return P,err


def sample_joints(lds,FL,N,npts): 
	# samples N times in angle the joints only up to index npts
	ang=np.linspace(0, 2 * np.pi, N)
	p=joints(ang,lds,FL,npts)
	return p


def setupCoo(FL):
	# Given a connection FL, pick random lds's that have allowed 
	# values for all locations of the crank.
	# INPUT: connection FL
	# OUTPUT: strut lengths lds
	
	Lmax=2.
	Nsamples=36 # sample Nsamples times in angle.
	NL=30       # each l1,l2 is sampled between 0,Lmax NL times.
	F,[nl1,nl2]=FL
	Npts=F.shape[0]+nl2   # This is how many nodes the network has.
	
	# ldsI contains NL**2 realizations to be sampled for correct triangle inequalities.
	# ldsI has dims: realization, joint, [x,y]
	ldsI=np.zeros([NL**2,Npts-nl1+2,2])# nl1=3 pts. so -nl1+2=-1 since first joint is at origin, so it's not a dof.
	# lds contains final accepted realization.
	lds=np.zeros([Npts-nl1+2,2])
	lds[0,0]=1.0 # The second point is placed at coordinates (0,1).
	lds[1,0]=np.abs(np.random.randn(1))[0]# This is the radius of the crank.
	ldsI[...,0,0]=lds[0,0]
	ldsI[...,1,0]=lds[1,0]
	
	y,x = np.indices((NL,NL))
	ss=1
	for i in range(nl1,Npts):
		L=[]
		la=0
		lb=Lmax
		s=1
		ind=[[]]
		ind1=[[]]
		
		
		while ((len(ind[0])==0) and (len(ind1[0])==0) and (s<=30)):
			#ss*=-1 # Either this or the next.
			ss=np.floor(np.random.rand(1)[0] - 0.5) * 2 + 1
			L=[]
			print(i)
			#ll=np.linspace(la,lb,NL)
			ll=np.random.rand(NL)*(lb-la)+la
			l1=ll[x.flatten()]
			ll=np.random.rand(NL)*(lb-la)+la
			l2=ll[y.flatten()]	
	
			ldsI[...,i-1,0]=l1
			ldsI[...,i-1,1]=l2
			_,err=sample_joints(ldsI,FL,Nsamples,i+1) 
			ind=np.where(err.sum(axis=1)<1.e-14)
			
			#print(" r ",err.shape, err.sum(axis=1).shape,l1.shape)
			if (len(ind[0])>0):
				print(ind[0].shape, ldsI.shape,ldsI[ind[0],...].shape)
				_,err1=sample_joints(ldsI[ind[0],...],FL,360*2,i+1)
				ind1=np.where(err1.sum(axis=1)<1.e-14)
				
				#print(ind)
				if (len(ind1[0])>0):
					# Pick one of the joints that work.
					jn=ind1[0]
					ii=np.random.randint(0,len(jn))
					lds[i-1,0]=l1[ind[0][jn[ii]]]
					lds[i-1,1]=l2[ind[0][jn[ii]]]*ss
					# Fix the i-1 joint to those lengths in all realizations.
					ldsI[...,i-1,0]=lds[i-1,0]
					ldsI[...,i-1,1]=lds[i-1,1]	
				else:
					ind=[[]] # force a restart of the search within larger radius (see Lb=Lmax*s below.)
					
			s+=1.0 # increase radius prefactor.
			lb=Lmax*s # Largest length to explore
	return lds
