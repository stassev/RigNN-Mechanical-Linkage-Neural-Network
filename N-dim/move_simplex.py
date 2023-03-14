from simplex import find_coordinates
import numpy as np
from Fgen3_ND import genF_Random,showF,genF_ManyLayers,genF_Stack
from scipy.linalg import null_space


def generate(coo,rmin=0.1): 
	# find some lds that satisfies triangle inequalities, when a vertex is placed lds away from coo
	restart=1
	rmax=rmin*1.5+0.5
	Ndim=coo[0].shape[0]
	I=0
	while (restart==1):
		I+=1
		if I%10==0:
			rmax*=1.4
		if (I>1e2):
			raise
		#print(1)
		restart=0
		lds=rmin+np.random.rand(Ndim)*(rmax-rmin)
		try:
			Z=find_coordinates(coo,lds)
			if len(Z[0])!=2:
				restart=1
		except:
			restart=1
		if restart==0:
			#return np.concatenate((coo,[Z[0][np.random.choice(2)]]),axis=0),lds
			c=np.random.choice(2)
			return Z[0][c],lds,c
		



def single_realization_step1():
	Nin=8
	Nout=1
	Ndim=3
	Nnodes=Nin+4+Ndim+1
	F=genF_Random(Ndim,Nnodes,Nin,Nout,triangles=False,shortestDistance=2,longestDistance=2+1+1,strictDistanceInequality=True)
	showF(F,lines=False,node0connection=False)
	return F,Nin,Nout,Ndim,Nnodes

def single_realization_step2(F,Nin,Nout,Ndim,Nnodes):
	coo=np.random.randn(Nin,Ndim)
	COO=np.zeros((Nnodes,Ndim))
	COO[:Nin,:]=coo.copy()
	LDS=[]
	CHOICE=[]
	for i in range(Nnodes-Nin):
		ind=list(np.where(F[0][:,i]==1))
		coo=COO[ind,:][0]
		#print(coo)
		cs,lds,c=generate(coo)
		COO[i+Nin,:]=cs
		LDS.append(lds)
		CHOICE.append(c)
	return COO,LDS,CHOICE
	

def single_realization():
	F,Nin,Nout,Ndim,Nnodes = single_realization_step1()
	COO,LDS,CHOICE = single_realization_step2(F,Nin,Nout,Ndim,Nnodes)
	return F,COO,LDS,CHOICE,Nin,Nout,Ndim,Nnodes

def check_rotation(Nangle):
	# check that triangle inequalities are satisfied for all Nangle values of angle.
	restart=1
	while (restart==1):
		realization=1
		F,Nin,Nout,Ndim,Nnodes = single_realization_step1()
		restart = 1
		while(restart==1):
			if (realization>1000):
				break
			realization+=1
			print(realization)
			restart=0
			try:
				COO,LDS,CHOICE = single_realization_step2(F,Nin,Nout,Ndim,Nnodes)
				# Construct v0,v1 such that they
				# are orthonormal; v1 is orthogonal to coo1-coo0
				# and v0=coo2-coo0
				if Ndim==2:
					return None
				v0=COO[1,:]-COO[0,:]
				r_rot=np.linalg.norm(v0)
				v0/=r_rot
				Mij=np.zeros((Ndim,Ndim))
				Mij[0,:]=v0
				for i in range(Ndim-2):
					Mij[1+i,:]=COO[2+i,:]-COO[1,:]
				v1=null_space(Mij)[:,0]
				
				angle=np.linspace(0, 2 * np.pi, Nangle)
				for ang in angle:
					COO[1,:]=COO[0,:]+r_rot*(np.cos(ang)*v0+np.sin(ang)*v1)
					for i in range(Nnodes-Nin):
						ind=np.where(F[0][:,i]==1)
						coo=COO[ind,:][0]
						lds=LDS[i]
						#print(coo)
						#print(lds)
						Z=find_coordinates(coo,lds)
						if len(Z[0])!=2:
							raise
						COO[i+Nin,:]=Z[0][CHOICE[i]]
			except:
				restart=1
	return COO[:Nin,:],LDS,CHOICE,[v0,v1,r_rot],F,Nin,Nout,Ndim,Nnodes
	
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#plt.style.use('seaborn-poster')


def plot_link(coo,LDS,CHOICE,V,F,Nin,Nout,Ndim,Nnodes,Nangle,Nturns):
	COO=np.zeros((Nnodes,Ndim,Nangle))
	v0,v1,r_rot=V
	#Nangle=100
	angle=np.linspace(0, Nturns*2 * np.pi, Nangle)
	for j in range(angle.shape[0]):
		COO[:Nin,:,j]=coo.copy()
		print(j)
		ang=angle[j]
		COO[1,:,j]=COO[0,:,j]+r_rot*(np.cos(ang)*v0+np.sin(ang)*v1)
		for i in range(Nnodes-Nin):
			ind=np.where(F[0][:,i]==1)
			c=COO[ind,:,j][0]
			lds=LDS[i]
			Z=find_coordinates(c,lds)
			#print(Z)
			COO[i+Nin,:,j]=Z[0][CHOICE[i]]
	#print(COO)
	fig = plt.figure()
	#plt.show()
	ax = fig.add_subplot(projection='3d')
	#ax = plt.axes(projection='3d')	
	ax.set_box_aspect((np.ptp(COO[:,0,:]), np.ptp(COO[:,1,:]), np.ptp(COO[:,2,:])))
	ax.axes.set_xlim3d(left=np.min(COO[:,0,:]), right=np.max(COO[:,0,:])) 
	ax.axes.set_ylim3d(bottom=np.min(COO[:,1,:]), top=np.max(COO[:,1,:])) 
	ax.axes.set_zlim3d(bottom=np.min(COO[:,2,:]), top=np.max(COO[:,2,:])) 
	
	ax.plot3D(COO[-1,0,:],COO[-1,1,:],COO[-1,2,:],linewidth=0.25,color='red')
	ax.plot3D(COO[1,0,:],COO[1,1,:],COO[1,2,:],linewidth=0.25,color='blue')
	plots=None
	output=1
	for j in range(angle.shape[0]):
		print(j)
		ang=angle[j]
		#ax.set_box_aspect((np.ptp(COO[:,0,:]), np.ptp(COO[:,1,:]), np.ptp(COO[:,2,:])))
		if plots:
			for p in plots:
				if p:
					for p1 in p:
						if p1:
							p1.remove()
		plots=[]
		for i in range(Nnodes-Nin):
			ind=np.where(F[0][:,i]==1)
			c=COO[ind,:,j][0]
			for k in range(c.shape[0]):
				x=[c[k,0],COO[i+Nin,0,j]]
				y=[c[k,1],COO[i+Nin,1,j]]
				z=[c[k,2],COO[i+Nin,2,j]]
				plots.append(ax.plot3D(x,y,z,lw=0.5))
		if ang>=np.pi*2.0:
			output=0
		if output==1:
			plt.savefig(str(j)+".png",dpi=200)
		plt.draw()
		plt.pause(0.01)
		#plt.close('all')



def save_state(filename,coo,LDS,CHOICE,V,F,Nin,Nout,Ndim,Nnodes):
	v0,v1,r_rot=V
	np.savez_compressed(filename,coo=coo,LDS=LDS,CHOICE=CHOICE,v0=v0,v1=v1,r_rot=r_rot,F=F[0],Nin=Nin,Nout=Nout,Ndim=Ndim,Nnodes=Nnodes)
	
def load_state(filename):
	loaded = np.load(filename)
	coo=loaded['coo']
	LDS=loaded['LDS']
	CHOICE=loaded['CHOICE']
	v0=loaded['v0']
	v1=loaded['v1']
	r_rot=loaded['r_rot']
	V=[v0,v1,r_rot]
	Nin=loaded['Nin']
	Nout=loaded['Nout']
	F=[loaded['F'],[Nin,Nout]]
	Ndim=loaded['Ndim']
	Nnodes=loaded['Nnodes']
	return coo,LDS,CHOICE,V,F,Nin,Nout,Ndim,Nnodes

	
#coo,LDS,CHOICE,V,F,Nin,Nout,Ndim,Nnodes=check_rotation(200)
#save_state('output.npz',coo,LDS,CHOICE,V,F,Nin,Nout,Ndim,Nnodes)
#plot_link(coo,LDS,CHOICE,V,F,Nin,Nout,Ndim,Nnodes,300,4)
