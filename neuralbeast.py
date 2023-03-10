#**Copyright:** Svetlin Tassev (2022-2023)
#**License:** GPLv3

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import time
import sys
sys.path.append('./')
from Fgen3 import *
from utilities import *


import torch
from torch.nn.modules import Module
from torch import Tensor
import torch.nn as nn
import numpy as np

from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy import interpolate


#dummies
def cs():
	None
NN=None
X=None
ax1=None
net=None
fig=None
epoch=None
loss=None
optimizer=None

def FindC_torch(A,B,l1,l2):
	# See FindC for extra info.
	D=B-A
	#Nl=A.shape[0]
	#Na=A.shape[1]
	
	s2=(D[...,0]**2+D[...,1]**2) #.reshape(Nl,Na)
	#s=np.sqrt(s2)
	d=torch.sign(l2)
	#T2=(l1 + l2 - s)*(l1 - l2 + s)*(-l1 + l2 + s)*(l1 + l2 + s)
	
	
	eps2=1.e-4
	
	T2=((l1+l2)**2-s2)*(s2-(l1-l2)**2) #16*area of triangle
	
	check=T2/4/((torch.sqrt(s2)+torch.abs(l1)+torch.abs(l2))/3.0)**4 # (2*area)^2/(average side)^4
	
	ind=torch.where(check>=eps2)
	T=torch.zeros_like(T2)
	T[ind]= torch.sqrt(T2[ind])/(2*s2[ind])
	
	Ut=torch.zeros_like(T2)
	Ut[ind] = ((l1**2 - l2**2)/(2*s2))[ind]
	#print(T2,d,T,Ut)
	err=-(check-eps2)
	err[ind]=0
	
	#Z is distance l1 from pt A and dist=l2 from pt B.
	#Z = (A+B)/2 if A,B,Z cannot make triangle since l1 and l2 do not
	#satisfy triangle inequality.
	Z=(A+B)/2
	Z[...,0] += D[...,0]*Ut - D[...,1]*T*d
	Z[...,1] += D[...,1]*Ut + D[...,0]*T*d
	
	return Z,err

def joints_last_torch(ang,lds,FL,npts):
	# Find position of npts joint
	# given angle of the crank (ang), a 
	# set of lengths of struts (lds),
	# the connection matrix FL.
	F,[nl1,nl2]=FL
	if ((nl1 != 3) or (nl2 != 1)):
		raise ValueError
	Npts=F.shape[0]+nl2
	Na=ang.shape[0]
	Nls=lds.shape
	if (len(Nls)==3):
		Nl=Nls[0]
	else:
		lds=lds.reshape(1,Nls[0],Nls[1])
		Nl=1
	P=torch.zeros([Nl,Na,Npts,2])
	# P 
	P[:,:,1,0]=lds[:,0,0].reshape(-1,1)
	P[:,:,1,1]=lds[:,0,1].reshape(-1,1)
	
	P[...,2,0] = P[...,0,0]+lds[...,1,0].reshape(-1,1)*torch.cos(ang)
	P[...,2,1] = P[...,0,1]+lds[...,1,0].reshape(-1,1)*torch.sin(ang)
	err=torch.zeros([Nl,Na])
	for i in range(nl1,npts):
		ind=torch.where(F[:,i-nl1])[0]
		l1=lds[...,i-1,0].reshape(-1,1)
		l2=lds[...,i-1,1].reshape(-1,1)
		fc = FindC_torch(P[...,ind[0],:],P[...,ind[1],:],l1,l2)
		P[...,i,:]=fc[0][...,:]
		err+=fc[1]
	return P[...,npts-1,:],err



def initialize():
	# sample input curve informly. 
	phi=np.random.rand(NN)*2.*np.pi
	X=torch.tensor(cs(phi)).detach().clone()
	Phi=torch.tensor(phi).detach().clone()
	return Phi,X

def initialize_lin():
	phi=np.linspace(0,2.*np.pi,NN)
	X=torch.tensor(cs(phi)).detach().clone()
	Phi=torch.tensor(phi).detach().clone()
	return Phi,X

class NNB(nn.Module):
	"""
	"""
	
	def __init__(self):
		r"""Initializer method.
		"""
		super(NNB, self).__init__()
		
		#F=addFs(connectResF(generateLongQwith3x3(15),generateLongQwith3x3(6)),FfromL([3,1]))
		#F=(genF_Random(24,3,1,triangles=False,shortestDistance=5))
		#F=(genF_Random(15,3,1,triangles=False,shortestDistance=3,longestDistance=5))
		#F=(genF_Random(20,3,1,triangles=False,shortestDistance=4,longestDistance=8))
		F=(genF_Random(30,3,1,triangles=False,shortestDistance=5,longestDistance=10,strictDistanceInequality=True))
		#F=genF_Stack(genF_ResNetConnect(genF_ManyLayers([3,3]),genF_ManyLayers([3,3,3,3,3])),genF_ManyLayers([3,1]))
		#F=[f[0][3:,3:],[3,1]]
		showF(F)
		showF(F,lines=False)
		#F=(genF_ManyLayers([3,4,5,5,1]))

		self.Npts=F[0].shape[0]+F[1][1]
		
		#self.lds=torch.nn.Parameter(torch.randn(self.Npts,2))
		self.lds=torch.nn.Parameter(torch.tensor(setupCoo(F)))
		
		self.theta=torch.nn.Parameter(torch.randn(1)[0]*2*np.pi)
		self.F0=torch.tensor(F[0])
		self.F1=torch.tensor(F[1])
		
		self.phase=torch.nn.Parameter(torch.tensor(1.0))
		
	def forward(self, input):
		
		
		##
		# Here we are trying to construct a function tau(length)=phi which takes
		# the length of the final trace and converts it to angle phi of the crank.
		# Since there may be sudden sharp jumps, we try to construct better
		# samples iteratively that better approximate a uniform sampleing of tau(length)
		# in lnegth
		
		try:
			uniform=np.linspace(0,2.*np.pi,Ninterp)
			for s in range(3):
				
				if (s>0): 
					phi=tau(uniform)
				else:
					phi=uniform
				S,_=joints_last_torch(torch.tensor(phi)+self.phase,self.lds,[self.F0,self.F1],self.Npts)
				y=np.sqrt(((S[0,1:,0]-S[0,:-1,0])**2+(S[0,1:,1]-S[0,:-1,1])**2).detach().numpy())
				yc=np.cumsum(y)
				yc=np.insert(yc,0,0.0)
				yc/=yc[-1]
				yc*=2.*np.pi
				tau = PchipInterpolator(yc,phi)
			###
			
			# assume input samples the length of the resulting curve. 
			# tau(input) converts length to phase phi, to which we add self.phase, which is to be optimized.
			# Then we find the positions of the last joint (last argument=self.Npts).
			P,err=joints_last_torch(torch.tensor(tau(input))+self.phase,self.lds,[self.F0,self.F1],self.Npts)
		except:
			P,err=joints_last_torch(input+self.phase,self.lds,[self.F0,self.F1],self.Npts)
		
		x0=P[0,:,0]
		y0=P[0,:,1]
		
		# Rotate resulting trace by angle theta that is optimized over.
		x=x0*torch.cos(self.theta)+y0*torch.sin(self.theta)
		y=-x0*torch.sin(self.theta)+y0*torch.cos(self.theta)
		
		# Center and scale:
		x1=x-x.mean()
		y1=y-y.mean()
		
		std=torch.sqrt((x1**2+y1**2).mean())
		
		x2=x1/std
		y2=y1/std
		std1=torch.tensor(0)
		if std<0.1:
			std1=1/(std+0.01) # make sure size of output is not too small
		
		
		return (x2,y2,err[0]+std1) #shape(Na) (Na) (Na). The [0] is to select the first (and only) realization.
		

def criterion(x,y,err):
	a,b,c=(((x-X[:,0])**2).mean(),# reduce deviation from curve in x,y
			((y-X[:,1])**2).mean(),
			err.mean()*1.e3) # mean error for triangle inequality and squashed triangles
	print(a,b,c)
	return a+b+c 



def update_plot():
	ax1.clear()
	Phi,X=initialize_lin()
	X=X.detach().cpu().numpy()
	ax1.plot(X[:,0],X[:,1],c='blue')
	x,y,err=net(Phi)
	ax1.plot(x.detach().cpu().numpy(),y.detach().cpu().numpy(),c='red')
	fig.canvas.draw()
	fig.canvas.flush_events()
	plt.pause(0.05)
	
	return

def save_state(filename,i):
		state = {
			'epoch': epoch,
			'state_net': net.state_dict(),
			'F0':net.F0,
			'F1':net.F1,
			'theta':net.theta,
			'Npts':net.Npts,
			'lds':net.lds,
			#'sig':net.sig,
			'state_optimizer': optimizer.state_dict(),
			'net': net,
			'optimizer': optimizer,
			'loss': loss
		}
		torch.save(state,filename+str(int(i))+'.pt')
		

def load_state(filename,i):
	#net=NNB()
	state=torch.load(filename+str(int(i))+'.pt',map_location=torch.device('cpu'))
	net=state['net']
	net.load_state_dict(state['state_net'])
	loss=state['loss']
	epoch=state['epoch']
	optimizer=state['optimizer']
	all_parameters = list(net.parameters())
	#optimizer = torch.optim.Adadelta(all_parameters)#,lr=0.1) 
	optimizer.load_state_dict(state['state_optimizer'])
	net.F0   =state['F0'   ]
	net.F1   =state['F1'   ]
	net.theta=state['theta']
	net.Npts =state['Npts' ]
	net.lds  =state['lds'  ]
	#net.sig  =state['sig'  ]
	return net,optimizer,epoch,loss


def RunNet():
	global cs
	global NN,net
	global X,ax1,fig
	global epoch,optimizer,loss
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dt=np.float64
	torch.set_default_tensor_type(torch.DoubleTensor)
	
	########################################
	# Input curve.
	def In(phi):
		xs=np.cos(phi)
		ys=np.sin(3*phi)
		xs-=xs.mean()
		ys-=ys.mean()
		std=np.sqrt((xs**2+ys**2).mean())
		
		xs/=std
		ys/=std
		return xs,ys
	# sample input curve uniformly in length.
	NinterpX=5000 # interpolation of input curve 
	Ninterp=1000 # interpolation of output curve
	NN=1000
	uniform=np.linspace(0.,2.*np.pi, NinterpX)
	phi=uniform
	xs,ys=In(phi)
	z=np.sqrt((xs[1:]-xs[:-1])**2+(ys[1:]-ys[:-1])**2)
	zc=np.cumsum(z)
	zc=np.insert(zc,0,0.0)
	zc/=zc[-1]
	zc*=2.*np.pi
	tauPhi = PchipInterpolator(zc,phi) # returns phase=tauPhi(length)
	xs,ys=In(tauPhi(uniform)) # uniformly sample along length
	y = np.c_[xs,ys]
	cs = PchipInterpolator(uniform, y) 
	#####################################################
	
	fig, ((ax1)) = plt.subplots(nrows=1,ncols=1)
	
	NiterCheck=5000
	restart=NiterCheck
	try:
		while (restart>=NiterCheck):
			restart=0
			loss=torch.tensor(1.e10)
			net=NNB()
			all_parameters = list(net.parameters())
			#optimizer = torch.optim.Rprop(all_parameters,step_sizes=(1.e-20,50)) 
			#optimizer = torch.optim.Rprop(all_parameters,step_sizes=(1.e-14,50))
			#optimizer = torch.optim.Rprop(all_parameters)#,step_sizes=(1.e-20,50)) 
			#optimizer = torch.optim.LBFGS(all_parameters)#,step_sizes=(1.e-20,50)) 
			#optimizer = torch.optim.RMSprop(all_parameters)#,lr=0.1) 
			#optimizer = torch.optim.SGD(all_parameters,lr=0.01)#,lr=0.1) 
			#optimizer = torch.optim.Adagrad(all_parameters)#,lr=0.1) 
			#optimizer = torch.optim.NAdam(all_parameters)#,lr=0.1) 
			#optimizer = torch.optim.RAdam(all_parameters)#,lr=0.1) 
			#optimizer = torch.optim.ASGD(all_parameters)#,lr=0.1) 
			optimizer = torch.optim.Adadelta(all_parameters,lr=1)#,lr=0.1) 
			#scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer,[5000,15000,20000,25000,30000,35000], gamma=0.3)
			scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99975)#75)
			
			epoch=0
			le=1e9
			s=0
			while (epoch<1e6):
				Phi,X=initialize()
				x,y,err=net(Phi)
				loss=criterion(x,y,err)
				
				print("[EPOCH]: %i, [LOSS]: %e" % (epoch, loss))
				optimizer.zero_grad()
				loss.backward()   #retain_graph=True)
				#optimizer.step(closure)
				optimizer.step()
				
				scheduler1.step()
				if ((epoch>100) and (loss.item()<le)):
					save_state('./output',s%2)
					s+=1
					le=loss.item()
				if (epoch%100==0):
					update_plot()
					plt.savefig('epoch'+str(epoch//100)+'.png')
				epoch+=1
				if (loss>0.5):
					restart+=1
				if (loss>0.1 and (epoch>30000)):
					restart+=1
				if  loss>1e3:
					restart+=10
				if torch.isnan(loss):
					restart=NiterCheck
				if restart>=NiterCheck:
					break
	except:
		from viz3 import showTrajNet
		showTrajNet(net)
		
	return net,epoch,loss
