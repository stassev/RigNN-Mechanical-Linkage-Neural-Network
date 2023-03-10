#**Copyright:** Svetlin Tassev (2022-2023)
#**License:** GNU General Public License v3.0
#This file is part of Mechanical-Linkage-Neural-Network (https://github.com/stassev/Mechanical-Linkage-Neural-Network).

import sys
sys.path.append('./')
from Fgen3 import *
from utilities import *
import torch

def plot_line(a,b,ax):
	return ax.plot([a[0],b[0]],[a[1],b[1]],c='black',linewidth=1)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def showTrajNet(net):
	#Shows the joints' trajectories and legs given an input neural net.
	F=[net.F0.detach().numpy(),net.F1.detach().numpy()]
	lds=net.lds.detach().clone().numpy()
	#lds[2:-1,1]*=(torch.sign(net.sig)).detach().numpy()[2:]
	showTraj(F,lds)
	return

def showTraj(F,lds,N=25000):
	# Shows the joints' trajectories and legs given F and lds.
	Npts=F[0].shape[0]+F[1][1]
	xyN=sample_joints(lds,F,N,Npts)[0][0]
	plt.gca().set_axis_off()
	[fig,ax] = plt.subplots(1, 1)
	
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
				hspace = 0, wspace = 0)
	plt.margins(0,0)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	
	
	P= joints(np.array([0]),lds,F,Npts)[0][0,0]
	plot_line(P[2],P[0],ax)
	for i in range(3,Npts):
		ind=np.where(F[0][:,i-3])[0]
		plot_line(P[ind[0]],P[i],ax)
		plot_line(P[ind[1]],P[i],ax)
	
	for i in range(Npts): 
		xy=xyN[:,i,:]
		points = np.array([xy[:,0], xy[:,1]]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
		w=0.3
		if (i==Npts-1):
			w=2
		lc = LineCollection(segments, cmap="jet",linewidths=w)
		lc.set_array(np.linspace(0,1,xy.shape[0]))
		#lc.set_linewidth(2)
		line = ax.add_collection(lc)
	
	
	#fig.colorbar(line, ax=ax)
	ax.set_xlim(xyN[...,0].min()-0.1,xyN[...,0].max()+0.1)
	ax.set_ylim(xyN[...,1].min()-0.1,xyN[...,1].max()+0.1)
	ax.axis('off')
	ax.set_aspect(1.0)
	plt.show()

#########
#########
#########
#########

from matplotlib import animation
def animateTrajNet(net,N=1500,numDataPoints=1500,save_file=r''):
	F=[net.F0.detach().numpy(),net.F1.detach().numpy()]
	lds=net.lds.detach().clone().numpy()
	#lds[2:-1,1]*=(torch.sign(net.sig)).detach().numpy()[2:]
	animateTraj(F,lds,N=N,numDataPoints=numDataPoints,save_file=save_file)
	return
	
	
def animateTraj(F,lds,N=1500,numDataPoints=1500,save_file=r''):
	Npts=F[0].shape[0]+F[1][1]
	xyN=sample_joints(lds,F,N,Npts)[0][0]
	#legB(lds,F,N,Npts)
	#xyN=legBtau(lds,F,N,Npts)
	
	plt.gca().set_axis_off()
	[fig,ax] = plt.subplots(1, 1)
	
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
				hspace = 0, wspace = 0)
	plt.margins(0,0)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	
	ax.plot([xyN[0,0,0]],[xyN[0,0,1]], marker="o", markersize=2, markeredgecolor="black")
	ax.plot([xyN[0,1,0]],[xyN[0,1,1]], marker="o", markersize=2, markeredgecolor="black")
	
	ax.set_xlim(xyN[...,0].min()-0.1,xyN[...,0].max()+0.1)
	ax.set_ylim(xyN[...,1].min()-0.1,xyN[...,1].max()+0.1)
	ax.set_aspect(1.0)
	ax.axis('off')
	
	P=joints(np.array([0]),lds,F,Npts)[0][0,0]
	p=[ax.plot([P[2,0],P[0,0]],[P[2,1],P[0,1]], lw = 1.5,c='black',alpha=1)]
	
	cl=plt.get_cmap("tab10")
	def cline(x):
		return cl(x%8) 
	
	for i in range(3,Npts):
		ind=np.where(F[0][:,i-3])[0]
		p.append(ax.plot([P[i,0],P[ind[0],0]],[P[i,1],P[ind[0],1]], lw = 1.5,alpha=0.75,c=cline(2*i)))
		p.append(ax.plot([P[i,0],P[ind[1],0]],[P[i,1],P[ind[1],1]], lw = 1.5,alpha=0.75,c=cline(2*i+1)))
	p.append(ax.plot([P[-1,0]],[P[-1,1]], marker="o", markersize=7, markeredgecolor="black",markerfacecolor = "None"))
	from matplotlib import cm
		
	i_beg=0
	do=0
	def animate(num):
		nonlocal p
		nonlocal i_beg
		nonlocal do
		t=(num/(numDataPoints)*2*np.pi)
		
		i_end=int(num/numDataPoints*N+0.5)
		i_end=i_end % N

		##
		if (do<1):
			for i in range(Npts): 
				xy=xyN[i_beg:i_end,i,:]
				if (i_end<i_beg):
					do+=1
					dn=(N-i_beg)+i_end
					if dn>0:
						xy=xyN[-dn-1:,i,:]
						xy[0:-(i_end)+2,:]=xyN[i_beg:,i,:]
						xy[-i_end+2:,:]=xyN[:i_end+1,i,:]
				w=0.3
				if (i==Npts-1):
					w=2
				c=cm.jet((num%numDataPoints)/numDataPoints)
				ax.plot(xy[:,0],xy[:,1],c=c,lw=w)
			
		if len(p) > 0:
			for item in p:
				item[0].remove()	
	
	
	
		P=joints(np.array([t]),lds,F,Npts)[0][0,0]
		p=[ax.plot([P[2,0],P[0,0]],[P[2,1],P[0,1]], lw = 1.5,c='black',alpha=1)]
		for i in range(3,Npts):
			ind=np.where(F[0][:,i-3])[0]
			p.append(ax.plot([P[i,0],P[ind[0],0]],[P[i,1],P[ind[0],1]], lw = 1.5,alpha=0.75,c=cline(2*i)))#,c='black',
			p.append(ax.plot([P[i,0],P[ind[1],0]],[P[i,1],P[ind[1],1]], lw = 1.5,alpha=0.75,c=cline(2*i+1)))#,c='black',
		p.append(ax.plot([P[-1,0]],[P[-1,1]], marker="o", markersize=7, markeredgecolor="black",markerfacecolor = "None"))
		
		i_beg=i_end-1
		if (i_beg<0):
			i_beg=0
		
	
	
	
	line_ani = animation.FuncAnimation(fig, animate, interval=1,   
									frames=numDataPoints*3)
	
	
	if len(save_file)>0:
		f = save_file+".mkv" 
		writervideo = animation.FFMpegWriter(fps=30)
		line_ani.save(f, writer=writervideo,dpi=300)
	
	plt.show()
	return

def StrandbeestOutput():
	# F[0] contains the connection matrix. See Fgen3.py for description.
	
	F=[np.array([[0,0,0,0,0],
				 [1,1,1,0,0],
				 [1,1,0,0,0],
				 [0,0,1,0,0],
				 [0,0,0,1,1],
				 [0,0,0,1,0],
				 [0,0,0,0,1]]),[3,1]]

	lds=np.array([[38.7923 , 0   ], # joint 1 is distance 38.7923 from joint 0 (which is at the origin)
				  [15      , 0   ], # joint 2 is attached to jonit 0 with a crank of radius 15
				  [41.5    , -50  ],# joint 3 is distance l1,l2 from the locations where F[:,0]==1
				  [39.3    ,  61.9],# joint 4 is distance l1,l2 from the locations where F[:,1]==1
				  [40.1    , -55.8],# etc.
				  [36.7    , -39.4],
				  [49.     , -65.7]])
	showTraj(F,lds) # output will not be rotated so flat section of curve is horizontal. do that separately if you want
	animateTraj(F,lds,N=150,numDataPoints=150,save_file=r'strandbeest')
