#**Copyright:** Svetlin Tassev (2022-2023)
#**License:** GNU General Public License v3.0
#This file is part of Mechanical-Linkage-Neural-Network (https://github.com/stassev/Mechanical-Linkage-Neural-Network).


"""
#different F generators

Same as the 2D version. This time for N-dimensions.

This module comes with a few predefined generators for F.

"""
import numpy as np
def check_vert(F,i):
    return F[:,i].sum()

def check_horiz(F,i):
    return F[i,:].sum()
    
def validateF(FL,Ndim):
    F,[nIn,nOut]=FL
    if (F.shape[0]+nOut!=F.shape[1]+nIn):
        raise Exception("Shape of F is not correct.")
    for j in range(F.shape[1]):
        if check_vert(F,j)!=Ndim:
            raise Exception("Nodes in F must have exactly Ndim back connections.")
    for i in range(1,F.shape[0]):
        if check_horiz(F,i)==0:
            raise Exception("Nodes in F must have at least one connection.")
    for j in range(F.shape[1]):
        for i in range(j+nIn,F.shape[0]):
            if F[i,j]!=0:
                raise Exception("Graph must be directed.")
    return True


def chec_triangle(F1,i,j,Ndim):
    n=F1[1]
    F=F1[0]
    Nin=n[0]
    Nout=n[1]
    Nnodes=F.shape[0]+Nout
    G=np.zeros([Nnodes,Nnodes],dtype=np.int32)
    G[:-Nout,Nin:]=F
    J=j+Nin
    
    # All input nodes are interconnected since they have fixed positions. That is all except node 1 (see below).
    G[:Nin,:Nin]=1
    
    #Node 1 (the crank) is only connected to node 0, which it orbits
    G[:,1]=0
    G[1,:]=0
    G[0,1]=0
    G[1,0]=0
    
    t=True


    if ((G[:,i]*G[:,J]).sum()>=Ndim-1):
        t=False

    if ((G[i,:]*G[J,:]).sum()>=Ndim-1):
        t=False

    if ((G[i,:].reshape(-1)*G[:,J].reshape(-1)).sum()>=Ndim-1):
        t=False

    return t


def get_shortest_distance_between_IJ(F1,I,J):
    F,n=F1
    Nin=n[0]
    Nout=n[1]
    Nnodes=F.shape[0]+Nout
    # Convert F to square matrix
    G=np.zeros([Nnodes,Nnodes],dtype=np.int32)
    G[:-Nout,Nin:]=F
    J+=Nin
    # distance vector
    d=np.zeros(Nnodes,dtype=np.int32)+10**10
    d[I]=0
    # visited? vector
    v=np.zeros(Nnodes,dtype=np.int32)
    v[I]=1

    r=1
    while(r>0):
        r=0
        for m in np.where(v==1)[0]:
            #for k in np.where(v==0)[0]:
            for k in range(Nnodes):
                if G[m,k]==1:
                    if d[k]>d[m]+1:
                        d[k]=d[m]+1
                        r+=1
                    if v[k]!=1:
                        v[k]=1
                        r+=1
    return d[J]


def get_shortest_distance_between_InOut(F1):
    F,n=F1
    Nin=n[0]
    Nout=n[1]
    Nnodes=F.shape[0]+Nout
    d=10000
    for i in range(Nin):
        for j in range(Nout):
            d1=get_shortest_distance_between_IJ(F1,i,F.shape[1]-j-1)
            if (d1<d):
                d=d1
    return d


def get_longest_distance_between_IJ(F1,I,J,skip_node_0=True):
    F,n=F1
    Nin=n[0]
    Nout=n[1]
    Nnodes=F.shape[0]+Nout
    # Convert F to square matrix
    G=np.zeros([Nnodes,Nnodes],dtype=np.int32)
    G[:-Nout,Nin:]=F
    if skip_node_0:
        G[0,1]=1
    J+=Nin
    # distance vector
    d=np.zeros(Nnodes,dtype=np.int32)+10**10
    d[I]=0
    # visited? vector
    v=np.zeros(Nnodes,dtype=np.int32)
    v[I]=1

    r=1
    while(r>0):
        r=0
        for m in np.where(v==1)[0]:
            #for k in np.where(v==0)[0]:
            for k in range(Nnodes):
                if G[m,k]==1:
                    if (d[k]<d[m]+1) or d[k]>1e5:
                        d[k]=d[m]+1
                        r+=1
                    if v[k]!=1:
                        v[k]=1
                        r+=1
    return d[J]


def get_longest_distance_between_InOut(F1,skip_node_0=True):
    F,n=F1
    Nin=n[0]
    Nout=n[1]
    Nnodes=F.shape[0]+Nout
    d=0
    for i in range(Nin):
        for j in range(Nout):
            d1=get_longest_distance_between_IJ(F1,i,F.shape[1]-j-1,skip_node_0=skip_node_0)
            if (d1>d):
                d=d1
    return d

def genF_Random(Ndim,Npts,Nin,Nout,triangles=True,shortestDistance=0,longestDistance=1000,strictDistanceInequality=False):
    restart=1
    NTries=0
    NTriesMax=50000
    while (restart and (NTries<NTriesMax)):
        NTries+=1
        restart=0 
        Ff=np.triu(np.random.rand(Npts, Npts), 1)[:-Nout,Nin:]

        nx=Npts-Nout
        ny=Npts-Nin
        endx=nx+1

        F=Ff*0
        for i in range(1,endx): 
            done=0
            while (not(done)):
                if (Ff[nx-i,:].sum()>1.e-6): # use every row at least once
                    ind=np.argmax(Ff[nx-i,:])#column index of max of each row
                    if ((check_vert(F,ind)<Ndim)):
                        if triangles or chec_triangle([F,[Nin,Nout]],nx-i,ind,Ndim):
                            F[nx-i,ind]=1 # create connection
                            done=1
                    Ff[nx-i,ind]=0 # pop that index
                else:
                    restart=1
                    done=1
        for i in range(ny):
            while ((check_vert(F,i)<Ndim) and (restart!=1)):
                if (Ff[:,i].sum()>1.e-6):
                    ind=np.argmax(Ff[:,i])
                    if triangles or chec_triangle([F,[Nin,Nout]],ind,i,Ndim):
                        F[ind,i]=1
                    Ff[ind,i]=0
                else:
                    restart=1
        if (NTries%100==0):
            print(NTries," out of ",NTriesMax,'. Short D=',get_shortest_distance_between_InOut([F,[Nin,Nout]]),". Long D=",get_longest_distance_between_InOut([F,[Nin,Nout]]))
            
        if (strictDistanceInequality):
            if (get_shortest_distance_between_InOut([F,[Nin,Nout]])!=shortestDistance) and shortestDistance!=0: 
                restart=1
            if (get_longest_distance_between_InOut([F,[Nin,Nout]])!=longestDistance) and longestDistance!=1000:
                restart=1
        else:
            if (get_shortest_distance_between_InOut([F,[Nin,Nout]])<shortestDistance) and shortestDistance!=0:
                #print(get_shortest_distance_between_InOut([F,[Nin,Nout]]))
                restart=1
            if (get_longest_distance_between_InOut([F,[Nin,Nout]])>longestDistance) and longestDistance!=1000:
                #print(get_longest_distance_between_InOut([F,[Nin,Nout]]))
                restart=1
    F=F.astype(bool)
    if NTries==NTriesMax:
        print("Failed. Check the input parameters or raise NTriesMax.")
        return None
    return [F,[Nin,Nout]]



###############
###############
###############
###############
###############


def genF_OneLayer(Nin,Nout,Ndim):
    # Create F connecting Nin joints to Nout joints
    if (Nin>Ndim*Nout):
        raise Exception("The input should have Nin<=2*Nout.")
    restart=1
    while (restart):
        restart=0
        Ff=np.random.rand(Nin, Nout)
        F=Ff*0
        for i in range(Nin):
            done=0
            while (not(done)):
                if (Ff[i,:].sum()>1.e-6):
                    ind=np.argmax(Ff[i,:])#column index of max of each row
                    if ((check_vert(F,ind)<Ndim)):
                        F[i,ind]=1
                        done=1
                    Ff[i,ind]=0
                else:
                    restart=1
                    done=1
        for i in range(Nout):
            while ((check_vert(F,i)<Ndim) and (restart!=1)):
                if (Ff[:,i].sum()>1.e-6):
                    ind=np.argmax(Ff[:,i])
                    F[ind,i]=1
                    Ff[ind,i]=0
                else:
                    restart=1
    F=F.astype(bool)
    return [F,[Nin,Nout]]

def genF_ManyLayers(layers,Ndim):
    #layers=[3,12,12,1]
    l=[]
    for i in range(1,len(layers)):
        l1=layers[i-1]
        l2=layers[i]
        if (l1>Ndim*l2):
            l00=l1
            while(l00>Ndim*l2):
                l01=l00//Ndim
                if l00%Ndim!=0:
                    l01+=1
                if l01<l2:
                    l01=l2
                if l01<Ndim and l2<Ndim:
                    l01=Ndim
                l.append([l00,l01])
                l00=l01
            l.append([l00,l2])
        else:
            l.append([l1,l2])
    l=np.array(l)
    Nx=l[:,0].sum()#+layers[-1]
    Ny=l[:,1].sum()
    F=np.zeros([Nx,Ny]).astype(bool)
    i0=0
    j0=0
    #print(l)
    for i in range(len(l)):
        l1=l[i,0]
        l2=l[i,1]
        F1=genF_OneLayer(l1,l2,Ndim)[0]
        F[i0:i0+l1,j0:j0+l2]=F1
        i0+=l1
        j0+=l2
    return [F,[layers[0],layers[-1]]]
####

def genF_ResNetConnect(baseF,residualF):
    #showF(genF_ResNet_connect(genF_multiple_layers([3,5,5, 2]), genF_multiple_layers([2,5,5, 2])))
    Fa,[na1,na2]=baseF
    Fb,[nb1,nb2]=residualF
    if not((nb1==nb2) and (nb1==na2)):
        raise ValueError
    
    nax,nay=Fa.shape
    nbx,nby=Fb.shape

    F=np.zeros([nax+nbx+na2,nay+nby+na2])
    F[0:nax,0:nay]=Fa
    F[nax:nax+nbx,nay:nay+nby]=Fb
    kx=nax+nbx
    ky=nay+nby
    Nv=nbx+1
    v=np.zeros([Nv-2])
    v=np.insert(v,0,1)
    v=np.insert(v,len(v),1)
    for i in range(na2):
        F[nax+i:nax+(Nv)+i,ky+i]=v[:]
    return [F,[na1,na2]]

def genF_Stack(F1,F2):
    Fa,[na1,na2]=F1
    Fb,[nb1,nb2]=F2
    if not(nb1==na2):
        raise ValueError
    
    nax,nay=Fa.shape
    nbx,nby=Fb.shape

    F=np.zeros([nax+nbx,nay+nby])
    F[0:nax,0:nay]=Fa
    F[nax:nax+nbx,nay:nay+nby]=Fb
    return [F,[na1,nb2]]

#showF(addFs(genF_ResNet_connect(genF_multiple_layers([3,12,12, 8]), genF_multiple_layers([8,5,7, 8])),genF_multiple_layers([8,1])))


###########
###########
###########
###########
###########


#######
#Utilities
###
###
import graphviz 
def showF(FL,lines=True,node0connection=True):
    """
   The showF function uses the Graphviz library to generate a graphical 
   representation of the matrix (FL) of connections generated by one of the functions
   in this module. 
   The graphical representation of the matrix of connections is 
   displayed as an image.
   
   EXAMPLES:

    
    
    showF(generateLongQwith3_2x2(30)) # 3->(2->2)->2
    showF(generateLongQwith3(30)) # 3->(3->3)->3->2->1
    showF(generateLongQ(30)) # 3->(1->1)->1
    showF(generateLongQwith3x3(30)) # 3->(3->3)->3
    
    showF(genF_multiple_layers([3,12,12,1]))# 3->12->12->1 plus intermediate layers as needed to keep it a valid graph.
    showF(addFs(genF_multiple_layers([3,12,4]),genF_multiple_layers([4,7,1]))) # stack one network on another. In this case resulting in 3->12->4->7->1 plus any intermediate layers.
    showF(genF_ResNet_connect(genF_multiple_layers([3,12,12, 2]), genF_multiple_layers([2,5,7, 2]))) # ResNet. connect the output from the first to the output of the second network directly; and also through the second network.
    showF(addFs(genF_ResNet_connect(genF_multiple_layers([3,12,12, 8]), genF_multiple_layers([8,5,7, 8])),genF_multiple_layers([8,1])))
    
    showF(setupF(60)) # 3->(?)->1 random connections between 60 nodes. 
    showF(setupFnoTriangles(30)) # 3->(?)->1 random connections between 30 nodes, this time without any triangular connections of the type A->B, A->C, B->C which lead to rigid motion of C relative to the segment AB.
    
    """
    F,[nl1,nl2]=FL
    nx,ny=F.shape
    dot = graphviz.Digraph()
    if lines:
        dot.graph_attr['splines'] = 'line'
    node=[]
    for i in range(nl1):
        node.append("i"+str(i))
    #nn=max(nx,ny)
    for i in range(nx-nl1):
        node.append(str(i))
    for i in range(nl2):
        node.append("o"+str(i)) 
    
    for n in node:
        dot.node(n,n)
    
    #dot.edge(node[0],node[1])  
    #dot.edge(node[1],node[2])  
    
    for i in range(nx):
        for j in range(ny):
            if F[i,j]:
                dot.edge(node[i],node[j+nl1])
    if (node0connection):
        dot.edge(node[0],node[2])
    #print(dot.source)
    dot.render('round-table.gv', view=True)
    return dot
##
