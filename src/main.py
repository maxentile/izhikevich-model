# Based on the Izhikevich model published in IEEE Transactions on Neural Networks (2003) 14:1569-1572
# Original paper + MATLAB implementation available at: http://www.izhikevich.org/publications/spikes.htm
# This implemntation simulates a population of densely coupled, electrophysiologically diverse neurons
# and enables interactive visual analysis of the behavior of the population and individual neurons 

""" The Izhikevich Model:
(1) dv/dt = 0.04v^2 + 5v + 140 - u + I
(2) du/dt = a(bv-u)
(3) if v = 30mV, then: v <- c; u <- u+d
where v = membrane potential, u = recovery variable,
a = time scale of recovery, b = sensitivity of recovery to subthreshold oscillations
c = after-spike reset value of v, d = after-spike reset increment of u
"""

import numpy as np
from numpy import *
from numpy.random import *
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import time
import mdp
import tsne

fw = 11
fh = 8

# sparsify synaptic matrix by randomly setting sqrt(|S|) of them to zero
#for i in range(N):
#    S[randint(0,N-1),randint(0,N-1)] = 0

def timeStep(a,b,c,d,v,u,i,thresh=20,substeps=2):
    """ worker function for simulation. given parameters and current state variables, compute next ms """
    ### local variable initiation ###
    fired=v>thresh     # array of indices of spikes
    v1 = v             # next step of membrane potential v
    u1 = u             # next step of recovery variable u
    spikesum = sum(fired)
    
    ### Action potentials ###
    v[fired] = thresh
    v1[fired] = c[fired]                # reset the voltage of any neuron that fired to c
    u1[fired] = u[fired] + d[fired]     # reset the recovery variable of any neuron that fired to u_0 + d
    i1 = i + sum(S[:,fired],axis=1)    # sum spontanous thalamic input and weighted inputs from all other neurons
    
    ### Step forward ###
    for i in range(substeps):  # for numerical stability, execute at least two substeps per ms
        v1=v1+(1.0/substeps)*(0.04*(v1**2)+(5*v1)+140-u+i1) # 
        u1=u1+(1.0/substeps)*a*(b*v1-u1)
    return v1, u1, i1,spikesum

def simulate(a,b,c,d,v0,u0,length,verbose=True):
    """ input:
    - for each of N neurons: parameters a,b,c,d, initial voltage v0 and recovery variables u0
    - length of simulation in milliseconds
    - NxN synaptic weight matrix
    
    processing:
        - simulates network evolution with spike-timing-dependent plasticity
    
    output:
        - Nxlength matrix of membrane voltages v over time
        - Nxlength matrix of recovery variables u over time
        - Nxlength matrix of synaptic inputs i over time
    """
    vout=np.zeros((N,length),dtype=double)
    vout[:,0] = v                                          # initial voltages
    uout = np.zeros((N,length),dtype=double)
    uout[:,0] = u
    iout = np.zeros((N,length),dtype=double)               # synaptic input matrix
    iout[:,0] = np.concatenate((5*rand(Ne),2*rand(Ni)))  # random thalamic input
    
    t0 = time.clock()
    
    ## simulate
    for t in range(1,length):
        I = np.concatenate((5*rand(Ne),2*rand(Ni)))
        vout[:,t],uout[:,t],iout[:,t],spikes[t]=timeStep(a,b,c,d,vout[:,t-1],uout[:,t-1],I)   
        
        # report progress
        if verbose and t % 100 == 0:
            print("Simulated " + str(t) + "ms of braintime in " + str(time.clock()-t0) + "s of computer time.") 
    
    t1 = time.clock()
    print("Simulation took " + str((t1-t0)) + "s")
    return vout,uout,iout

### Plotting results and analysis ###
def plotSummaryResults():
    # membrane potentials v
    pl.figure(figsize=(fw,fh))
    pl.subplot(2, 1, 1)
    pl.imshow(vo[1:ceil(length/4),:])
    pl.title('Membrane potentials (mV)')
    pl.xlabel('time (ms)')
    pl.ylabel('Neuron ID')
    
    # trace of a random neuron
    lucky = randint(0,N-1)
    pl.subplot(2,1,2)
    pl.plot(vo[lucky,:])
    pl.title('Trace of lucky neuron ' + str(lucky))
    pl.xlabel('time (ms)')
    pl.ylabel('Membrane potential (mV)')
    #pl.show()
    pl.savefig(str(N) + "--" + "summaryResults.pdf")

def computePCAmap(odim=2,transpo=False):
    print("Computing low-dimensional embedding...")
    pcanode1 = mdp.nodes.PCANode(output_dim=odim)
    if transpo:
        pcanode1.train(transpose(vo))
    else:
        pcanode1.train(vo)
    pcanode1.stop_training()
    proj = pcanode1.get_projmatrix()
    expvar = pcanode1.explained_variance
    print("PCA explains "  + str(pcanode1.explained_variance) + " of the variance with " + str(len(proj[0,:])) + " dimensions.")
    return proj,expvar

def phasePlot(neuron=1):
    # trace of a chosen neuron + a phase-plot
    x = np.array(range(length))
    y = vo[neuron,:]
    z = np.gradient(y)
    fig = pl.figure(figsize=(fw,fh))
    ax = fig.gca(projection='3d')
    x = x
    ax.plot(x, y, z,alpha=1)
    ax.legend()
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membane Potential (mV)')
    ax.set_zlabel('dV/dt (mV/ms)')
    pl.title('Trajectory of neuron ' + str(neuron) + ' through phase space')
    #pl.show()
    pl.savefig(str(N) + "--" + "phaseplot.pdf")

def embedSystem(embedding,explained_variance):
    pl.figure(figsize=(fw,fh))
    pl.scatter(embedding[:,0],embedding[:,1],c=spikes,linewidths=0)
    pl.title('Two-dimensional embedding explains ' + str(round(explained_variance,3)*100) + '% of variance among observed system states')
    pl.xlabel('Principal component 1')
    pl.ylabel('Principal component 2')
    pl.tick_params(labelleft='off', labelbottom='off')
    pl.colorbar().set_label('Instantaneous spike rate (spikes/ms)')
    #pl.show()
    pl.savefig(str(N) + "--" + "2Dsystemembedding.pdf")
    
def embedIndividuals(embedding,explained_variance):
    pl.figure(figsize=(fw,fh))
    pl.scatter(embedding[:,0],embedding[:,1],c=d,linewidths=0)
    pl.title('Two-dimensional embedding explains ' + str(round(explained_variance,3)*100) + '% of variance among individual neurons')
    pl.xlabel('Principal component 1')
    pl.ylabel('Principal component 2')
    pl.tick_params(labelleft='off', labelbottom='off')
    pl.colorbar().set_label('Parameter d (after-spike increment value of recovery variable u)')
    #pl.show()
    pl.savefig(str(N) + "--" + "2Dindividualembedding.pdf")

def embeddedPhasePlot(embedding):
    # trace of a chosen neuron + a phase-plot
    x = np.array(range(len(embedding[:,0])))
    y = embedding[:,0]
    z = embedding[:,1]
    fig = pl.figure(figsize=(fw,fh))
    ax = fig.gca(projection='3d')
    x = x
    ax.plot(x, y, z,alpha=1)
    ax.legend()
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Principal Component 1')
    ax.set_zlabel('Principal Component 2')
    pl.title('Trajectory of the entire neuronal population through principal-component space')
    #pl.show()
    pl.savefig(str(N) + "--" + "3Dsystemembedding.pdf")

### t-SNE!

def embedSystemtsne(embedding):
    pl.figure()
    pl.scatter(embedding[:,0],embedding[:,1],c=range(len(embedding[:,0])),linewidths=0)
    pl.title('t-distributed stochastic neighbor embedding of observed system states')
    pl.tick_params(labelleft='off', labelbottom='off')
    pl.colorbar().set_label('Time (ms)')
    pl.show()
    
def embedIndividualstsne(embedding):
    pl.figure()
    pl.scatter(embedding[:,0],embedding[:,1],c=d,linewidths=0)
    pl.title('t-distributed stochastic neighbor embedding of all individual neurons')
    pl.tick_params(labelleft='off', labelbottom='off')
    pl.colorbar().set_label('Parameter d (after-spike increment value of recovery variable u)')
    pl.show()

def embeddedPhasePlottsne(embedding):
    # trace of a chosen neuron + a phase-plot
    x = np.array(range(len(embedding[:,0])))
    y = embedding[:,0]
    z = embedding[:,1]
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z,alpha=0.7,linewidths=0.1,c=range(len(embedding[:,0])))
    ax.legend()
    ax.set_xlabel('Time (ms)')
    pl.title('Trajectory of the entire neuronal population through t-SNE space')
    pl.show()


for i in range(10):
    ### Initialize simulation parameters ###
    print("Initializing simulation " + str(i) + "...")    
    ### Run an ensemble of simulations and save the results ###
    length = 2000    # simulation time in ms
    Ne = 40*i         # number of excitatory neurons
    Ni = 10*i         # number of inhibitory neurons
    N = Ne+Ni        # total number of neurons
    S = np.concatenate((0.5*rand(N,Ne), -0.9*rand(N,Ni)),axis=1)  #initial synaptic weights
    spikes = np.zeros(length)   # counter of spikes per time-step
    re=np.array(rand(Ne),dtype=double)  # uniformly distributed random doubles [0,1)
    ri=np.array(rand(Ni),dtype=double)  # uniformly distributed random doubles [0,1)
    
    a = np.concatenate((0.02+0.001*re,0.02+0.02*ri))     # a = time scale of recovery
    b = np.concatenate((0.2+0.001*re, 0.25-0.05*ri))     # b = sensitivity of recovery to subthreshold oscillations
    c = np.concatenate((-65+5*(re**2),-65+0.5*(ri**2)))  # c = after-spike reset value of v
    d = np.concatenate((8-6*(re**2),  2+0.5*(ri**2)))    # d = after-spike reset increment of u
    v = -65.0 * np.ones(N,dtype=double); u=b*v    # Initial values of u,v
    
    ### Perform simulation ###
    vo,uo,io = simulate(a,b,c,d,v,u,length,verbose=True)
    
    ### Save results ###
    vo.tofile(str(N) + "--" + "voltage.csv")
    uo.tofile(str(N) + "--" + "recovery.csv")
    io.tofile(str(N) + "--" + "synapticinputs.csv")

    # Plot summary results
    plotSummaryResults()

    # Plot an interactive 3D phase-space trajectory of a random neuron
    phasePlot(randint(0,N-1))
    
    # Compute and display a 2D embedding of individual neurons' behavior
    lowDimI,expVarI = computePCAmap(transpo=True)
    lowDimI.tofile(str(N) + "--" + "lowDimI.csv")
    embedIndividuals(lowDimI,expVarI)
    
    tsne=False

    # Compute and display a 2D embedding of overall system behavior
    lowDimS,expVarS = computePCAmap()
    embedSystem(lowDimS,expVarS)
    lowDimS.tofile(str(N) + "--" + "lowDimS.csv")

    # Plot an interactive 3D phase-space trajectory of the population through 2D principal-component space
    embeddedPhasePlot(lowDimS)
    
    ### alternately, use a non-parametric low-dimensionality embedding
    if tsne:
       # Preprocess
       lowishDimI,eI = computePCAmap(odim=50,transpo=True)
       lowishDimS,eS = computePCAmap(odim=50)

       tsneI = tsne.tsne(lowishDimI, 2, 50, 20.0)
       tsneI.tofile('tsneI' + str(int(10*time.clock()))+'.csv')
       embedIndividualstsne(tsneI)

       tsneS = tsne.tsne(lowishDimS, 2, 50, 20.0)
       tsneS.tofile('tsneS' + str(int(10*time.clock()))+'.csv')
       embedSystemtsne(tsneS)
       embeddedPhasePlottsne(tsneS)
