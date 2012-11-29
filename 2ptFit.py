#The first 2-point correlation function fit code ever
#The first Python code ever...
#By Yuzhi Liu on 11/14/2012.

import os,sys,string
import numpy as np
from collections import deque
import twoptfunction as twopt
from Bs2K2pt import params
#import gdev as gd
#from gdev import gdev,sqrt
#from lsqfit import GPrior,nonlinear_fit
import gvar as gv
import lsqfit
import pylab as plt

from PyGrace.grace import Grace

ens = 'l2464f21b676m005m050'
dir = '/home/yuzhi/concat/yuzhi/LL2pt/' + ens + '/'
#dir = '/home/yuzhi/analysis/yuzhi/2ptcorr/test/'
#filename= 'pi_d_d_m0.0050_m0.0336_t48_p000'
#filename= 'pi_d_d_m0.0050_m0.0336_t0_p000'
#file = open( dir + filename, 'r')

#Read the concatenated files.
ID = 'C005_0050_000'
if len(sys.argv) > 1:
    ID = sys.argv[1]
ens, mval, mom = ID.split('_')
pms = params(ens, mval, mom)
T = pms['T']
nE = 4
tmin = pms['LLdd'][0]
tmax = pms['LLdd'][1]

#for momentum in ['000','100','110','111','200']:
for momentum in ['000']:
    print momentum
    dataFold = []
    dataFoldDict = {}
    for tsource in pms['tsrcs']:
        print tsource
        filename = 'pi_d_d_m' + pms['mlval'] + '_m' + pms['msval'] + '_t' + str(tsource) + '_p' + str(momentum)
        print filename
        file = open( dir + filename, 'r')
    
#       Read data files.  
        data = twopt.read_concat_file(dir + filename,headerNum=4,blocksize=pms['T'])
#       Shift data by some time source.
        dataShift = twopt.shift(data,tsource)
#       Fold data. The output dataFold is a numpy array.
        dataFold.append(twopt.fold(dataShift))
        dataFoldDict[tsource] = twopt.foldDict(dataShift)

#   Sum over 4 different time sources. Need to be careful.
    sumed = np.array(dataFold[0]+dataFold[1]+dataFold[2]+dataFold[3])
#   Average over four time sources.
    dataAve =  np.array([x /4.0 for x in sumed]) 
    print dataAve[0][0]
#   Average over different gauge configurations
    mean = np.mean(dataAve,axis=0)
    error = np.std(dataAve,axis=0)
#   print('mean =',mean)
#    mean = gv.dataset.avg_data(dataAve)
#   Calculate covariance matrixs. rowvar=False: each column represents a variable, while the rows contain observations.
#   bias=False: default normalization is by (N - 1), where N is the number of observations given (unbiased estimate). 
#   If bias is 1, then normalization is by N. These values can be overridden by using the keyword ddof in numpy versions >= 1.5.
    cov = np.cov(dataAve,rowvar=False,bias=False)/len(dataAve)
    print cov[1]
#x = gv.gvar(mean, cov)
#print('x =',x)    
print('gv mean =',gv.dataset.avg_data(dataAve))
#gvdata = gv.gvar(dataAve)
print dataFold[0][0]
print dataShift['000036']
print dataFoldDict[0]['000036']



#Plot data with matplotlib
plt.xlabel('time')
plt.ylabel('2pts')
plt.title('2pts data for %s'%pms['ens'])
#for configID in dataShift.iterkeys():
#    plt.plot(dataShift[configID])
#plt.plot(dataFold[0][0])
x = []
y = []
yerr = []
for i in range(len(mean)):
    x.append(i)
    y.append(mean[i])
    yerr.append(error[i])
    
print('error =',yerr)
print x
print y
plt.errorbar(x, y, yerr=yerr, fmt='o')
plt.show()
#plt.savefig('2pts_raw_data_%s'%pms['ens'])





#print('xcov =',gv.evalcov(gv.gvar(dataAve)))    
#x=np.arange(tmin,tmax+1)
#t0 = 0
#catRange = np.arange(t0 +  tmin, t0+tmax+1)
#avg=np.array([dataAve[i,1] for i in catRange ])
#cov=np.array([ [cov[i,j] for j in catRange ] for i in catRange])

#prior = GPrior()

#prior['E'] = [ gdev(pms['E'][i], 1.0) for i in range(nE) ]
#prior['Z'] = [ gdev(pms['ZE'][i], 1.0) for i in range(nE) ]

#p0=None
#fit=nonlinear_fit(data=(x,avg,cov),fcn=twopt.f,p0=p0,prior=prior,svdcut=1.e-12,reltol=1.e-15,maxit=5000)
#print fit
#test git
