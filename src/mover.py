# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:31:39 2019

@author: mritch3
"""

import os, glob
import numpy as np, scipy.stats as st
from scipy.io import loadmat
p=.6 #training p

os.chdir('..') #get to root directory
home=os.getcwd()

x=loadmat(home + r'\data\data.mat')
c=x['c'][0,:]
ims=x['images'][0,:]
gt=x['gt'][0,:]
countTotals=x['countTotals'][0,:]
annotations=x['annotations'][0,:]

randVec=np.loadtxt(r'data/randvec.txt').astype(int)
#annotations=annotations[randVec]
nameList=[]


check=[]
check2=[]
ciOut=np.zeros((1474,5))
diffs=np.zeros((1474,4))
#%% make confidence intervals
for i in range(len(annotations)):
    a=annotations[i]
    a=np.sum(a,(0,1))
    if a.size>1:
        (low,high)=st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
        mean=np.mean(a)
        diffs[i,:]=a-mean
        stdev=np.std(a)
        ciOut[i,:]=np.asarray([low,high,c[i],mean,stdev])
    else:
        ciOut[i,:]=np.asarray([a,a,c[i],a,0])
        diffs[i,:]=0



sze=192
check=[]
# %% blur annotations, retaining same count
for i in range(len(gt)):
    check.append(sum(sum((gt[i]))))


# %% get means for all
meanIm=np.mean([x for x in ims])
sdevIm=np.std([x for x in ims])
meanGT=np.mean([x for x in gt])
sdevGt=np.std([x for x in gt])
maxGT=np.max([x for x in gt])


# %% deposit all
print('deposit all')
count=[0,0,0]

#chdir+delete
os.chdir(home + '/data')
if not os.path.exists('all'):
    os.mkdir('all')
os.chdir('all')
files = glob.glob('*')
for f in files:
    os.remove(f)


for i in range(len(c)):
    if c[i]==1:
        first='y-rats'
        count[0]=count[0]+1
    elif  c[i]==2:
        first='g-rats'
        count[1]=count[1]+1
    elif  c[i]==3:
        first='b-rats'
        count[2]=count[2]+1

    name=first+"_%03d"%count[c[i]-1]
    nameList.append(name)

    im_this=ims[i]
    gt_this=gt[i]

    #print(i/len(c))

# %% randomize along previously used vec
if True:

    c=c[randVec]
    ims=ims[randVec]
    gt=gt[randVec]
    nameList = [nameList[i] for i in randVec]
    ciOut=ciOut[randVec,:]
    countTotals=countTotals[randVec]
    annotations=annotations[randVec]

check=[]
# %% blur annotations, retaining same count
for i in range(len(gt)):
    check.append(sum(sum((gt[i]))))


# %% deposit test/train split

#set up split data
n=len(c)
trainC=c[:int(n*p)]
trainIms=ims[:int(n*p)]
trainGt=gt[:int(n*p)]
trainNameList=[nameList[i] for i in range(int(n*p))]
testC=c[int(n*p):]
testIms=ims[int(n*p):]
testGt=gt[int(n*p):]
testNameList=[nameList[i+int(n*p)] for i in range(int(1+n*(1-p)))]

# %%depost train
print('deposit train')
os.chdir('..')
if not os.path.exists('train'):
    os.mkdir('train')
os.chdir('train')
if not os.path.exists('image'):
    os.mkdir('image')
files = glob.glob('image/*')
for f in files:
    os.remove(f)
if not os.path.exists('label'):
    os.mkdir('label')
files = glob.glob('label/*')
for f in files:
    os.remove(f)

os.chdir('image')
for i in range(len(trainC)):

    name=trainNameList[i]
    im_this=trainIms[i]

    np.save(name + '.npy',im_this)
    #im = Image.fromarray(A)
    #im.save("your_file.jpeg")
    print(i/len(trainC))

os.chdir('../label')
for i in range(len(trainC)):

    name=trainNameList[i]
    gt_this=trainGt[i]

    np.save(name + '.npy', gt_this)
    #im = Image.fromarray(A)
    #im.save("your_file.jpeg")
    print(i/len(trainC))

os.chdir('..')
# %%depost test
print('deposit test')
os.chdir('..')
if not os.path.exists('test'):
    os.mkdir('test')
os.chdir('test')
if not os.path.exists('image'):
    os.mkdir('image')
files = glob.glob('image/*')
for f in files:
    os.remove(f)
if not os.path.exists('gt'):
    os.mkdir('gt')
files = glob.glob('gt/*')
for f in files:
    os.remove(f)
if not os.path.exists('pred'):
    os.mkdir('pred')
files = glob.glob('pred/*')
for f in files:
    os.remove(f)

os.chdir('image')
for i in range(len(testC)):

    name=testNameList[i]
    im_this=testIms[i]
    name='test_' + str('%03d' % i)
    np.save(name + '.npy',im_this)
    print(i/len(testC))

os.chdir('../gt')
for i in range(len(testC)):

    name=testNameList[i]
    gt_this=testGt[i]
    name='test_' + str('%03d' % i)
    np.save(name + '.npy' , gt_this)
    print(i/len(testC))

os.chdir('..')
