import numpy as np
from os import listdir
import matplotlib.pyplot as plt

EFile = open('ke.txt',mode='r')
XFile = open('kx.txt',mode='r')
  
# read all lines at once
EIds = EFile.read().split()
XIds = XFile.read().split()

print(EIds[0])

twenty=0
fourty=0
ESeries=np.zeros((2121,2400,3))
XSeries=np.zeros((1218,2400,3))
ECounter=0
XCounter=0
for f in listdir('waveforms'):
  if f in EIds:
      for z in listdir('waveforms/'+f):
            b = np.load('waveforms/'+f+'/'+z)
            if b['Z'].shape[0]==24001 and b['E'].shape[0]==24001 and b['N'].shape[0]==24001:
                  ESeries[ECounter,:,0]=np.mean(b['Z'][:24000].reshape(-1, 10), 1)               
                  ESeries[ECounter,:,1]=np.mean(b['E'][:24000].reshape(-1, 10), 1) 
                  ESeries[ECounter,:,2]=np.mean(b['N'][:24000].reshape(-1, 10), 1) 
                  ECounter+=1
            if b['Z'].shape[0]==48001 and b['E'].shape[0]==48001 and b['N'].shape[0]==48001:
                  ESeries[ECounter,:,0]=np.mean(b['Z'][:48000].reshape(-1, 20), 1)               
                  ESeries[ECounter,:,1]=np.mean(b['E'][:48000].reshape(-1, 20), 1) 
                  ESeries[ECounter,:,2]=np.mean(b['N'][:48000].reshape(-1, 20), 1) 
                  ECounter+=1
  if f in XIds:
      for z in listdir('waveforms/'+f):
            b = np.load('waveforms/'+f+'/'+z)
            if b['Z'].shape[0]==24001 and b['E'].shape[0]==24001 and b['N'].shape[0]==24001:
                  XSeries[XCounter,:,0]=np.mean(b['Z'][:24000].reshape(-1, 10), 1)               
                  XSeries[XCounter,:,1]=np.mean(b['E'][:24000].reshape(-1, 10), 1) 
                  XSeries[XCounter,:,2]=np.mean(b['N'][:24000].reshape(-1, 10), 1) 
                  XCounter+=1
            if b['Z'].shape[0]==48001 and b['E'].shape[0]==48001 and b['N'].shape[0]==48001:
                  XSeries[XCounter,:,0]=np.mean(b['Z'][:48000].reshape(-1, 20), 1)               
                  XSeries[XCounter,:,1]=np.mean(b['E'][:48000].reshape(-1, 20), 1) 
                  XSeries[XCounter,:,2]=np.mean(b['N'][:48000].reshape(-1, 20), 1) 
                  XCounter+=1
np.save('ESeries.npy' ,ESeries)
np.save('XSeries.npy' ,XSeries)
print(ECounter)
print(XCounter)
# close the file
EFile.close()
XFile.close()
