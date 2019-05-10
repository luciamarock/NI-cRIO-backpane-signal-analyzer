import numpy as np
import math
from numpy import genfromtxt
#import matplotlib.pyplot as plt
#import scipy.fftpack

data=genfromtxt("TX100.xls") #THIS IS THE INPUT FILE acquired from teh cRIO backpane via LabVIEW vi 

col1=data[:,1]
col2=data[:,2]
col3=data[:,3]
col4=data[:,4]
col5=data[:,5]
col6=data[:,6]
col7=data[:,7]
col8=data[:,8]
col9=data[:,9]

concatenated=np.concatenate((col1,col2,col3,col4,col5,col6,col7,col8,col9))
#print('concatenated length = {}'.format(len(concatenated)))
#col0=data[:,1]
col0=concatenated

Fs=5000000 #THIS IS THE SAMPLING FREQUENCY OF THE cRIO IO MODULES 
#noe=975000
noe=len(concatenated)
conv_gain=0.00305382/2.
dinamics=16384. # 14 bit 

time = []
samples0=[]
#samples=[]
#samples2=[]

for i in range (noe):
    time.append(i)
    samples0.append((col0[i]-dinamics/2.)*conv_gain)
    
    #samples.append(col1[i])
    #samples2.append(col2[i])
print(' ')    
peak = max(samples0)
print('signal amplitude = {}'.format(peak))
spectra=np.fft.fft(np.array(samples0))
#print(len(spectra))
normamp=np.abs(spectra[:noe//2])*2./noe
#spectra[:noe//2] takes the first half of spectra

gaps=float(Fs)/noe
print('fft bins length = {}'.format(gaps))

freq=[]
amps=[]

for i in range (noe/2):
    amps.append(normamp[i])
    freq.append(gaps/2.+i*gaps)
    

amps.pop(0)
freq.pop(0)
fpeak = max(amps)
index = np.argmax(amps)
peaksvect=[0.]*len(amps)
peaksvect[0]=amps[0]

Aa=0
Ca=0
An=0
Cn=0
for i in range (len(amps)-2):
    if (amps[i+1]-amps[i])>0 and (amps[i+1]-amps[i+2])>0:
        peaksvect[i+1]=amps[i+1]
for i in range (len(peaksvect)):
    if i==(index*2) or i==(index*3+1) or i==(index*4+2) or i==(index*5+3) or i==(index*6+3) or i==(index*7+4):
        Aa=Aa+peaksvect[i]
        Ca=Ca+1        
    elif i!=index:
        An=An+amps[i]
        Cn=Cn+1
Ps=10.*math.log10(pow(fpeak,2)/2.)
#Aa=float(Aa/Ca)
An=float(An/Cn)
Pn=10.*math.log10(pow(An,2)/2.)
Pd=10.*math.log10(pow(Aa,2)/2.)
SINAD=(Ps+Pn+Pd)/(Pn+Pd)
ENOB=((SINAD-1.76)/6.02)*100 + 26
#ENOB=(SINAD-1.76)/6.02
print('frequency of the peak = {}'.format(freq[index]))
print('amplitude of the frequency peak = {}'.format(fpeak))
print('Aa = {} An = {}'.format(Aa, An))
print('ENOB = {}'.format(ENOB))
#UNCOMMENT THE FOLLOWING RAWS FOR PLOTTING (uncomment also the import for plt.)
'''
plt.figure()
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude (V)')
plt.semilogx(freq,peaksvect,'.')
plt.grid()
'''
print(' ')