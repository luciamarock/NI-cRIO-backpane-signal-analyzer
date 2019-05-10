# NI-cRIO-backpane-signal-analyzer
NI cRIO backpane cseries modules signal analyzer

This is a signal analyzer for signals generated by National Instruments cSeries IO modules for cRIO chassis.  The Python program estimates a few parameters such as: 

signal amplitude
fft bins length
frequency of the peak
amplitude of the frequency peak
ENOB

Code needs a textual input file for working. In my case the input file where generated directly from a LabVIEW vi for data acquisition of the cRIO IO chassis. 
