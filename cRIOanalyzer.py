import numpy as np
import math
from numpy import genfromtxt
import logging

logging.basicConfig(level=logging.INFO)

def load_data(filename: str) -> np.ndarray:
    """
    Loads data from a CSV file containing cRIO backplane measurements and prepares it for analysis.

    Parameters:
    filename (str): The path to the CSV file containing the data.

    Returns:
    np.ndarray: A concatenated array of columns from the data file, suitable for analysis.
    """
    try:
        # Load data from CSV file containing cRIO backplane measurements
        data = genfromtxt(filename, skip_header=1, delimiter=',', dtype=None, encoding=None)
        
        # Extract columns from data array
        cols = [data[:, i] for i in range(1, 10)]
        
        # Concatenate all columns into single array for analysis
        return np.concatenate(cols)
    except FileNotFoundError:
        logging.error(f"File {filename} not found.")
        return np.array([])
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return np.array([])

def initialize_parameters():
    # Define acquisition parameters
    Fs = 5000000  # Sampling frequency of cRIO IO modules in Hz 
    conv_gain = 0.00305382 / 2.  # Conversion gain factor
    dinamics = 16384.  # Dynamic range for 14-bit system
    return Fs, conv_gain, dinamics

def convert_to_voltage(raw_data, dinamics, conv_gain):
    # Initialize arrays for time domain analysis using vectorization
    time = np.arange(len(raw_data))
    samples = (raw_data - dinamics / 2) * conv_gain
    return time, samples

def perform_fft(samples, Fs):
    noe = len(samples)
    spectra = np.fft.fft(np.array(samples))
    normamp = np.abs(spectra[:noe // 2]) * 2. / noe
    
    # Calculate frequency resolution
    gaps = float(Fs) / noe
    
    # Create frequency and amplitude arrays
    freq = [gaps / 2. + i * gaps for i in range(noe // 2)]
    amps = list(normamp)
    
    # Remove DC component
    amps.pop(0)
    freq.pop(0)
    
    return freq, amps, gaps

def find_peaks(amps):
    fpeak = max(amps)
    index = np.argmax(amps)
    peaksvect = [0.] * len(amps)
    peaksvect[0] = amps[0]
    
    # Detect peaks in spectrum
    for i in range(len(amps) - 2):
        if (amps[i + 1] - amps[i]) > 0 and (amps[i + 1] - amps[i + 2]) > 0:
            peaksvect[i + 1] = amps[i + 1]
            
    return fpeak, index, peaksvect

def calculate_signal_metrics(amps, peaksvect, index):
    Aa = 0  # Accumulator for harmonic amplitudes
    Ca = 0  # Counter for harmonics
    An = 0  # Accumulator for noise amplitudes
    Cn = 0  # Counter for noise components
    
    # Separate harmonics from noise
    for i in range(len(peaksvect)):
        if i == (index * 2) or i == (index * 3 + 1) or i == (index * 4 + 2) or i == (index * 5 + 3) or i == (index * 6 + 3) or i == (index * 7 + 4):
            Aa += peaksvect[i]
            Ca += 1        
        elif i != index:
            An += amps[i]
            Cn += 1
            
    # Avoid division by zero
    An = float(An / Cn) if Cn > 0 else 0.0  # Average noise amplitude
    return Aa, An

def calculate_power_metrics(fpeak, Aa, An):
    Ps = 10. * math.log10(pow(fpeak, 2) / 2.)  # Signal power
    Pn = 10. * math.log10(pow(An, 2) / 2.)  # Noise power
    Pd = 10. * math.log10(pow(Aa, 2) / 2.)  # Distortion power
    
    # Calculate SINAD and ENOB
    SINAD = (Ps + Pn + Pd) / (Pn + Pd)
    ENOB = ((SINAD - 1.76) / 6.02) * 100 + 26
    
    return ENOB

def main():
    # Load and process data
    raw_data = load_data("TX100.xls")
    Fs, conv_gain, dinamics = initialize_parameters()
    
    # Convert to voltage
    time, samples = convert_to_voltage(raw_data, dinamics, conv_gain)
    
    logging.info(' ')
    logging.info('Signal amplitude = {}'.format(max(samples)))
    
    # Perform FFT analysis
    freq, amps, gaps = perform_fft(samples, Fs)
    logging.info('FFT bins length = {}'.format(gaps))
    
    # Find peaks and calculate metrics
    fpeak, index, peaksvect = find_peaks(amps)
    Aa, An = calculate_signal_metrics(amps, peaksvect, index)
    ENOB = calculate_power_metrics(fpeak, Aa, An)
    
    # Log results
    logging.info('Frequency of the peak = {}'.format(freq[index]))
    logging.info('Amplitude of the frequency peak = {}'.format(fpeak))
    logging.info('Aa = {} An = {}'.format(Aa, An))
    logging.info('ENOB = {}'.format(ENOB))
    logging.info(' ')

if __name__ == "__main__":
    main()
