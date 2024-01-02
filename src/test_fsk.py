import signal_processing
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

_, signal = wav.read("sender.wav")

res = signal_processing.FSK_demodulation(signal, signal_processing.fsk_config)

print(res)