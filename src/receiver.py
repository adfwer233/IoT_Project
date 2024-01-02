import signal_processing
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

_, signal = wav.read("sender.wav")

def get_preamble_signal() -> np.array:

    preamble_string = "0101010101"
    preamble_binary = [eval(c) for c in preamble_string]

    return signal_processing.FSK_modulation(preamble_binary, signal_processing.fsk_config)
    

preamble_signal = get_preamble_signal()

print(len(preamble_signal))

signal = signal.astype(np.float64) / 1000


# print(temp.argmax())

config = signal_processing.bpsk_config

# signal = sig.lfilter(
#     *sig.iirfilter(
#         1, config.signal_freq + 3000,
#         btype="lowpass",
#         fs=config.sampling_freq
#     ),
#     signal
# )

# signal = sig.lfilter(
#     *sig.iirfilter(
#         1, config.signal_freq - 2000,
#         btype="highpass",
#         fs=config.sampling_freq
#     ),
#     signal
# )


temp = np.convolve(signal, np.flip(preamble_signal))[len(preamble_signal):]

plt.plot(signal)

plt.plot(temp / temp.max())

bit_len = len(preamble_signal) // 10
start_pos = temp.argmax()

print(start_pos, bit_len, len(signal), start_pos + bit_len * 18)

preamble_header_signal = signal[start_pos + bit_len: start_pos + bit_len * 50]

result, ip = signal_processing.FSK_modulation(preamble_header_signal, signal_processing.fsk_config)

print(result)

print(len(result))

print(ip)
plt.show()
