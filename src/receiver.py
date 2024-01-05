import signal_processing
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

from typing import List

t, signal = wav.read("test.wav")

# print(signal.shape)

def get_preamble_signal() -> np.array:

    preamble_string = "0101010101"
    preamble_binary = [eval(c) for c in preamble_string]

    return signal_processing.BPSK_modulation(preamble_binary, signal_processing.bpsk_config)
    

preamble_signal = get_preamble_signal()

print(len(preamble_signal))

signal = signal.astype(np.float64) / 1000


# print(temp.argmax())

config = signal_processing.bpsk_config

signal = sig.lfilter(
    *sig.iirfilter(
        1, 7500,
        btype="lowpass",
        fs=config.sampling_freq
    ),
    signal
)

signal = sig.lfilter(
    *sig.iirfilter(
        1, 2500,
        btype="highpass",
        fs=config.sampling_freq
    ),
    signal
)

# temp = np.convolve(signal, np.flip(preamble_signal))[len(preamble_signal):]

temp = sig.correlate(signal, preamble_signal, mode="same")

plt.plot(signal)

plt.plot(temp / temp.max())

bit_len = len(preamble_signal) // 10

temp = temp / temp.max()

start_pos = temp.argmax() - len(preamble_signal) // 2

print(start_pos, bit_len, len(signal), start_pos + bit_len * 18)

preamble_header_signal = signal[start_pos + bit_len * 10: start_pos + bit_len * 18]

result = signal_processing.FSK_demodulation(preamble_header_signal, signal_processing.fsk_config)

header_binary_array = result

def binary_array_to_val(binary_array: List[int]):
    binary_string = '0b' + "".join([str(c) for c in binary_array])
    return eval(binary_string)

print(result)

print(len(result))

payload_length = binary_array_to_val(header_binary_array)

print(payload_length)

payload_result = signal_processing.FSK_demodulation(signal[start_pos + bit_len * 18: start_pos + bit_len * 18 + bit_len * (8 * payload_length)], signal_processing.fsk_config)

result_string = ""

print(payload_result)

for i in range(payload_length):
    result_string += chr(binary_array_to_val(payload_result[i * 8: 8 + i * 8]))

print(result_string)

plt.show()
