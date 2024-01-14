from dataclasses import dataclass
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import scipy.fft as fft
from typing import Callable, List, Tuple

def DFT(signal: np.ndarray, sample_frequency: 48000) :
    r = 16
    n = len(signal)
        
    fft_signal = fft.fft(signal, r * n, norm="forward")
    fft_signal = fft.fftshift(fft_signal)
    fft_signal = np.abs(fft_signal)

    freq_axis = np.linspace(-r * n/2, r * n /2-1, r * n) * sample_frequency / (r * n)

    return freq_axis, fft_signal

@dataclass
class FSKConfig:
    sampling_freq: int
    signal_freq_low: int
    signal_freq_hight:int
    bit_per_symbol: int
    amplitude: float
    symbol_duration: float

fsk_config = FSKConfig(48000, 5000, 7000, 1, 1, 0.050)

def bit_freq_map(config: FSKConfig):
    '''
        bit_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    '''
    bit_list = []
    freq_list = list(range(config.signal_freq_low, config.signal_freq_hight, (config.signal_freq_hight - config.signal_freq_low) // (2 ** config.bit_per_symbol)))
    for i in range(2 ** config.bit_per_symbol):
        binary_string = bin(i)[2:]
        bit_list.append([0] * (config.bit_per_symbol - len(binary_string)) + [eval(c) for c in binary_string])
    return np.array(bit_list), np.array(freq_list)

def FSK_modulation(data: np.ndarray, config: FSKConfig) -> np.ndarray:
    bit_list, freq_list = bit_freq_map(config)
    time_line = np.arange(0, config.symbol_duration, 1 / config.sampling_freq)
    
    print(len(data))
    assert len(data) % (config.bit_per_symbol) == 0

    signal_list = []
    for i in range(0, len(data), (config.bit_per_symbol)):
        symbol_list = data[i: i + (config.bit_per_symbol)]
        symbol_val = eval("0b" + "".join([str(c) for c in symbol_list]))
        # print(freq_list, symbol_val, bit_list, data)
        signal_list.append(config.amplitude * np.cos(2 * np.pi * freq_list[symbol_val] * time_line))
    signal = np.concatenate(signal_list)
    return signal

def FSK_demodulation(signal: np.ndarray, config: FSKConfig) -> np.ndarray:
    # signal = sig.lfilter(
    #     *sig.iirfilter(
    #         1, 5500,
    #         btype="lowpass",
    #         fs=config.sampling_freq
    #     ),
    #     signal
    # )

    # signal = sig.lfilter(
    #     *sig.iirfilter(
    #         1, 2500,
    #         btype="highpass",
    #         fs=config.sampling_freq
    #     ),
    #     signal
    # )

    start, end = 0, len(signal)
    data_length = round((end - start) / config.sampling_freq / config.symbol_duration)

    bit_list, freq_list = bit_freq_map(config)
    time_line = np.arange(0, config.symbol_duration, 1 / config.sampling_freq)
    data = []
    for i in range(data_length):
        symbol_begin = int(i * config.symbol_duration * config.sampling_freq)
        symbol_end = min(end, symbol_begin + int(config.symbol_duration * config.sampling_freq))
        symbol = signal[symbol_begin: symbol_end]

        freq_axis, symbol_frequency_field = DFT(symbol, config.sampling_freq)
        symbol_frequency_field += symbol_frequency_field[::-1]
        symbol_frequency_field = symbol_frequency_field[len(symbol_frequency_field) // 2:]
        freq_axis = freq_axis[len(freq_axis) // 2:]

        freq = freq_axis[np.argmax(symbol_frequency_field)]

        # print(freq, symbol_begin, symbol_end, len(signal))

        # plt.cla()
        # plt.scatter(freq_axis, symbol_frequency_field)
        # plt.show()

        delta_freq_list = abs(np.array(freq_list) - freq)
        idx = delta_freq_list.argmin()
        for c in bit_list[idx]:
            data.append(c) 

    return data

@dataclass
class BPSKConfig:
    sampling_freq: int
    signal_freq: int
    amplitude: float
    symbol_duration: float

bpsk_config = BPSKConfig(48000, 7000, 1, 0.050)

def BPSK_modulation(data: np.ndarray, config: BPSKConfig) -> np.ndarray:
    time_line = np.arange(0, config.symbol_duration, 1 / config.sampling_freq)
    signal = config.amplitude * np.cos(2 * np.pi * config.signal_freq * time_line)
    signal = np.concatenate([signal if x == 0 else -signal for x in data])
    return signal

def BPSK_demodulation(signal: np.ndarray, config: BPSKConfig) -> np.ndarray:

    carrier_signal = np.arange(0, len(signal), 1) / config.sampling_freq
    carrier_signal = np.cos(2 * np.pi * config.signal_freq * carrier_signal)
    signal = signal * carrier_signal / config.amplitude

    # signal = sig.lfilter(
    #     *sig.iirfilter(
    #         1, 1 / config.symbol_duration,
    #         btype="lowpass",
    #         fs=config.sampling_freq
    #     ),
    #     signal
    # )

    tmp = np.nonzero(abs(signal) > 0.25)

    start, end = np.min(tmp), np.max(tmp)
    signal = signal[start: end]

    data_length = round((end - start) / config.sampling_freq / config.symbol_duration)

    data = []
    for i in range(data_length):
        symbol_begin = int(i * config.symbol_duration * config.sampling_freq)
        symbol_end = min(end, symbol_begin + int(config.symbol_duration * config.sampling_freq))
        data.append(np.mean(signal[symbol_begin: symbol_end]))

    result = [0 if x > 0 else 1 for x in data]

    return result, signal