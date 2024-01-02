from dataclasses import dataclass
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import scipy.fft as fft
from typing import Callable, List

@dataclass
class BPSKConfig:
    sampling_freq: int
    signal_freq: int
    amplitude: float
    symbol_duration: float

bpsk_config = BPSKConfig(48000, 20000, 1, 0.025)

def BPSK_modulation(data: np.ndarray, config: BPSKConfig) -> np.ndarray:
    time_line = np.arange(0, config.symbol_duration, 1 / config.sampling_freq)
    signal = config.amplitude * np.cos(2 * np.pi * config.signal_freq * time_line)
    signal = np.concatenate([signal if x == 0 else -signal for x in data])
    return signal

def BPSK_demodulation(signal: np.ndarray, config: BPSKConfig) -> np.ndarray:

    carrier_signal = np.arange(0, len(signal), 1) / config.sampling_freq
    carrier_signal = np.cos(2 * np.pi * config.signal_freq * carrier_signal)
    signal = signal * carrier_signal / config.amplitude

    signal = sig.lfilter(
        *sig.iirfilter(
            1, 1 / config.symbol_duration,
            btype="lowpass",
            fs=config.sampling_freq
        ),
        signal
    )

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