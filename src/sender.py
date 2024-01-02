import numpy as np
import scipy.io.wavfile as wav
import signal_processing
from typing import List

def generate_packets(text: str) -> List[int] :

    if (len(text) > 30):
        print("generate_packets: text too long")
        return []

    preamble_string = "0101010101"
    preamble_binary = [eval(c) for c in preamble_string]

    # convert text to binary
    binary_string = [bin(ord(c))[2:] for c in text]
    binary = []
    for binary_string_of_char in binary_string:
        binary += [0] * (8 - len(binary_string_of_char))
        binary += [eval(c) for c in binary_string_of_char]

    text_length_binary_string = bin(len(text))[2:]
    text_length_binary = [0] * (8 - len(text_length_binary_string)) + [eval(c) for c in text_length_binary_string]
    
    return np.array(preamble_binary + text_length_binary + binary)

pack = generate_packets("test")

signal = signal_processing.FSK_modulation(pack, signal_processing.fsk_config) * 1000
signal = signal.astype(np.int16)

wav.write("sender.wav", 48000, signal)