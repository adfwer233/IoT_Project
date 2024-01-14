import signal_processing
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

from typing import List

import tkinter
import argparse

# print(signal.shape)

def parse_args():
    parser = argparse.ArgumentParser(description="receiver")
    parser.add_argument('--input',default="MoRWBjhpHBQHKPVSaHHvSKlWEdiTuRYsnYzFjLWipROQKKmktivddPnOGeBYfpzrbvVzPOhePjABPUOrhYnhYzQAxQSdYrRZZEdR")
    args = parser.parse_args()
    return args

def get_preamble_signal() -> np.array:

    preamble_string = "0101010101"
    preamble_binary = [eval(c) for c in preamble_string]

    return signal_processing.BPSK_modulation(preamble_binary, signal_processing.bpsk_config)

def generate_packets(text: str) -> List[int] :

    # convert text to binary
    binary_string = [bin(ord(c))[2:] for c in text]
    binary = []
    for binary_string_of_char in binary_string:
        binary += [0] * (8 - len(binary_string_of_char))
        binary += [eval(c) for c in binary_string_of_char]

    return np.array(binary)

def get_bit_error(result: np.ndarray, ans: np.array):
    return 1 - (result == ans).sum() / len(result)


if __name__ == "__main__":
    args = parse_args()
    t, signal = wav.read("test.wav")

    preamble_signal = get_preamble_signal()

    signal = signal.astype(np.float64) / 1000


    config = signal_processing.bpsk_config

    temp = sig.correlate(signal, preamble_signal, mode="same")

    plt.plot(signal)

    # plt.plot(temp / temp.max())

    bit_len = len(preamble_signal) // 10

    temp = temp / temp.max()

    start_pos = temp.argmax() - len(preamble_signal) // 2

    plt.axvline(start_pos, c = 'r')
    plt.axvline(start_pos + bit_len * 10, c = 'r')
    plt.show()


    preamble_length = 10
    slience_length = 1
    byte_length = 8 // signal_processing.fsk_config.bit_per_symbol

    preamble_length += slience_length

    preamble_header_signal = signal[start_pos + bit_len * preamble_length: start_pos + bit_len * (preamble_length + byte_length)]

    result = signal_processing.FSK_demodulation(preamble_header_signal, signal_processing.fsk_config)

    header_binary_array = result

    def binary_array_to_val(binary_array: List[int]):
        binary_string = '0b' + "".join([str(c) for c in binary_array])
        return eval(binary_string)

    payload_length = binary_array_to_val(header_binary_array)

    payload_length = 100

    payload_result = signal_processing.FSK_demodulation(signal[start_pos + bit_len * (preamble_length + byte_length): start_pos + bit_len * (preamble_length + byte_length) + bit_len * (byte_length * payload_length)], signal_processing.fsk_config)

    result_string = ""
    for i in range(payload_length):
        result_string += chr(binary_array_to_val(payload_result[i * 8: 8 + i * 8]))

    std_string = args.input
    bit_error = get_bit_error(payload_result, generate_packets(std_string))

    tkloop = tkinter.Tk()

    label_result = tkinter.Label(tkloop, text=f"Received String: {result_string}")
    label_bit_error = tkinter.Label(tkloop, text=f"Bit error rate: {bit_error}")

    label_result.pack()
    label_bit_error.pack()

    tkloop.mainloop()
