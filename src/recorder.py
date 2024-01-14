import scipy
import scipy.signal as sig
import numpy as np
import scipy.io.wavfile as wav
import ffmpeg
import matplotlib.pyplot as plt
import argparse
import pyaudio

def record(filename: str, duration: float, sample_frequency: int):
    print("start recording")
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_frequency,
        input=True,
        frames_per_buffer=1024
    )
    frames = []
    for _ in range(0, int(sample_frequency / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    result = np.frombuffer(b"".join(frames), dtype=np.int16)
    print("finished")

    wav.write(filename, sample_frequency, result)

record("test.wav", 45, sample_frequency=48000)