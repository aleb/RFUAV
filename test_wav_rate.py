import sys

from scipy.io import wavfile
import soundfile

for f in sys.argv:
    if f.endswith(".wav"):
        sampling_rate, _data = wavfile.read(f)
        info = soundfile.info(f)
        print(f, sampling_rate, info.samplerate, "" if (sampling_rate != info.samplerate) else "<" * 50)
        print(info)
        print()
