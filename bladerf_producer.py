import bladerf
from bladerf import _bladerf
import numpy as np
import os
import shutil

# Settings
N = 4096  # samples per block from sync_rx
SAMPLE_DTYPE = np.int16  # Use int16 for SC16_Q11
BLOCK_SIZE = 10_000_000  # write to file after 10M samples
OUTPUT_DIR = "iq_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open bladeRF
sdr = bladerf.BladeRF()
RX = 0  # RX module

# Configure RX
sdr.set_bandwidth(RX, 18e6)
sdr.set_frequency(RX, 2.41e9)
sdr.set_sample_rate(RX, 20e6)
sdr.gain_mode = _bladerf.GainMode.Manual
sdr.gain = 20
sdr.enable_module(RX, True)

# Configure synchronous streaming (without flags)
sdr.sync_config(
    layout=_bladerf.ChannelLayout.RX_X1,
    fmt=_bladerf.Format.SC16_Q11,
    num_buffers=16,
    buffer_size=N,
    num_transfers=8,
    stream_timeout=3500
)

print("Producer started. Saving 10M-sample blocks to raw .iq files...")

# Accumulation buffer (interleaved int16)
buffer_accum = np.empty(BLOCK_SIZE * 2, dtype=np.int16)  # I/Q interleaved
accum_index = 0
file_count = 1
tmp = "/tmp/tmp.iq"

try:
    while True:
        # Allocate buffer for one block of samples (interleaved I/Q)
        samples = np.empty(N * 2, dtype=np.int16)
        sdr.sync_rx(samples, N)

        end_index = accum_index + N * 2
        if end_index <= BLOCK_SIZE * 2:
            buffer_accum[accum_index:end_index] = samples
            accum_index = end_index
        else:
            # Fill remaining space
            remaining = BLOCK_SIZE * 2 - accum_index
            buffer_accum[accum_index:] = samples[:remaining]

            # Write raw IQ file
            buffer_accum.tofile(tmp)
            filename = os.path.join(OUTPUT_DIR, f"iq_block_{file_count}.iq")
            shutil.move(tmp, filename)
            print(f"Saved {filename}")
            file_count += 1

            # Start new buffer with leftover samples
            leftover = N * 2 - remaining
            buffer_accum[:leftover] = samples[remaining:]
            accum_index = leftover

except KeyboardInterrupt:
    print("Stopping producer...")

finally:
    sdr.enable_module(RX, False)
