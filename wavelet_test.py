import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Load signal from CSV file
def load_signal(file_path):
    # Assuming the CSV has a single column named 'signal' or just values
    df = pd.read_csv(file_path)
    if 'signal' in df.columns:
        signal = df['signal'].values
    else:
        signal = df.iloc[:, 7].values
    return signal


# Apply Wavelet Packet Transform and flatten coefficients
def wavelet_packet_transform(signal, wavelet='db1', maxlevel=3):
    # Create Wavelet Packet decomposition
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)

    # Collect coefficients in all nodes at the maximum level
    nodes = wp.get_level(wp.maxlevel, order='natural')
    flattened_coeffs = np.concatenate([node.data for node in nodes])

    return wp, flattened_coeffs


# Reconstruct the signal from flattened coefficients
def reconstruct_signal(wp, flattened_coeffs):
    # Allocate coefficients back to each node
    nodes = wp.get_level(wp.maxlevel, order='natural')
    start = 0
    for node in nodes:
        end = start + len(node.data)
        node.data = flattened_coeffs[start:end]
        start = end

    # Reconstruct the signal from modified Wavelet Packet structure
    reconstructed_signal = wp.reconstruct(update=True)
    return reconstructed_signal


# Full process: Load, Transform, Flatten, Reconstruct
def process_wavelet_packet(file_path, wavelet='db1', maxlevel=3):
    # Step 1: Load signal
    signal = load_signal(file_path)
     # Step 2: Apply Wavelet Packet Transform and flatten coefficients
    wp, flattened_coeffs = wavelet_packet_transform(signal, wavelet=wavelet, maxlevel=maxlevel)
    # Step 3: Reconstruct the signal from flattened coefficients
    reconstructed_signal = reconstruct_signal(wp, flattened_coeffs)

    return signal, flattened_coeffs, reconstructed_signal


# Example usage
file_path = 'output/combined_dataset.csv'  # Replace with your file path
signal, flattened_coeffs, reconstructed_signal = process_wavelet_packet(file_path)

fig, axs = plt.subplots(2,1)
axs[0].plot(signal)
axs[1].plot(flattened_coeffs)
axs[1].plot(reconstructed_signal)
plt.show()



