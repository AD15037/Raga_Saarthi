import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import periodogram, welch
from scipy.signal.windows import hamming, hann
from scipy.fft import fft, ifft

# ----------------------------- Functions --------------------------------

def load_audio(file_path):
    x, fs = librosa.load(file_path, sr=None, mono=True)
    return x, fs

def plot_waveform(x, fs):
    N = len(x)
    t1 = np.arange(N) / fs
    plt.figure()
    plt.plot(t1, x, color='r')
    plt.grid(True)
    plt.xlim([0, np.max(t1)])
    plt.ylim([-1.1*np.max(np.abs(x)), 1.1*np.max(np.abs(x))])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Signal in Time Domain')
    plt.show()

def plot_spectrogram(x, fs):
    plt.figure()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=1024, hop_length=256)), ref=np.max)
    librosa.display.specshow(D, sr=fs, hop_length=256, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.show()

def plot_periodogram(x, fs):
    f_pxx, pxx = periodogram(x, window=hamming(len(x)), fs=fs, scaling='density')
    plt.figure()
    plt.plot(f_pxx, 10 * np.log10(pxx))
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title('Periodogram')
    plt.show()

def plot_amplitude_spectrum(x, fs):
    w = hann(len(x))
    f_X, X = periodogram(x, fs=fs, window=w, scaling='spectrum')
    X_dB = 20 * np.log10(np.sqrt(X) * np.sqrt(2))
    plt.figure()
    plt.semilogx(f_X, X_dB, 'r')
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Amplitude Spectrum')
    plt.show()

def plot_histogram(x):
    plt.figure()
    plt.hist(x, bins=100, color='red', alpha=0.7)
    plt.grid(True)
    plt.xlabel('Amplitude')
    plt.ylabel('Number of Samples')
    plt.title('Signal Histogram')
    plt.show()

def compute_statistics(x):
    maxval = np.max(x)
    minval = np.min(x)
    mean_val = np.mean(x)
    rms_val = np.std(x)
    dynamic_range = 20 * np.log10(maxval / np.min(np.abs(x[np.nonzero(x)])))
    crest_factor = 20 * np.log10(maxval / rms_val)
    return maxval, minval, mean_val, rms_val, dynamic_range, crest_factor

def plot_autocorrelation(x, fs):
    N = len(x)
    Rx_full = np.correlate(x, x, mode='full')
    Rx = Rx_full / np.max(Rx_full)
    lags = np.arange(-N + 1, N)
    d = lags / fs

    plt.figure()
    plt.plot(d, Rx, color='r')
    plt.grid(True)
    plt.xlim([-np.max(np.abs(d)), np.max(np.abs(d))])
    plt.xlabel('Delay (s)')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation of the Signal')
    plt.axhline(y=0.05, color='k', linestyle='--', linewidth=2)
    plt.show()

    # Only look at positive lags (after center)
    center_index = N - 1
    positive_lags_Rx = Rx[center_index:]

    indices_above_threshold = np.where(positive_lags_Rx < 0.05)[0]
    if len(indices_above_threshold) == 0:
        RT = 0  # No crossing found
    else:
        RT = indices_above_threshold[0] / fs  # First time it drops below 0.05

    return RT


def plot_fft(x, fs):
    N = len(x)
    NFFT = 2 ** int(np.ceil(np.log2(N)))
    xf = np.abs(fft(x, NFFT))
    f = np.linspace(0, fs/2, int(NFFT/2) + 1)

    plt.figure()
    plt.plot(f, xf[:NFFT//2+1])
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Spectrum')
    plt.show()

def plot_psd_welch(x, fs):
    f_welch, Pxx_welch = welch(x, fs=fs)
    plt.figure()
    plt.semilogy(f_welch, Pxx_welch)
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title('Welch PSD Estimate')
    plt.show()

def plot_periodogram_fft(x, fs):
    N = len(x)
    xdft = fft(x)
    xdft = xdft[:N//2+1]
    psdx = (1/(fs*N)) * np.abs(xdft)**2
    psdx[1:-1] *= 2
    freq = np.linspace(0, fs/2, len(psdx))

    plt.figure()
    plt.plot(freq, 10 * np.log10(psdx))
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title('Periodogram Using FFT')
    plt.show()

def plot_normalized_frequency(x):
    N = len(x)
    xdft = fft(x)
    xdft = xdft[:N//2+1]
    psdx = (1/(2*np.pi*N)) * np.abs(xdft)**2
    psdx[1:-1] *= 2
    freq_norm = np.linspace(0, np.pi, len(psdx))

    plt.figure()
    plt.plot(freq_norm/np.pi, 10*np.log10(psdx))
    plt.grid(True)
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.ylabel('Power/Frequency (dB/rad/sample)')
    plt.title('Periodogram with Normalized Frequency')
    plt.show()

# --------------------------- Main Execution --------------------------------

if __name__ == "__main__":
    # Load
    x, fs = load_audio('bhairavi30.wav')        # Load the audio file

    # Plots and Analysis
    plot_waveform(x, fs)
    plot_spectrogram(x, fs)
    plot_periodogram(x, fs)
    plot_amplitude_spectrum(x, fs)
    plot_histogram(x)

    maxval, minval, mean_val, rms_val, dynamic_range, crest_factor = compute_statistics(x)
    print(f"Max value: {maxval}")
    print(f"Min value: {minval}")
    print(f"Mean value: {mean_val}")
    print(f"RMS value: {rms_val}")
    print(f"Dynamic Range: {dynamic_range:.2f} dB")
    print(f"Crest Factor: {crest_factor:.2f} dB")

    plot_fft(x, fs)
    plot_psd_welch(x, fs)
    plot_periodogram_fft(x, fs)
    plot_normalized_frequency(x)

    # autocorr_time = plot_autocorrelation(x, fs)
    # print(f"Autocorrelation Time: {autocorr_time:.4f} s")