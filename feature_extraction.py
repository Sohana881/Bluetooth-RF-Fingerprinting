import numpy as np

from scipy.signal import hilbert, find_peaks
from scipy.stats import skew, kurtosis
def load_signal(file_path):
    """
    Reads a waveform text file and returns a NumPy array.
    """

    signal = np.loadtxt(file_path)

    signal = signal.astype(np.float32)

    return signal
def normalize_signal(signal):
    """
    Zero-mean and unit-variance normalization.
    """

    signal = signal - np.mean(signal)

    std = np.std(signal)

    if std != 0:
        signal = signal / std

    return signal
def generate_iq(signal):
    """
    Generate analytic IQ signal using Hilbert transform.
    """

    analytic_signal = hilbert(signal)

    I = np.real(analytic_signal)

    Q = np.imag(analytic_signal)

    return I, Q
def extract_features(I, Q):

    features = {}

    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.unwrap(np.arctan2(Q, I))
    energy = I**2 + Q**2

    fft = np.abs(np.fft.rfft(amplitude))
    fft_freq = np.fft.rfftfreq(len(amplitude))

    psd = fft**2
    psd = psd / (np.sum(psd) + 1e-12)

    # ===================================================
    # Amplitude Features
    # ===================================================

    features["amp_mean"] = np.mean(amplitude)
    features["amp_std"] = np.std(amplitude)
    features["amp_var"] = np.var(amplitude)
    features["amp_min"] = np.min(amplitude)
    features["amp_max"] = np.max(amplitude)
    features["amp_range"] = np.ptp(amplitude)
    features["amp_median"] = np.median(amplitude)
    features["amp_rms"] = np.sqrt(np.mean(amplitude**2))
    features["amp_skew"] = skew(amplitude)
    features["amp_kurtosis"] = kurtosis(amplitude)

    features["amp_q25"] = np.percentile(amplitude,25)
    features["amp_q75"] = np.percentile(amplitude,75)
    features["amp_iqr"] = features["amp_q75"]-features["amp_q25"]

    # ===================================================
    # Phase Features
    # ===================================================

    features["phase_mean"] = np.mean(phase)
    features["phase_std"] = np.std(phase)
    features["phase_var"] = np.var(phase)
    features["phase_range"] = np.ptp(phase)
    features["phase_skew"] = skew(phase)
    features["phase_kurtosis"] = kurtosis(phase)

    # ===================================================
    # I Features
    # ===================================================

    features["I_mean"] = np.mean(I)
    features["I_std"] = np.std(I)
    features["I_var"] = np.var(I)
    features["I_rms"] = np.sqrt(np.mean(I**2))
    features["I_skew"] = skew(I)
    features["I_kurtosis"] = kurtosis(I)

    # ===================================================
    # Q Features
    # ===================================================

    features["Q_mean"] = np.mean(Q)
    features["Q_std"] = np.std(Q)
    features["Q_var"] = np.var(Q)
    features["Q_rms"] = np.sqrt(np.mean(Q**2))
    features["Q_skew"] = skew(Q)
    features["Q_kurtosis"] = kurtosis(Q)

    # ===================================================
    # Energy
    # ===================================================

    features["energy_sum"] = np.sum(energy)
    features["energy_mean"] = np.mean(energy)
    features["energy_std"] = np.std(energy)
    features["energy_var"] = np.var(energy)
    features["energy_max"] = np.max(energy)

    # ===================================================
    # Crest Factor & PAPR
    # ===================================================

    rms = np.sqrt(np.mean(amplitude**2))

    features["crest_factor"] = np.max(amplitude)/(rms+1e-12)

    power = amplitude**2

    features["papr"] = np.max(power)/(np.mean(power)+1e-12)

    # ===================================================
    # Zero Crossing Rate
    # ===================================================

    features["zcr"] = np.mean(np.diff(np.sign(amplitude))!=0)

    # ===================================================
    # Spectral Features
    # ===================================================

    features["fft_mean"] = np.mean(fft)
    features["fft_std"] = np.std(fft)
    features["fft_max"] = np.max(fft)
    features["fft_energy"] = np.sum(fft**2)

    features["spectral_entropy"] = -np.sum(psd*np.log2(psd+1e-12))

    centroid = np.sum(fft_freq*fft)/(np.sum(fft)+1e-12)

    features["spectral_centroid"] = centroid

    bandwidth = np.sqrt(
        np.sum(((fft_freq-centroid)**2)*fft)/(np.sum(fft)+1e-12)
    )

    features["spectral_bandwidth"] = bandwidth

    cumulative = np.cumsum(psd)

    features["spectral_rolloff"] = fft_freq[np.where(cumulative>=0.85)[0][0]]

    geometric = np.exp(np.mean(np.log(fft+1e-12)))
    arithmetic = np.mean(fft)

    features["spectral_flatness"] = geometric/(arithmetic+1e-12)

    # ===================================================
    # Hjorth Parameters
    # ===================================================

    d1 = np.diff(amplitude)
    d2 = np.diff(d1)

    var0 = np.var(amplitude)
    var1 = np.var(d1)
    var2 = np.var(d2)

    features["hjorth_activity"] = var0

    features["hjorth_mobility"] = np.sqrt(var1/(var0+1e-12))

    features["hjorth_complexity"] = (
        np.sqrt(var2/(var1+1e-12))
        /(features["hjorth_mobility"]+1e-12)
    )

    # ===================================================
    # Peak Features
    # ===================================================

    peaks,_ = find_peaks(amplitude)

    features["num_peaks"] = len(peaks)

    if len(peaks)>0:

        features["peak_mean"] = np.mean(amplitude[peaks])

        features["peak_std"] = np.std(amplitude[peaks])

    else:

        features["peak_mean"] = 0

        features["peak_std"] = 0

    # ===================================================
    # Autocorrelation
    # ===================================================

    ac = np.correlate(amplitude,amplitude,mode="full")

    ac = ac[len(ac)//2:]

    features["autocorr_lag1"] = ac[1]

    features["autocorr_mean"] = np.mean(ac)

    features["autocorr_std"] = np.std(ac)

    return features