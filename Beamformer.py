import numpy as np
import librosa

c = 343
epsilon = 1e-3

def config_UCA(Nr: int, radius: float):
    """
    Nr: number of microphones
    radius: radius of 2D UCA

    Returns:
    mic_pos: positions of microphones
        np.array, shape (2, num_channels)
    """
    phi = np.arange(Nr)*2*np.pi/Nr
    mic_pos = radius*np.stack((np.cos(phi), np.sin(phi)))
    return mic_pos

def cal_delay(mic_pos: np.array, theta: float):
    """
    mic_pos: positions of microphones
        np.array, shape (2, num_channels)
    theta: direction of arrival
        float
    
    Returns:
    tau: time delay
        np.array, shape (num_channels,)
    """
    u = np.array([[np.cos(theta), np.sin(theta)]])
    tau = u.dot(mic_pos)[0]/c
    return tau

def cal_steering_vec(tau: float, f: float):
    """
    tau: time delay
        np.array, shape (num_channels,)
    f: frequency

    Returns:
    steering_vec: steering vector
        np.array, shape (num_channels, )
    """
    return np.exp(-2j*np.pi*f*tau)

class Beamformer:
    def __init__(self, 
                 mic_pos: np.array, 
                 sr: int, 
                 n_fft: int,
                 win_length: int,
                 hop_length: int):
        self.mic_pos = mic_pos
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.frequencies = np.arange(0, n_fft//2+1)/n_fft*float(sr)

class DSB_beamformer(Beamformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x: np.array, theta: float):
        """
        x: multichannel signal
            np.array, shape (num_channels, num_samples)
        theta: direction of arrival
            float

        Returns:
        x_hat: beamformed signal
            np.array, shape(1, num_samples)
        """
        tau = cal_delay(self.mic_pos, theta)
        
        X = librosa.stft(x,
                         n_fft = self.n_fft,
                         win_length = self.win_length,
                         hop_length = self.hop_length)    
    
        W = []
        for i, f in enumerate(self.frequencies):
            w = cal_steering_vec(tau, f)
            w = w / np.linalg.norm(w)
            W.append(w)
        W = np.stack(W)
        X = X.transpose(1, 0, 2)
        W = np.expand_dims(W, axis = 1)
        X_weighted = np.einsum('FNM, FMT -> FNT', W.conj(), X)
        X_weighted = X_weighted.transpose(1, 0, 2)
        x_weighted = np.real(librosa.istft(X_weighted, 
                                           win_length = self.win_length,
                                           hop_length = self.hop_length,
                                           n_fft = self.n_fft))
        return x_weighted


class MVDR_beamformer(Beamformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x: np.array, theta: float):
        """
        x: multichannel signal
            np.array, shape (num_channels, num_samples)
        theta: direction of arrival
            float

        Returns:
        x_hat: beamformed signal
            np.array, shape(1, num_samples)
        """
        tau = cal_delay(self.mic_pos, theta)
        
        X = librosa.stft(x,
                         n_fft = self.n_fft,
                         win_length = self.win_length,
                         hop_length = self.hop_length)    

        # calculate the spatial covariance matrix R
        X = X.transpose(1, 0, 2) # (num_freq_bins, num_channels, num_frames)
        X_conj = X.conj().transpose(0, 2, 1)
        R = np.einsum('fmc, fcn -> fmn', X, X_conj)/X.shape[2]
        #R = R + epsilon * np.eye(R.shape[-1])[None, :, :]
        
        # calculate steering vector
        S = []
        for i, f in enumerate(self.frequencies):
            s = cal_steering_vec(tau, f)
            s = s / np.linalg.norm(s)
            S.append(s)
        S = np.stack(S) # (num_freq_bins, num_channels)
        S = np.expand_dims(S, axis = -1)
        R_inv = np.linalg.pinv(R)
        R_inv_s = np.einsum('FmM, FMN -> FmN', R_inv, S)
        denominator = np.einsum('FNM, FMn -> FNn', S.conj().transpose(0, 2, 1), R_inv_s) + 1e-8
        W = R_inv_s/denominator       
        X_weighted = np.einsum('FNM, FMT -> FNT', W.transpose(0, 2, 1).conj(), X)
        X_weighted = X_weighted.transpose(1, 0, 2)
        x_weighted = np.real(librosa.istft(X_weighted, 
                                           win_length = self.win_length,
                                           hop_length = self.hop_length,
                                           n_fft = self.n_fft))
        return x_weighted

    def call(self, x: np.array, theta: float):
        """
        x: multichannel signal
            np.array, shape (num_channels, num_samples)
        theta: direction of arrival
            float

        Returns:
        x_hat: beamformed signal
            np.array, shape(1, num_samples)
        """
        tau = cal_delay(self.mic_pos, theta)
        
        X = librosa.stft(x,
                         n_fft = self.n_fft,
                         win_length = self.win_length,
                         hop_length = self.hop_length)    

        # calculate the spatial covariance matrix R
        X = X.transpose(1, 0, 2) # (num_freq_bins, num_channels, num_frames)
        X_conj = X.conj().transpose(0, 2, 1)
        R = np.einsum('fmc, fcn -> fmn', X, X_conj)/X.shape[2]
        #R = R + epsilon * np.eye(R.shape[-1])[None, :, :]
        
        # calculate steering vector
        S = []
        for i, f in enumerate(self.frequencies):
            s = cal_steering_vec(tau, f)
            s = s / np.linalg.norm(s)
            S.append(s)
        S = np.stack(S) # (num_freq_bins, num_channels)
        S = np.expand_dims(S, axis = -1)
        
        W_list = []
        for f_idx in range(len(self.frequencies)):
            Rinv = np.linalg.pinv(R[f_idx]+1e-2 * np.eye(R.shape[1]))
            s = S[f_idx].reshape(-1, 1)
            w = (Rinv @ s)/(s.conj().T @ Rinv @ s+1e-9)
            W_list.append(w)  # shape (M,)

        # Stack weights: (F, M, 1)
        W = np.stack(W_list, axis=0)

        #R_inv = np.linalg.pinv(R)
        #R_inv_s = np.einsum('FmM, FMN -> FmN', R_inv, S)
        #denominator = np.einsum('FNM, FMn -> FNn', S.conj().transpose(0, 2, 1), R_inv_s) + 1e-8
        #W = R_inv_s/denominator       
        X_weighted = np.einsum('FNM, FMT -> FNT', W.transpose(0, 2, 1).conj(), X)
        X_weighted = X_weighted.transpose(1, 0, 2)
        x_weighted = np.real(librosa.istft(X_weighted, 
                                           win_length = self.win_length,
                                           hop_length = self.hop_length,
                                           n_fft = self.n_fft))
        return x_weighted
    
    def DOA(self, x: np.array):
        X = librosa.stft(x,
                         n_fft = self.n_fft,
                         win_length = self.win_length,
                         hop_length = self.hop_length)    

        # calculate the spatial covariance matrix R
        X = X.transpose(1, 0, 2) # (num_freq_bins, num_channels, num_frames)
        X_conj = X.conj().transpose(0, 2, 1)
        R = np.einsum('fmc, fcn -> fmn', X, X_conj)/X.shape[2]

        theta_scan = np.linspace(0, 2*np.pi, 100)
        
        powers = []
        #f_idx_valid = np.where((self.frequencies > 300) & (self.frequencies < 3500))[0]

        for theta in theta_scan:
            tau = cal_delay(self.mic_pos, theta)
            power = 0
            for i, f in enumerate(self.frequencies):
                s = cal_steering_vec(tau, f)
                s = s / np.linalg.norm(s)
                s = s.reshape(-1, 1)  # (M,1)
                Rinv = np.linalg.pinv(R[i] + 1e-6 * np.eye(R.shape[1]))
                denom = np.real(s.conj().T @ Rinv @ s) + 1e-9
                power += 10*np.log10(1. / denom)
            powers.append(power.squeeze() / len(self.frequencies))
        return powers

    def beamforming_with_null(self, x: np.array, theta: float, null_thetas: list = []):
        """
        x: multichannel signal
            np.array, shape (num_channels, num_samples)
        theta: direction of arrival
            float

        Returns:
        x_hat: beamformed signal
            np.array, shape(1, num_samples)
        """
        tau_target = cal_delay(self.mic_pos, theta)
        tau_interferers = [cal_delay(self.mic_pos, th) for th in null_thetas]

        # STFT (num_channels, num_samples) -> (num_freq_bins, num_channels, num_frames)
        X = librosa.stft(x,
                         n_fft = self.n_fft,
                         win_length = self.win_length,
                         hop_length = self.hop_length)
        X = X.transpose(1, 0, 2)
        X_conj = X.conj().transpose(0, 2, 1)

        # Spatial covariance matrix R: shape (F, M, M)
        R = np.einsum('fmc, fcn -> fmn', X, X_conj)/X.shape[2]

        # Calculate steering vectors (target + interferers)
        S_list = []
        for f_idx, f in enumerate(self.frequencies):
            s_target = cal_steering_vec(tau_target, f)
            s_target = s_target / np.linalg.norm(s_target)
            s_list = [s_target]

            for tau_i in tau_interferers:
                s_i = cal_steering_vec(tau_i, f)
                s_i = s_i / np.linalg.norm(s_i)
                s_list.append(s_i)

            S_list.append(np.stack(s_list, axis=1))  # (M, K+1)

        # Stack over frequency → shape (F, M, K+1)
        S = np.stack(S_list, axis=0)
        f_vec = np.zeros((len(null_thetas)+1, 1), dtype=np.complex64)
        f_vec[0, :] = 1.0  # gain at target

        # Compute LCMV beamforming weights
        W_list = []
        for f_idx in range(len(self.frequencies)):
            C = S[f_idx]                     # (M, K+1)
            Rf_inv = np.linalg.pinv(R[f_idx])
    
            inv_term = np.linalg.pinv(C.conj().T @ Rf_inv @ C)
            w = Rf_inv @ C @ inv_term @ f_vec
            W_list.append(w)  # shape (M,)

        # Stack weights: (F, M, 1)
        W = np.stack(W_list, axis=0)

        # Apply beamforming
        X_weighted = np.einsum('FNM, FMT -> FNT', W.transpose(0, 2, 1).conj(), X)
        X_weighted = X_weighted.transpose(1, 0, 2)
        x_weighted = np.real(librosa.istft(X_weighted,
                                           win_length=self.win_length,
                                           hop_length=self.hop_length,
                                           n_fft=self.n_fft))
        return x_weighted