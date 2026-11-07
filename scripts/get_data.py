import os
import numpy as np

def get_16qam_constellation():
    points = []
    for i in [-3.0, -1.0, 1.0, 3.0]:
        for q in [-3.0, -1.0, 1.0, 3.0]:
            points.append(complex(i, q))
    constellation = np.array(points, dtype=np.complex128)
    constellation = constellation / np.sqrt(10.0)
    return constellation

def apply_saleh_pa(s_in, alpha_a=2.0, beta_a=1.0, alpha_phi=np.pi/3.0, beta_phi=1.0):
    r = np.abs(s_in)
    theta = np.angle(s_in)
    
    amp = (alpha_a * r) / (1.0 + beta_a * (r ** 2))
    phase = theta + (alpha_phi * (r ** 2)) / (1.0 + beta_phi * (r ** 2))
    
    return amp * np.exp(1j * phase)

def generate_samples(num_samples, snr_db, seed=None, input_scale=0.7):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    constellation = get_16qam_constellation()
    labels = rng.integers(0, 16, size=num_samples)
    s_norm = constellation[labels]
    
    s_in = input_scale * s_norm
    s_pa = apply_saleh_pa(s_in)
    
    sigma = np.sqrt(1.0 / (2.0 * (10.0 ** (snr_db / 10.0))))
    noise = sigma * (rng.normal(0, 1, size=num_samples) + 1j * rng.normal(0, 1, size=num_samples))
    r_recv = s_pa + noise
    
    X = np.column_stack((r_recv.real, r_recv.imag)).astype(np.float32)
    y = labels.astype(np.int64)
    return X, y

def main():
    os.makedirs("data", exist_ok=True)
    
    # Generate train and validation sets at 18 dB
    X_train, y_train = generate_samples(20000, snr_db=18.0, seed=42)
    X_val, y_val = generate_samples(5000, snr_db=18.0, seed=142)
    
    np.savez_compressed("data/train.npz", X=X_train, y=y_train)
    np.savez_compressed("data/val.npz", X=X_val, y=y_val)
    print(f"Generated train set: {X_train.shape}, val set: {X_val.shape}")

if __name__ == "__main__":
    main()
