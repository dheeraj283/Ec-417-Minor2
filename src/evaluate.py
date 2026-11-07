import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure matplotlib uses writable config directory
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.expanduser("~"), ".matplotlib_cache")

from scripts.get_data import generate_samples
from src.data import add_polar_features
from src.models import IdealGridDetector, KnownPACentroidDetector
from src.train import train_model

def evaluate_detector(detector, snr_list, num_samples=50000, seed=242, is_polar=False):
    ser_dict = {}
    for snr in snr_list:
        X_test, y_test = generate_samples(num_samples, snr_db=snr, seed=seed)
        if is_polar:
            X_eval = add_polar_features(X_test)
        else:
            X_eval = X_test
        preds = detector.predict(X_eval)
        ser = float(np.mean(preds != y_test))
        ser_dict[snr] = ser
    return ser_dict

def plot_ser_curves(df_metrics, snr_list, save_path="results/figures/ser_vs_snr.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    colors = {
        "Baseline (Ideal Grid)": "#d62728",
        "Known-PA Centroids": "#2ca02c",
        "Main Model (MLP)": "#1f77b4",
        "Improvement (Polar MLP)": "#9467bd"
    }
    markers = {
        "Baseline (Ideal Grid)": "s",
        "Known-PA Centroids": "o",
        "Main Model (MLP)": "^",
        "Improvement (Polar MLP)": "d"
    }
    
    for model_name in ["Baseline (Ideal Grid)", "Known-PA Centroids", "Main Model (MLP)", "Improvement (Polar MLP)"]:
        sub = df_metrics[df_metrics["model"] == model_name]
        mean_sers = []
        std_sers = []
        for snr in snr_list:
            vals = sub[sub["snr"] == snr]["ser"].values
            mean_sers.append(np.mean(vals))
            std_sers.append(np.std(vals))
            
        mean_sers = np.array(mean_sers)
        std_sers = np.array(std_sers)
        
        plt.semilogy(snr_list, mean_sers, label=model_name, color=colors[model_name],
                     marker=markers[model_name], linewidth=2, markersize=6)
        if np.max(std_sers) > 0:
            plt.fill_between(snr_list, np.maximum(mean_sers - std_sers, 1e-5), mean_sers + std_sers,
                             color=colors[model_name], alpha=0.15)
            
    plt.title("Symbol Error Rate (SER) vs. SNR under Saleh PA Distortion", fontsize=12, pad=10)
    plt.xlabel("SNR (dB)", fontsize=11)
    plt.ylabel("Symbol Error Rate (SER)", fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.ylim(1e-4, 1.0)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved SER curve to {save_path}")

def plot_decision_regions(mlp_model, save_path="results/figures/decision_regions.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x_range = np.linspace(-1.5, 1.5, 300)
    y_range = np.linspace(-1.5, 1.5, 300)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    
    preds = mlp_model.predict(grid_points).reshape(xx.shape)
    
    # Also get true centroids to overlay
    known_pa = KnownPACentroidDetector()
    centroids = known_pa.centroids
    
    plt.figure(figsize=(7, 7))
    plt.contourf(xx, yy, preds, levels=np.arange(-0.5, 16.5, 1), cmap="tab20", alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="x", s=60, linewidths=2, label="PA Centroids")
    
    plt.title("Learned MLP Decision Regions in I-Q Plane", fontsize=12, pad=10)
    plt.xlabel("In-Phase (I)", fontsize=11)
    plt.ylabel("Quadrature (Q)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved decision regions to {save_path}")

def run_all_evaluations(seeds=[42, 43, 44]):
    snr_list = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
    os.makedirs("results/figures", exist_ok=True)
    
    records = []
    
    # 1. Baseline
    grid_det = IdealGridDetector()
    t0 = time.time()
    ser_grid = evaluate_detector(grid_det, snr_list, num_samples=50000, seed=242)
    t_grid = time.time() - t0
    for s in seeds:
        for snr, val in ser_grid.items():
            records.append({"model": "Baseline (Ideal Grid)", "seed": s, "snr": snr, "ser": val, "train_time": 0.0})
            
    # 2. Known-PA Centroids Benchmark
    pa_det = KnownPACentroidDetector()
    t0 = time.time()
    ser_pa = evaluate_detector(pa_det, snr_list, num_samples=50000, seed=242)
    t_pa = time.time() - t0
    for s in seeds:
        for snr, val in ser_pa.items():
            records.append({"model": "Known-PA Centroids", "seed": s, "snr": snr, "ser": val, "train_time": 0.0})
            
    # 3. Main Model (MLP) across seeds
    last_mlp = None
    for s in seeds:
        t0 = time.time()
        mlp = train_model("mlp", seed=s, epochs=15)
        train_time = time.time() - t0
        last_mlp = mlp
        ser_mlp = evaluate_detector(mlp, snr_list, num_samples=50000, seed=242)
        for snr, val in ser_mlp.items():
            records.append({"model": "Main Model (MLP)", "seed": s, "snr": snr, "ser": val, "train_time": train_time})
            
    # 4. Improvement (Polar MLP) across seeds
    for s in seeds:
        t0 = time.time()
        polar_mlp = train_model("polar_mlp", seed=s, epochs=15)
        train_time = time.time() - t0
        ser_pol = evaluate_detector(polar_mlp, snr_list, num_samples=50000, seed=242, is_polar=True)
        for snr, val in ser_pol.items():
            records.append({"model": "Improvement (Polar MLP)", "seed": s, "snr": snr, "ser": val, "train_time": train_time})
            
    df = pd.DataFrame(records)
    df.to_csv("results/metrics.csv", index=False)
    print("Saved all results to results/metrics.csv")
    
    # Generate Figures
    plot_ser_curves(df, snr_list)
    if last_mlp is not None:
        plot_decision_regions(last_mlp)
        
    # Print slide 30 mandatory results table for 14, 18, 22 dB
    print("\n" + "="*70)
    print("MANDATORY RESULTS TABLE (EC-417 Slide 30 Format)")
    print("="*70)
    print(f"{'Model':<26} | {'SER @ 14 dB':<16} | {'SER @ 18 dB':<16} | {'SER @ 22 dB':<16} | {'Train time'}")
    print("-" * 75)
    for model_name in ["Baseline (Ideal Grid)", "Known-PA Centroids", "Main Model (MLP)", "Improvement (Polar MLP)"]:
        sub = df[df["model"] == model_name]
        v14 = sub[sub["snr"] == 14.0]["ser"]
        v18 = sub[sub["snr"] == 18.0]["ser"]
        v22 = sub[sub["snr"] == 22.0]["ser"]
        t = sub["train_time"].mean()
        t_str = f"{t:.2f} s" if t > 0 else "-"
        print(f"{model_name:<26} | {v14.mean():.4f} +/- {v14.std():.4f} | {v18.mean():.4f} +/- {v18.std():.4f} | {v22.mean():.4f} +/- {v22.std():.4f} | {t_str}")
    print("="*70)

if __name__ == "__main__":
    run_all_evaluations()
