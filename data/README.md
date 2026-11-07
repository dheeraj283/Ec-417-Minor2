# Dataset Documentation — Synthetic Saleh 16-QAM

## Source & Generation
This project uses synthetically generated digital communication data. It simulates a 16-QAM transmitter passing through a nonlinear Radio-Frequency (RF) Power Amplifier (PA) governed by the Saleh model, followed by Additive White Gaussian Noise (AWGN).

* **Source:** Synthetic generation (`scripts/get_data.py`)
* **Reference:** A. A. M. Saleh, "Frequency-independent and frequency-dependent nonlinear models of TWT amplifiers," IEEE Transactions on Communications, 1981.
* **Access Date:** November 2026

## Configuration & Parameters
* **Constellation:** 16-QAM on grid $I, Q \in \{-3, -1, 1, 3\}$, normalized to average symbol energy $E_s = 1.0$ by dividing by $\sqrt{10}$.
* **Transmitter Scaling:** Input scale $\gamma_{in} = 0.7$, driving the amplifier into its nonlinear compression and phase-shift region.
* **Saleh Model Parameters:**
  * AM/AM amplitude compression: $A(r) = \frac{\alpha_a r}{1 + \beta_a r^2}$ with $\alpha_a = 2.0$, $\beta_a = 1.0$.
  * AM/PM phase rotation: $\Phi(r) = \frac{\alpha_\phi r^2}{1 + \beta_\phi r^2}$ with $\alpha_\phi = \frac{\pi}{3}$, $\beta_\phi = 1.0$.
* **Noise Definition:** Complex circular AWGN $w \sim \mathcal{CN}(0, \sigma^2)$ where $\sigma = \sqrt{\frac{1.0}{2 \cdot 10^{\text{SNR}_{\text{dB}}/10}}}$.
* **Evaluated SNR Range:** 10 dB to 22 dB (with primary evaluation points at 14 dB, 18 dB, and 22 dB).

## Split Protocol & Seed Control
To prevent data leakage, training, validation, and test datasets are generated independently with distinct random number generator seeds:
* **Training set:** 20,000 training examples (Seed: 42)
* **Validation set:** 5,000 examples (Seed: 142)
* **Test set:** 50,000 examples per SNR level to measure low symbol error rates reliably (Seed: 242)

## Reproduction
To generate the dataset locally, run:
```bash
python scripts/get_data.py
```
Per course rules, dataset files are ignored by git and regenerated locally via this script.
