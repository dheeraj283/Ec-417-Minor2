# P15: Learned Demodulator for a Nonlinear Power Amplifier
> **Headline Result:** A compact 2-layer feed-forward neural network recovers distorted 16-QAM symbols under severe Saleh power-amplifier compression and phase rotation, reducing the symbol error rate from 58.1% (ideal grid baseline) down to 3.18% at 18 dB SNR, closely approaching the optimal theoretical ceiling (2.69%).

**Course:** EC-417 Introduction to Machine Intelligence  
**Institution:** Department of Electronics and Communication Engineering, National Institute of Technology Andhra Pradesh  
**Team Roll Numbers:** 623160, 623192  

---

## 1. Title & Executive Summary
In high-power wireless communications, pushing radio-frequency power amplifiers near saturation introduces severe nonlinear distortions (AM/AM amplitude flattening and AM/PM rotational warping). Conventional receivers assuming an undistorted grid break down completely. This project demonstrates that a lightweight feed-forward neural network (MLP) trained on received $(I, Q)$ coordinates successfully learns the warped decision regions, recovering near-optimal demodulation performance without requiring prior knowledge of the amplifier parameters.

---

## 2. Problem
* **Task:** Demodulate transmitted 16-QAM symbols from noisy, nonlinearly distorted baseband in-phase and quadrature $(I, Q)$ samples.
* **Why it matters:** Modern mobile transmitters operate power amplifiers near saturation to maximize electrical energy efficiency. However, severe nonlinearities warp the transmitted constellation. If the receiver cannot compensate for this distortion, high-order modulation schemes suffer catastrophic error rates.
* **Metric & Why:** Symbol Error Rate (SER), defined as the ratio of incorrectly detected symbols to total transmitted symbols:
  $$\text{SER} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i \neq y_i)$$
  SER directly quantifies the physical communication link quality across different Signal-to-Noise Ratios (SNR in dB).

---

## 3. Data
* **Source:** Synthetic generation using the frequency-independent Saleh power amplifier model followed by complex circular Additive White Gaussian Noise (AWGN).
* **Size:** 
  * Training set: 20,000 independent examples generated at 18 dB SNR.
  * Validation set: 5,000 independent examples generated at 18 dB SNR.
  * Test set: 50,000 independent symbols evaluated at each SNR point from 10 dB to 22 dB.
* **License:** Synthetically generated in Python; released under the MIT License.
* **Class Balance:** Perfectly uniform distribution across all 16 constellation points ($p(s_k) = 1/16$).
* **Split & Leak-Free Guarantee:** Training, validation, and test datasets are generated using distinct random number generator seeds (Seeds: 42 for train, 142 for validation, 242 for test). No noise realization or sample is shared across splits.

---

## 4. Method
We evaluate four distinct detectors under identical channel and noise conditions:

```
[Transmitted 16-QAM] --> [Saleh PA Distortion] --> [+ AWGN] --> [Received (I, Q)]
                                                                       |
           +-----------------------+-------------------+---------------+
           |                       |                   |               |
     [Baseline Grid]        [Known-PA]             [Main MLP]     [Polar MLP]
    (Assumes rigid grid) (Optimal Centroids)      (2->64->64->16) (4->64->64->16)
```

1. **Baseline (Ideal Grid Detector):** Finds the nearest symbol on the undistorted 16-QAM constellation scaled to match the received average power.
2. **Optimal Benchmark (Known-PA Centroids):** Computes minimum Euclidean distance to the 16 exact distorted cluster centroids. This represents the theoretical performance ceiling in AWGN.
3. **Main Model (Feed-Forward MLP):** A fully-connected neural network ($2 \to 64 \to 64 \to 16$) with ReLU activations and Cross-Entropy Loss, trained with the Adam optimizer ($\text{lr} = 0.003$).
4. **Improvement (Polar-Augmented MLP):** The same architecture expanded to 4 input dimensions: $(I, Q, |r|, \angle r)$, providing explicit magnitude and phase features.

---

## 5. Results

### Mandatory Results Table (Evaluated across 3 Seeds: 42, 43, 44)

| Model | SER @ 14 dB | SER @ 18 dB | SER @ 22 dB | Train Time |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Ideal Grid)** | $0.6057 \pm 0.0000$ | $0.5807 \pm 0.0000$ | $0.5853 \pm 0.0000$ | — |
| **Known-PA Centroids** | $0.1403 \pm 0.0000$ | $0.0269 \pm 0.0000$ | $0.0014 \pm 0.0000$ | — |
| **Main Model (MLP)** | $0.1439 \pm 0.0010$ | $0.0318 \pm 0.0004$ | $0.0029 \pm 0.0003$ | $4.18\text{ s}$ |
| **Improvement (Polar MLP)** | $0.1474 \pm 0.0022$ | $0.0363 \pm 0.0031$ | $0.0049 \pm 0.0017$ | $3.77\text{ s}$ |

### Key Figures

1. **SER vs. SNR Performance Curves (`results/figures/ser_vs_snr.png`):**  
   Shows the symbol error rate from 10 dB to 22 dB. The baseline hits a hard error floor ($\sim 58\%$), while the Main MLP closely tracks the Known-PA theoretical benchmark across the entire range.
2. **Learned Decision Regions (`results/figures/decision_regions.png`):**  
   Visualizes the curved, warped decision boundaries learned by the neural network in the $I\text{-}Q$ plane, cleanly matching the distorted pinwheel constellation.

---

## 6. Error Analysis
* **Why the Baseline Fails:** The Saleh amplifier causes a $+28.1^\circ$ phase shift and strong amplitude compression on outer symbols, but only $+5.4^\circ$ on inner symbols. Because the baseline assumes rigid square boundaries, outer points consistently fall into adjacent sectors, creating an inescapable error floor of $\approx 58\%$.
* **Where the MLP Makes Errors:** At low SNR (10–14 dB), errors occur almost exclusively between the middle ring ($r \approx 0.94$) and outer ring ($r \approx 1.00$). The radial distance between these two rings collapsed by $75.6\%$ due to AM/AM compression, causing their noise distributions to overlap heavily.
* **Why Polar Features Did Not Improve Accuracy:** While adding $(|r|, \angle r)$ accelerated training speed slightly ($3.77\text{ s}$ vs $4.18\text{ s}$), the $\text{atan2}(Q, I)$ function introduces an artificial branch cut discontinuity at $\pm \pi$. The standard Cartesian MLP smoothly parameterized the boundaries without this boundary artifact.

---

## 7. Limitations and Ethics
* **Limitations:** The simulation assumes a memoryless amplifier and static channel conditions. Real physical transmitters can exhibit memory effects (distortion dependent on past symbols) and carrier frequency offsets.
* **Leakage Risks:** To prevent synthetic data leakage, datasets for training, validation, and testing were strictly generated with independent seeds. No overlapping noise samples were present.
* **Misuse & Ethics:** This is an educational communications engineering benchmark. It does not process private user data or present ethical risks.

---

## 8. Reproducibility
The entire pipeline can be reproduced from a clean environment with the following commands:

```bash
# 1. Clone repository
git clone <repo-url>
cd ec417-miniproject-P15-623160-623192

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data
python scripts/get_data.py

# 4. Run evaluation and generate all results and figures
python -m src.evaluate
```
* **Hardware:** Standard Laptop CPU (Intel / AMD / Apple Silicon).
* **Runtime:** Data generation takes $\sim 2\text{ seconds}$; training and multi-seed evaluation take $\sim 15\text{ seconds}$ total.

---

## 9. AI-Use Statement
* **Tools Used:** Claude and Antigravity / Gemini assistant.
* **Scope of Use:** Assisted in code structuring, verifying mathematical equations for the Saleh model against standard literature, and drafting report sections.
* **Manual Verification:** All code implementations, data generation pipelines, model architectures, training runs, numerical calibrations, and interpretations were verified and executed by the team.

---

## 10. References
1. A. A. M. Saleh, "Frequency-independent and frequency-dependent nonlinear models of TWT amplifiers," *IEEE Transactions on Communications*, vol. 29, no. 11, pp. 1715–1720, Nov. 1981.
2. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008.
3. T. O'Shea and J. Hoydis, "An introduction to deep learning for the physical layer," *IEEE Transactions on Cognitive Communications and Networking*, vol. 3, no. 4, pp. 563–575, 2017.
