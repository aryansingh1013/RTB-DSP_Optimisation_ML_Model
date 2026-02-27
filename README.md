# RTB DSP Optimization Engine

This project implements a High-Frequency Real-Time Bidding (RTB) engine.
It consists of an offline **Python Training Pipeline** and an ultra-low-latency **Java Inference Engine**.

## Directory Structure

```
rtb_dsp_project/
├── python_training/
│   ├── train_models.ipynb  # Jupyter Notebook for Training & Weight Export
│   └── requirements.txt    # Python dependencies
├── java_inference/
│   ├── src/com/dtu/hackathon/bidding/Bid.java  # Main Bidding Logic
│   ├── build.sh            # Script to compile and package JAR
│   └── run.sh              # Script to run with JVM tuning flags
└── README.md
```

## 1. Python Training Workflow

The training pipeline is implemented in `python_training/train_models.ipynb`.

**Key Features:**
- **Polars**: Used for high-performance data loading and cleaning.
- **Hashing Trick**: Features are hashed into a $2^{20}$ dimensional space using a **Custom Zero-Allocation Hash** (compatible with Java).
- **Negative Downsampling**: Balances class distribution by sampling 10% of negatives.
- **FTRL/SGD**: Trains Logistic Regression models for CTR and CVR.
- **Weight Export**: Exports `ctr_weights.csv` and `cvr_weights.csv` for Java.

**Steps to Run:**
1.  Result directories are created automatically.
2.  Install dependencies: `pip install -r python_training/requirements.txt`.
3.  Open and run `train_models.ipynb`.
4.  Copy the generated `ctr_weights.csv` and `cvr_weights.csv` to `java_inference/` (or run from root).

## 2. Java Inference Engine

The online inference engine is strict Java 11 code designed for < 5ms latency.

**Key Features:**
- **Zero-Allocation**: No `new` objects created during `getBid()`.
- **Primitive Arrays**: Weights loaded into `double[]`.
- **PID Controller**: Dynamically paces budget using Proportional-Integral-Derivative control.
- **Recalibration**: Corrects predicted probabilities for negative downsampling.

**Compilation & Execution:**

1.  Navigate to `java_inference/`:
    ```bash
    cd java_inference
    ```
2.  Ensure `ctr_weights.csv` and `cvr_weights.csv` are present.
3.  Build the project:
    ```bash
    ./build.sh
    ```
4.  Run the engine:
    ```bash
    ./run.sh
    ```

**JVM Tuning:**
The `run.sh` script applies `-Xms512m -Xmx512m -XX:+UseSerialGC` to strictly enforce the 512MB memory limit and minimize GC pauses.
