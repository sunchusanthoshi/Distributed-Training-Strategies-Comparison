# ⚙️ Comparative Study on Distributed Training Strategies

## 📌 Project Overview

This project benchmarks the **training efficiency of four distributed training strategies** using the **Fashion MNIST dataset**, with the goal of evaluating how different approaches affect performance and scalability. These strategies are:

1. Without MPI (Single-machine training)  
2. With MPI (Message Passing Interface-based parallelism)  
3. TensorFlow's Mirrored Strategy (multi-GPU synchronous training)  
4. Custom Data Parallelism (manual model and data distribution)
   
## 🧠 Objective

To analyze how distributed strategies affect training time and performance when applied to the same model architecture and dataset, and identify which is most efficient for deep learning tasks.

## 🧪 Dataset Description

- **Dataset:** [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- **Size:** 60,000 training images and 10,000 test images  
- **Classes:** 10 clothing categories (e.g., T-shirt/top, Trouser, Coat, Sneaker)
- **Image size:** 28×28 grayscale  
- **Preprocessing:** Pixel normalization to [0, 1]

## ⚙️ Training Strategies Compared

### 1. Without MPI
- Model runs on a single machine with no parallelization  
- Sequential training using Keras with basic architecture  
- Baseline for comparing distributed approaches

### 2. With MPI
- Utilizes `mpi4py` for distributing data and training across nodes  
- Each node trains on a partitioned dataset  
- Uses MPI's collective operations to average results and sync parameters

### 3. Mirrored Strategy (TensorFlow)
- Native TensorFlow multi-GPU synchronous strategy  
- Replicates model across GPUs, each computing gradients  
- Gradients averaged and synchronized before weight update

### 4. Custom Data Parallelism
- Manual GPU assignment with `tf.function` and custom loops  
- Direct control over data/model split, gradient aggregation  
- Most flexible but complex to implement


## 📈 Results Summary

| Strategy                 | Training Time (sec) |
|--------------------------|---------------------|
| Without MPI              | 46.18               |
| With MPI                 | 35.49               |
| Mirrored Strategy        | 29.21               |
| Custom Data Parallelism  | 17.46 ✅            |

✅ **Custom Data Parallelism** was the most efficient, cutting training time by over 60% compared to the baseline.


## 🔍 Observations

- Mirrored Strategy offered a solid balance of speed and ease of use  
- MPI brought moderate improvements, though more effective at larger scale  
- Custom parallelism performed best, but required significant manual setup  
- Due to Fashion MNIST's small size, training time differences were modest

## 🛠 Tools & Frameworks

- Python 3.x  
- TensorFlow 2.x / Keras  
- `mpi4py` for MPI training  
- Jupyter Notebook  
- NumPy, Matplotlib
