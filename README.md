# 🧠 Golang Mini Neural Network (MNIST)

A minimal neural network built **from scratch in Go**, featuring a **computational graph** for gradient calculation. Trained on the **MNIST dataset**, it recognizes **handwritten digits (0–9)** using a **self‑implemented forward and backward pass from scratch** — inspired by [Andrej Karpathy’s *micrograd*](https://www.youtube.com/watch?v=VMj-3S1tku0).

---

## 🚀 Overview

- **Input layer:** 784 neurons (28×28 pixels)
- **Hidden layer:** 16 neurons
- **Output layer:** 10 neurons (digits 0–9)
- **Accuracy:** ~93% after 4 epochs

| Epoch |   Accuracy |   Loss |
| :---- | ---------: | -----: |
| 1     |      8.64% | 0.6443 |
| 2     |     90.87% | 0.3230 |
| 3     |     92.07% | 0.2882 |
| 4     | **92.82%** | 0.2684 |

---

## ⚙️ Run

```bash
git clone https://github.com/<NiklasWillecke>/go-ml.git
cd go-ml
go run main.go
```

---

## 📚 Reference

Inspired by [Andrej Karpathy – *Building micrograd from scratch*](https://www.youtube.com/watch?v=VMj-3S1tku0)