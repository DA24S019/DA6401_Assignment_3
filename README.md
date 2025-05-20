
# DA6401 Assignment 3 — Sequence-to-Sequence Models with and without Attention

This repository contains implementations of character-level sequence-to-sequence transliteration models for Indian languages using vanilla RNNs and RNNs with Bahdanau attention.

> 🔗 **GitHub Repo:** https://github.com/DA24S019/DA6401_Assignment_3  
> 📊 **W&B Report:** [View Report](https://api.wandb.ai/links/da24s019-indian-institute-of-technology-madras/rtftrqpr)

---

## 📁 Repository Structure

The repository is organized into two main subdirectories:

---

### 🔹 `Vanilla_RNN/`

Implements a basic RNN encoder-decoder model **without attention**.

📂 [Explore the Folder](https://github.com/DA24S019/DA6401_Assignment_3/tree/main/Vanilla_RNN)

#### Contents:
| File/Folder | Description |
|-------------|-------------|
| `downloading_dataset.py` | Script to download and set up the dataset. |
| `dataset.py` | Data preprocessing utilities. |
| `config.py` | Configuration for model parameters and paths. |
| `loader.py` | Loads preprocessed datasets for training and validation. |
| `model.py` | Implementation of Encoder, Decoder, and Seq2Seq classes. |
| `evaluate.py` | Functions for evaluating the model during/after training. |
| `pred.py` | Predicts on test data and saves results to `Prediction_vanilla/`. |
| `rnn.py` | Basic demonstration script for Question 1. |
| `sweep.py` | Defines W&B hyperparameter sweep configuration. |
| `train_sweep.py` | Training script integrated with W&B sweeps. |
| `Prediction_vanilla/` | Contains output predictions from the best vanilla model. |

📝 Example training notebook: `training_vanilla.ipynb`

---

### 🔹 `Attention_RNN/`

Implements an RNN encoder-decoder model **with Bahdanau Attention**.

📂 [Explore the Folder](https://github.com/DA24S019/DA6401_Assignment_3/tree/main/Attention_RNN)

#### Contents:
| File/Folder | Description |
|-------------|-------------|
| `downloading_dataset.py` | Script to download and set up the dataset. |
| `data.py` | Data preprocessing and loading. |
| `config.py` | Configuration for model parameters and paths. |
| `model.py` | Implementation of Encoder, Attention, Decoder, and Seq2Seq. |
| `pred.py` | Generates predictions and saves to `Predictions_Attention/`, with attention maps. |
| `attention_heatmaps.py` | Code to plot attention heatmaps for model interpretability. |
| `app.py` | Visualization app for interactive or static attention visualizations. |
| `sweep.py` | W&B sweep configuration. |
| `train_sweep.py` | Training script integrated with W&B sweeps. |

📝 Example training notebook: `training_bahdanau_attention_dummy.ipynb`

---

## ⚠️ Notes

- **Model files not included:**  
  Due to file size limits on GitHub, the following pretrained model files are not included:
  - `best_model.pth` (Vanilla)
  - `best_model_attention.pth` (Attention)

- **How to Proceed:**
  - Run the training scripts with W&B sweeps to regenerate the best model checkpoints.
  - After training, you can use `pred.py`, `attention_heatmaps.py`, and `app.py` to visualize predictions and attention maps.

---
# For dataset download : 

cd Vanilla_RNN

python downloading_dataset.py

# For dataset download : 

cd Attention_RNN

python downloading_dataset.py

# For sweeping The Vanilla RNN

cd Vanilla_RNN

python sweep.py

# For sweeping The Attention RNN

cd Attention_RNN

python sweep.py

# For vanilla model

cd Vanilla_RNN

python pred.py

# For attention model

cd Attention_RNN

python pred.py

python attention_heatmaps.py

python app.py  # if running the interactive visualization

