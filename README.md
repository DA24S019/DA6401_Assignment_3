Github Link : https://github.com/DA24S019/DA6401_Assignment_3
Wandb Link  Report : https://api.wandb.ai/links/da24s019-indian-institute-of-technology-madras/rtftrqpr

There are 2 files and 2 folder in the repo.
Folder name : Vanilla_RNN contains files related to vanilla rnn implementation.
Folder name : VAttention_RNN contains files related to rnn with attention implementation.

Following are the description of Vanilla_RNN folder: 
(Link : https://github.com/DA24S019/DA6401_Assignment_3/tree/main/Vanilla_RNN)

1 downloading_dataset.py  : This file download the dataset required for this assignment.//
2. dataset.py : It contains codes for datapreprocessing//
3. config.py : It contains configuration of the model
4. loader.py : it loads the data file for training and validation.
5. model.py : It contains the model artitechture. (Encoder, Decoder and Seq2Seq)
6. evaluate.py : It evaluate the data from trained model.
7. pred.py : It contains the code for prediction on my test data and saving the Predictions_Vanilla Folder.
8. rnn.py : This file is small demo for the Question 1.
9. sweep.py : This file contains sweep for the training .
10. train_sweep.py : This file contains the training the model through sweep 
11. Prediction_vanilla  Folder : This folder contains predictions for all the Test dataset by best model trained .
Note : All of this files in the  training_vanilla.ipynb

Following are the description of Attention_RNN folder: 
(Link : https://github.com/DA24S019/DA6401_Assignment_3/tree/main/Attention_RNN)

1 downloading_dataset.py  : This file download the dataset required for this assignment.
2. data.py : It contains codes for datapreprocessing and loading data for training and validation.
3. config.py : It contains configuration of the model
4. model.py : It contains the model artitechture. (Attention, Encoder, Decoder and Seq2Seq)
5. pred.py : It contains the code for prediction on my test data and saving the Predictions_Attention Folder and attention maps.
6. attention_heatmaps : It contains the code for the attention heatmap plots.
7. app.py : This contains the app for the visulization for the question no. 6.
8. sweep.py : This file contains sweep for the training .
9. train_sweep.py : This file contains the training the model through sweep 

Note : All of this files in the  training_bahdanau_attention_dummy.ipynb

Only I couldnot upload is best_model.pth and best_model_attention.pth because both are over 100 MB. So first we have to create the sweep and then we get the best model then other files like prediction and heatmap and app.py will work thanks.
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

## 🧪 How to Run (Example Instructions)

### 1. Install Dependencies
```bash
pip install -r requirements.txt



