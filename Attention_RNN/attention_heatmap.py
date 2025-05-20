import os
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# Step 1: Create the directory for fonts/plots if it doesn't exist
os.makedirs("visualization", exist_ok=True)

# Step 2: Download the Noto Sans Devanagari font file
font_url = "https://github.com/openmaptiles/fonts/raw/master/noto-sans/NotoSansDevanagari-Regular.ttf"
font_path = "visualization/NotoSansDevanagari-Regular.ttf"
if not os.path.isfile(font_path):
    urllib.request.urlretrieve(font_url, font_path)
    print(f"Font downloaded to {font_path}")
else:
    print(f"Font already exists at {font_path}")

# Step 3: Load the font for matplotlib
font_prop = FontProperties(fname=font_path, size=16)

# Step 4: Prepare plotting
fig, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)

# Sample 9 random indices for 3x3 grid plotting
sample_indices = np.random.choice(len(test_latin), 9, replace=False)

for ax, idx in zip(axes.flat, sample_indices):
    inp = test_latin[idx]   # English input word
    tgt = test_deva[idx]    # Hindi target word

    pred, attn = translator.translate_with_attention(inp)

    input_tokens = ['<sos>'] + list(inp) + ['<eos>']
    output_tokens = [c for c in pred if c.strip() and c not in ['<pad>', '<unk>']]

    im = ax.imshow(attn, aspect='auto', cmap='viridis')

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(range(len(input_tokens)))
    ax.set_xticklabels(input_tokens)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_yticks(range(len(output_tokens)))
    ax.set_yticklabels(output_tokens, fontproperties=font_prop)

    # Add text annotations above plot
    ax.text(0.5, 1.22, f"Input: {inp}", fontsize=16, ha='center', va='bottom', transform=ax.transAxes)
    ax.text(0.0, 1.13, "Pred:", fontsize=14, ha='left', va='bottom', transform=ax.transAxes)
    ax.text(1.0, 1.13, pred, fontsize=16, ha='right', va='bottom', transform=ax.transAxes, fontproperties=font_prop)
    ax.text(0.0, 1.05, "Target:", fontsize=14, ha='left', va='bottom', transform=ax.transAxes)
    ax.text(1.0, 1.05, tgt, fontsize=16, ha='right', va='bottom', transform=ax.transAxes, fontproperties=font_prop)

    ax.set_xlabel("Input Characters")
    ax.set_ylabel("Output Characters")

    fig.colorbar(im, ax=ax)

# Turn off any unused subplots (if any)
for ax in axes.flat[len(sample_indices):]:
    ax.axis('off')

# Save and show the plot
plt.savefig('visualization/attention_heatmap_grid.png')
plt.show()
print("Plot saved to visualization/attention_heatmap_grid.png")
