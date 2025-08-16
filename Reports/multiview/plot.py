import matplotlib.pyplot as plt
import numpy as np
import ast  # instead of json

# Data extracted from your provided logs (Python dict style string)
log_data_str = "[[{'epoch': 1, 'train_loss': 0.8017999146133661, 'train_acc': 45.0, 'val_loss': 0.687383696436882, 'val_acc': 60.0}, {'epoch': 2, 'train_loss': 0.7051395747810603, 'train_acc': 51.24999999999999, 'val_loss': 0.6880769670009613, 'val_acc': 55.00000000000001}, {'epoch': 3, 'train_loss': 0.7055709950625897, 'train_acc': 43.75, 'val_loss': 0.7001931935548782, 'val_acc': 40.0}], [{'epoch': 1, 'train_loss': 0.868337694182992, 'train_acc': 50.0, 'val_loss': 0.6936113148927688, 'val_acc': 50.0}, {'epoch': 2, 'train_loss': 0.7145303521305323, 'train_acc': 46.25, 'val_loss': 0.674004727602005, 'val_acc': 65.0}, {'epoch': 3, 'train_loss': 0.6876641001552344, 'train_acc': 58.75, 'val_loss': 0.6738312542438507, 'val_acc': 60.0}], [{'epoch': 1, 'train_loss': 0.787956225615926, 'train_acc': 48.75, 'val_loss': 0.673162916302681, 'val_acc': 60.0}, {'epoch': 2, 'train_loss': 0.6981392167508602, 'train_acc': 50.0, 'val_loss': 0.6889968365430832, 'val_acc': 55.00000000000001}, {'epoch': 3, 'train_loss': 0.7019606210291386, 'train_acc': 52.5, 'val_loss': 0.6857840329408645, 'val_acc': 60.0}]]"

# Parse safely as Python literal
log_data = ast.literal_eval(log_data_str)

# Prepare data for plotting
epochs = [d['epoch'] for d in log_data[0]]
num_folds = len(log_data)
colors = plt.cm.viridis(np.linspace(0, 1, num_folds))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plotting Loss for each fold
for i, fold in enumerate(log_data):
    train_loss = [d['train_loss'] for d in fold]
    val_loss = [d['val_loss'] for d in fold]
    ax1.plot(epochs, train_loss, label=f'Fold {i+1} Train', linestyle='--', color=colors[i])
    ax1.plot(epochs, val_loss, label=f'Fold {i+1} Val', linestyle='-', color=colors[i])

ax1.set_title('Training and Validation Loss per Fold', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_xticks(epochs)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plotting Accuracy for each fold
for i, fold in enumerate(log_data):
    train_acc = [d['train_acc'] for d in fold]
    val_acc = [d['val_acc'] for d in fold]
    ax2.plot(epochs, train_acc, label=f'Fold {i+1} Train', linestyle='--', color=colors[i])
    ax2.plot(epochs, val_acc, label=f'Fold {i+1} Val', linestyle='-', color=colors[i])

ax2.set_title('Training and Validation Accuracy per Fold', fontsize=16)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_xticks(epochs)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("./Reports/multiview/multi_view_performance.png", bbox_inches="tight")  # Save to file
plt.show()
