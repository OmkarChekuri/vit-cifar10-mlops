import matplotlib.pyplot as plt
import numpy as np

# Data from provided logs
data = [{'epoch': 1, 'train_loss': 0.5941, 'train_acc': 79.74, 'val_loss': 0.4465, 'val_acc': 84.48}, 
        {'epoch': 2, 'train_loss': 0.3753, 'train_acc': 87.19, 'val_loss': 0.3625, 'val_acc': 87.4}, 
        {'epoch': 3, 'train_loss': 0.2920, 'train_acc': 89.95, 'val_loss': 0.3090, 'val_acc': 89.33}]

epochs = [d['epoch'] for d in data]
train_acc = [d['train_acc'] for d in data]
val_acc = [d['val_acc'] for d in data]

# Plotting Accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, label='Training Accuracy', marker='o', linestyle='-', color='#2ca02c')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o', linestyle='-', color='#d62728')
plt.title('Training and Validation Accuracy per Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(epochs)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# Save the figure
plot_path = "./Reports/cifar10/accuracy_curve.png"
plt.savefig(plot_path, bbox_inches="tight")
plt.close()

plot_path
