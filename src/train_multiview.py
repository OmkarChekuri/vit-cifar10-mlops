import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
import timm
from torch.utils.data import DataLoader, random_split
import os
import time
import copy
import yaml
import logging
import random
import numpy as np
from tabulate import tabulate # You may need to install this: pip install tabulate

def setup_logging(log_file):
    """Sets up logging to both console and a file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- New Model Architecture Classes ---
class Adapter(nn.Module):
    """A simple bottleneck adapter as described in the paper."""
    def __init__(self, in_features, bottleneck_ratio=32):
        super().__init__()
        hidden_features = in_features // bottleneck_ratio
        self.fc_down = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc_up = nn.Linear(hidden_features, in_features)
        self.bottleneck_ratio = bottleneck_ratio

    def forward(self, x):
        return self.fc_up(self.gelu(self.fc_down(x)))

class AdaptedViTBackbone(nn.Module):
    """ViT backbone with adapter modules injected into transformer blocks."""
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True, use_adapter=True, embed_dim=192, bottleneck_ratio=32):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.use_adapter = use_adapter
        self.embed_dim = embed_dim
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.adapters_msa = nn.ModuleList()
        self.adapters_mlp = nn.ModuleList()
        self.gammas_msa = nn.ParameterList()
        self.gammas_mlp = nn.ParameterList()

        if use_adapter:
            for i, block in enumerate(self.model.blocks):
                adapter_msa = Adapter(embed_dim, bottleneck_ratio=bottleneck_ratio)
                adapter_mlp = Adapter(embed_dim, bottleneck_ratio=bottleneck_ratio)

                self.adapters_msa.append(adapter_msa)
                self.adapters_mlp.append(adapter_mlp)

                self.gammas_msa.append(nn.Parameter(torch.ones(embed_dim)))
                self.gammas_mlp.append(nn.Parameter(torch.ones(embed_dim)))

        if use_adapter:
            for adapter in self.adapters_msa:
                for param in adapter.parameters():
                    param.requires_grad = True
            for adapter in self.adapters_mlp:
                for param in adapter.parameters():
                    param.requires_grad = True
            for param in self.gammas_msa:
                param.requires_grad = True
            for param in self.gammas_mlp:
                param.requires_grad = True

    def forward(self, x):
        if x.dim() == 4:
            x = self.model.patch_embed(x)
        
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)

        for i, block in enumerate(self.model.blocks):
            x_attn = block.norm1(x)
            x_attn = block.attn(x_attn)
            x = x + block.drop_path1(x_attn)

            if self.use_adapter:
                adapter_msa = self.adapters_msa[i]
                gamma_msa = self.gammas_msa[i]
                x = x + gamma_msa * adapter_msa(x)

            x_mlp = block.norm2(x)
            x_mlp = block.mlp(x_mlp)
            x = x + block.drop_path2(x_mlp)

            if self.use_adapter:
                adapter_mlp = self.adapters_mlp[i]
                gamma_mlp = self.gammas_mlp[i]
                x = x + gamma_mlp * adapter_mlp(x)
        
        return self.model.norm(x)

class MultiInputClassifier(nn.Module):
    """Combines a single shared ViT backbone for multi-view processing."""
    def __init__(self, num_views=4, num_classes=1, model_name='vit_tiny_patch16_224', use_adapter=True, bottleneck_ratio=32):
        super().__init__()
        self.num_views = num_views
        dummy_model = timm.create_model(model_name)
        self.embed_dim = dummy_model.embed_dim
        del dummy_model

        self.shared_local_backbone = AdaptedViTBackbone(
            model_name=model_name,
            pretrained=True,
            use_adapter=use_adapter,
            embed_dim=self.embed_dim,
            bottleneck_ratio=bottleneck_ratio
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim * num_views),
            nn.Linear(self.embed_dim * num_views, self.embed_dim * num_views // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * num_views // 2, num_classes)
        )

    def forward(self, inputs):
        view_cls_tokens = []
        for i, x_view in enumerate(inputs):
            features = self.shared_local_backbone(x_view)
            cls_token = features[:, 0]
            view_cls_tokens.append(cls_token)
        
        combined_features = torch.cat(view_cls_tokens, dim=1)
        
        logits = self.mlp_head(combined_features)
        return logits

# --- Dummy Dataset ---
class DummyMultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, img_size, num_views, num_channels=3):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_views = num_views
        self.num_channels = num_channels
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        views = [torch.randn(self.num_channels, self.img_size, self.img_size) for _ in range(self.num_views)]
        label = torch.tensor(random.randint(0, 1), dtype=torch.float32)
        return views, label

def train_and_evaluate_fold(fold_idx, model, optimizer, criterion, scheduler, train_loader, val_loader, num_epochs, warmup_epochs, device, checkpoint_dir, logger):
    logger.info(f"\n--- Starting Training for Fold {fold_idx + 1} ---")
    best_val_accuracy = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (inputs_views, labels) in enumerate(train_loader):
            inputs_views = [view.to(device) for view in inputs_views]
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs_views)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted_classes = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)
        
        if epoch >= warmup_epochs:
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples * 100

        model.eval()
        running_val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs_val_views, labels_val in val_loader:
                inputs_val_views = [view_val.to(device) for view_val in inputs_val_views]
                labels_val = labels_val.to(device).unsqueeze(1)

                outputs_val = model(inputs_val_views)
                loss_val = criterion(outputs_val, labels_val)

                running_val_loss += loss_val.item()
                predicted_val_classes = (torch.sigmoid(outputs_val) > 0.5).float()
                correct_val_predictions += (predicted_val_classes == labels_val).sum().item()
                total_val_samples += labels_val.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_accuracy = correct_val_predictions / total_val_samples * 100

        logger.info(f"Fold {fold_idx+1}, Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}% | Val Acc: {epoch_val_accuracy:.2f}%")

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # Create checkpoint directory if it doesn't exist
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            
            # Use os.path.join for platform-independent paths
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best_model_fold_{fold_idx+1}.pth'))
    
    logger.info(f"--- Fold {fold_idx + 1} Finished. Best Val Acc: {best_val_accuracy:.2f}% ---\n")
    return best_val_accuracy

# --- Main Execution ---
if __name__ == "__main__":
    # Load configuration
    config = load_config('configs/multiview_config.yaml')
    
    # New: Setup logging before anything else
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file_name = f"training_log_multiview_{int(time.time())}.log"
    logger = setup_logging(os.path.join('logs', log_file_name))
    
    # Store results for the final table
    results_data = []

    # Loop for conceptual K-fold cross-validation
    for fold_idx in range(config['num_folds']):
        # Re-initialize model for each fold to ensure fresh weights
        model = MultiInputClassifier(
            num_views=config['num_views'],
            num_classes=config['num_classes'],
            model_name=config['model_name_for_vit'],
            use_adapter=config['use_adapters'],
            bottleneck_ratio=config['bottleneck_ratio']
        ).to(config['device'])

        # Print model parameters for the first fold
        if fold_idx == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model: {config['model_name_for_vit']}")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters (with adapters): {trainable_params:,}")
            logger.info(f"Percentage trainable: {trainable_params / total_params * 100:.2f}%")

        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'] - config['warmup_epochs'])

        # Create dummy dataset and loaders for this fold
        fold_dummy_dataset = DummyMultiViewDataset(
            num_samples=100,
            img_size=config['img_size'],
            num_views=config['num_views']
        )
        
        # Simulate train/val split for the fold
        fold_train_size = int(0.8 * len(fold_dummy_dataset))
        fold_val_size = len(fold_dummy_dataset) - fold_train_size
        fold_train_subset, fold_val_subset = random_split(fold_dummy_dataset, [fold_train_size, fold_val_size])

        fold_train_loader = DataLoader(fold_train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
        fold_val_loader = DataLoader(fold_val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)

        fold_accuracy = train_and_evaluate_fold(
            fold_idx, model, optimizer, criterion, scheduler,
            fold_train_loader, fold_val_loader, config['num_epochs'], config['warmup_epochs'], config['device'], config['checkpoint_dir'], logger
        )
        results_data.append(fold_accuracy)

    logger.info("\n--- Summary of Classification Accuracy (ACC) ---")
    
    headers = ["Model", "Finetuning Method", "Param (M)", "Tunable Param (M)",
               *[f"Fold {i+1}" for i in range(config['num_folds'])], "Mean ACC (%) ± STD"]
    
    mean_acc = np.mean(results_data)
    std_acc = np.std(results_data)

    model_params_m = round(total_params / 1_000_000, 1)
    tunable_params_m = round(trainable_params / 1_000_000, 3)

    row = [
        config['model_name_for_vit'],
        "Adapter",
        model_params_m,
        tunable_params_m,
        *[f"{acc:.1f}" for acc in results_data],
        f"{mean_acc:.1f} ± {std_acc:.1f}"
    ]
    
    logger.info(tabulate([row], headers=headers, tablefmt="grid"))
    logger.info("\n--- Project Complete ---")