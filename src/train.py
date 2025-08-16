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
import logging # New: Import the logging module

def setup_logging(log_file):
    """Sets up logging to both console and a file."""
    # New: Create a logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # New: Create a file handler to save logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # New: Create a console handler to print logs to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # New: Define the log message format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # New: Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_environment(config, logger):
    """Sets up device and output directories."""
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}") # New: Use logger instead of print
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    return device

def get_data_loaders(config, logger):
    """Downloads, transforms, and creates data loaders for CIFAR-10."""
    logger.info("\n--- Data Loading and Preprocessing ---") # New: Use logger instead of print

    # Define transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(config['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(config['img_size']),
        transforms.CenterCrop(config['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    full_train_dataset = datasets.CIFAR10(root=config['data_dir'], train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10(root=config['data_dir'], train=False, download=True, transform=val_transforms)

    # Split training dataset into training and validation sets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    logger.info(f"Training samples: {len(train_subset)}") # New: Use logger
    logger.info(f"Validation samples: {len(val_subset)}") # New: Use logger
    logger.info(f"Test samples: {len(test_dataset)}") # New: Use logger
    logger.info(f"Number of batches per epoch (train): {len(train_loader)}") # New: Use logger

    return train_loader, val_loader, test_loader, train_subset, val_subset, test_dataset

def get_model(config, device, logger):
    """Loads and configures the Vision Transformer model."""
    logger.info("\n--- Model Architecture ---") # New: Use logger
    model = timm.create_model(config['model_name'], pretrained=config['pretrained'], num_classes=config['num_classes'])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config['model_name']}") # New: Use logger
    logger.info(f"Total parameters: {total_params:,}") # New: Use logger
    logger.info(f"Trainable parameters: {trainable_params:,}") # New: Use logger
    logger.info(f"Percentage trainable: {trainable_params / total_params * 100:.2f}%") # New: Use logger

    return model

def train_model(model, train_loader, val_loader, train_subset, val_subset, device, config, logger):
    """Runs the training and validation loop."""
    logger.info("\n--- Training Loop ---") # New: Use logger
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    best_val_accuracy = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config['mixed_precision']):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train_samples += labels.size(0)
            correct_train_predictions += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_subset)
        epoch_train_accuracy = correct_train_predictions / total_train_samples * 100
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config['mixed_precision']):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                correct_val_predictions += (predicted == labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_subset)
        epoch_val_accuracy = correct_val_predictions / total_val_samples * 100

        scheduler.step()
        
        epoch_duration = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} | Duration: {epoch_duration:.2f}s") # New: Use logger
        logger.info(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.2f}%") # New: Use logger
        logger.info(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.2f}%") # New: Use logger

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], f'best_model_epoch_{epoch+1}.pth'))
            logger.info(f"Saved best model checkpoint with Val Acc: {best_val_accuracy:.2f}%") # New: Use logger
            
    logger.info("\nTraining finished!") # New: Use logger
    logger.info(f"Best Validation Accuracy: {best_val_accuracy:.2f}%") # New: Use logger
    model.load_state_dict(best_model_wts)
    logger.info("Loaded best model weights for final evaluation.") # New: Use logger
    return model

def evaluate_model(model, test_loader, test_dataset, device, config, logger):
    """Evaluates the model on the test set."""
    logger.info("\n--- Final Evaluation ---") # New: Use logger
    model.eval()
    running_test_loss = 0.0
    correct_test_predictions = 0
    total_test_samples = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config['mixed_precision']):
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            running_test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test_samples += labels.size(0)
            correct_test_predictions += (predicted == labels).sum().item()
    
    test_loss = running_test_loss / len(test_dataset)
    test_accuracy = correct_test_predictions / total_test_samples * 100
    
    logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%") # New: Use logger

if __name__ == "__main__":
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # New: Setup logging before anything else
    log_file_name = f"training_log_{int(time.time())}.log"
    logger = setup_logging(os.path.join('logs', log_file_name))
    
    # Setup environment
    device = setup_environment(config, logger)
    
    # Get data loaders
    train_loader, val_loader, test_loader, train_subset, val_subset, test_dataset = get_data_loaders(config, logger)
    
    # Get model
    model = get_model(config, device, logger)
    
    # Train the model
    best_model = train_model(model, train_loader, val_loader, train_subset, val_subset, device, config, logger)
    
    # Evaluate the final model
    evaluate_model(best_model, test_loader, test_dataset, device, config, logger)