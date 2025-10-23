"""
PyTorch training script for an AlexNet-style model on the Imagenette dataset.

This script includes:
- Model definition (AlexNet)
- Data loading and transformation
- Training loop
- Validation loop
- Checkpoint saving and loading
"""

import os
import time
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from typing import Tuple

# --- 1. Configuration ---

# Training settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
NUM_CLASSES = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 128
NUM_EPOCHS = 4

# Data and Checkpoint paths
DATA_ROOT = "data"
CHECKPOINT_PATH = "alexnet_imagenette.pth"

# DataLoader settings
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 4
# 'mps' device does not support pin_memory, so we set it based on the device
PIN_MEMORY = False if DEVICE.type == 'mps' else True

# -----------hyperparameter for Transformer --------------

batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
# max_iters = 200
# eval_interval = 10
# learning_rate = 3e-4
device = 'mps' if torch.mps.is_available() else 'cpu'
eval_iters = 20
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.2
# ------------
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(9216, NUM_CLASSES)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T, C = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        # x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(idx) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        x = torch.flatten(x,1)
        output = self.lm_head(x) # (B,T,vocab_size)

        # if targets is None:
        #     loss = None
        # else:
        #     B, T, C = logits.shape
        #     logits = logits.view(B*T, C)
        #     targets = targets.view(B*T)
        #     loss = F.cross_entropy(logits, targets)
        #
        return output

    # def generate(self, idx, max_new_tokens):
    #     # idx is (B, T) array of indices in the current context
    #     for _ in range(max_new_tokens):
    #         # crop idx to the last block_size tokens
    #         idx_cond = idx[:, -block_size:]
    #         # get the predictions
    #         logits, loss = self(idx_cond)
    #         # focus only on the last time step
    #         logits = logits[:, -1, :] # becomes (B, C)
    #         # apply softmax to get probabilities
    #         probs = F.softmax(logits, dim=-1) # (B, C)
    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
    #         # append sampled index to the running sequence
    #         idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    #     return idx

# --- 2. Model Definition ---

class AlexNet(nn.Module):
    """
    AlexNet-style architecture adapted for smaller inputs and a specific 
    number of classes.
    """
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )
        self.transformer = GPTLanguageModel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], 36, 256)
        x = self.transformer(x)
        return x


# --- 3. Helper Functions ---

def get_data_loaders(
    root: str, 
    batch_size: int, 
    num_workers: int, 
    pin_memory: bool
) -> Tuple[DataLoader, DataLoader]:
    """Creates and returns the training and validation DataLoaders."""
    print("Loading Imagenette dataset...")
    
    # Standard normalization for ImageNet-trained models
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.Imagenette(root=root,
                                        split='train',
                                        transform=transform,
                                        download=True)

    val_dataset = datasets.Imagenette(root=root,
                                      split='val',
                                      transform=transform,
                                      download=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    
    print("Datasets loaded successfully.")
    return train_loader, val_loader


def save_checkpoint(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    epoch: int, 
    loss: float, 
    history: dict,
    path: str
):
    """Saves the model checkpoint."""
    print(f"✅ Saving checkpoint at epoch {epoch+1} to {path}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'history': history, 
    }, path)


def load_checkpoint(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    path: str,
    device: torch.device
) -> Tuple[int, float, dict]:
    """Loads the model checkpoint."""
    print(f"🔄 Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    
    history = checkpoint.get('history',{
        'train_loss':[],'train_acc':[], 'val_loss':[],'val_acc':[]
    })

    print(f"Resumed from epoch {start_epoch} (last saved loss: {loss:.4f})")
    return start_epoch, loss,history


def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """Runs one full training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """Runs one full validation epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc


# --- 4. Main Execution ---

def main():
    """Main training and validation script."""
    parser = argparse.ArgumentParser(description='Train AlexNet on Imagenette')
    parser.add_argument(
        '-e', '--epochs', 
        type=int, 
        default=2, 
        help='Total number of epochs to train for (default: 2)'
    )
    args = parser.parse_args()
    num_epochs = args.epochs  # This is our new "total epochs"

    print(f"--- AlexNet Training on Imagenette ---")
    print(f"Using device: {DEVICE}")
    print(f"Using {NUM_WORKERS} workers and Pin Memory: {PIN_MEMORY}")

    # --- 1. Load Data ---
    train_loader, val_loader = get_data_loaders(
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # --- 2. Initialize Model, Loss, and Optimizer ---
    model = AlexNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params/1e6:.2f}M trainable parameters.")

    # --- 3. Load Checkpoint (if exists) ---
    start_epoch = 0
    best_val_loss=float('inf')
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
    }
    if os.path.exists(CHECKPOINT_PATH):
        try:
            start_epoch, _ = load_checkpoint(
                model, optimizer, CHECKPOINT_PATH, DEVICE
            )
        except Exception as e:
            print(f"Warning: Could not load checkpoint. Starting fresh. Error: {e}")
    else:
        print("No checkpoint found, starting fresh.")

    if start_epoch >= num_epochs:
            print(f"\nModel has already been trained for {start_epoch} epochs. Target is {num_epochs}.")
            print("To train for more epochs, increase the --epochs argument.")
    # --- 4. The Training Loop ---
    else:
        print(f"\n--- Starting Training from Epoch {start_epoch+1} ---")
        total_start_time = time.time()

        for epoch in range(start_epoch, num_epochs):  # <-- USE num_epochs
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE
            )
            
            # Validate
            val_loss, val_acc = validate(
                model, val_loader, criterion, DEVICE
            )
            
            # Save metrics for plotting
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            epoch_duration = time.time() - epoch_start_time
            
            # Log results
            print(f"Epoch {epoch+1:02}/{num_epochs} | "  # <-- USE num_epochs
                  f"Time: {epoch_duration:.2f}s | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint only if validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss, history, CHECKPOINT_PATH
                )
            else:
                print(f"Validation loss did not improve from {best_val_loss:.4f}.")

        total_end_time = time.time()
        print("\n--- Training Finished ---")
        print(f"Total training time: {(total_end_time - total_start_time) / 60:.2f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final model and checkpoint saved to {CHECKPOINT_PATH}")

    # # --- 5. Plot and Save Curves ---
    # # This will now plot the *full* history, even when resuming
    # if not history['train_loss']:
    #      print("\nNo training was performed, skipping plot generation.")
    # else:
    #     plot_save_path = 'training_curves.png'
    #     plot_curves(history, plot_save_path)
    #     print(f"Training curves saved to {plot_save_path}")
    #
if __name__ == '__main__':
    main()
