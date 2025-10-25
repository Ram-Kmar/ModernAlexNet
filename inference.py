"""
PyTorch training script for an AlexNet-style model on the Imagenette dataset.

This script includes:
- Model definition (AlexNet)
- Data loading and transformation
- Training loop
- Validation loop
- Checkpoint saving and loading
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import tqdm # For progress bar
from torch.nn import functional as F
from PIL import Image
from typing import Tuple
import matplotlib.pyplot as plt # Keep imports even if unused directly here

# --- Configuration (Copied from your inference script) ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 32 # Keep moderate for data collection to avoid OOM with activations
block_size = 256 # Max context length (relevant for GPT definition)
num_classes = 10
n_embd = 256
n_head = 16
n_layer = 16 # IMPORTANT: Ensure this matches your trained model
dropout = 0.5
checkpoint_path = "alexnet_imagenette.pth" # Path to your trained weights

# --- Configuration for Activation Collection ---
DATASET_PATH = 'path/to/your/imagenette' # IMPORTANT: Set this path
OUTPUT_DIR = 'activation_dataset_block6_resid' # Directory to save activations
TARGET_BLOCK_INDEX = 6 # 7th block is index 6 (Residual Stream Output)
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 4
PIN_MEMORY = False if device.type == 'mps' else True
IMG_SIZE = 224 # Image size your model expects

# --- Helper: Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving activations to: {os.path.abspath(OUTPUT_DIR)}")
print(f"Using device: {device}")

# --- Model Definition (Copied from your inference script) ---
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Note: block_size is used here, ensure it's appropriate.
        # For vision, causal mask might not be needed depending on task.
        # If your model wasn't trained with a causal mask, remove this line
        # and the masked_fill below. Let's assume it was for now.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # --- Causal Mask Check ---
        # If your model is NOT causal (e.g., standard ViT), comment out/remove this line:
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # ------------------------
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
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
        # Check if head_size * num_heads matches n_embd, adjust proj input if needed
        if out.shape[-1] != self.proj.in_features:
             print(f"Warning: MHA output dim {out.shape[-1]} != proj input dim {self.proj.in_features}. Check n_embd, n_head.")
             # Option 1: Adjust projection layer (if definition is flexible)
             # self.proj = nn.Linear(out.shape[-1], n_embd).to(x.device) # Recreate proj layer
             # Option 2: Pad or truncate 'out' (less ideal)
             # Option 3: Fix model definition (best if possible)
             pass # Let it potentially error or handle based on your model's actual behavior

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
        super().__init__()
        head_size = n_embd // n_head
        if n_embd % n_head != 0:
            print(f"Warning: n_embd ({n_embd}) not perfectly divisible by n_head ({n_head}).")
            # Adjust head_size calculation or check parameters if this is unintended
            head_size = n_embd // n_head # Integer division might lose info

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x # This is the residual stream output for this block

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # The lm_head input dimension looks specific (9216 = 36 * 256).
        # Ensure the input sequence length T matches this after flattening.
        # Input to transformer is (B, 36, 256), output is (B, 36, 256)
        # After ln_f, it's (B, 36, 256). Flattened it's (B, 36 * 256) = (B, 9216)
        self.lm_head = nn.Linear(n_embd * 36, num_classes) # Adjusted input size based on AlexNet forward
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): # Not used here but good practice
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): # idx has shape (B, T, C) = (B, 36, 256)
        x = self.blocks(idx) # Output (B, 36, 256)
        x = self.ln_f(x) # Output (B, 36, 256)
        x = torch.flatten(x, 1) # Output (B, 36 * 256) = (B, 9216)
        output = self.lm_head(x) # Output (B, num_classes)
        # Note: Loss calculation part removed as we only need forward pass for hooks
        return output

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None: # Default to 10 for ImageNette
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), # Output: [B, 96, 55, 55]
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2), # Output: [B, 96, 27, 27]

            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2), # Output: [B, 256, 27, 27]
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2), # Output: [B, 256, 13, 13]

            nn.Conv2d(256, 384, kernel_size=3, padding=1), # Output: [B, 384, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2), # Output: [B, 384, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2), # Output: [B, 256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # Output: [B, 256, 6, 6]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) # Ensures output is [B, 256, 6, 6]
        self.transformer = GPT()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)         # Output: [B, 256, 6, 6]
        x = self.avgpool(x)           # Output: [B, 256, 6, 6]
        # Reshape for transformer: treat the 6x6 grid as a sequence of 36 'patches'
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1) # Output: [B, 36, 256]
        # Now x has shape [Batch, SequenceLength=36, EmbeddingDim=256]
        x = self.transformer(x)      # Pass through transformer blocks and final head
        return x

# --- Instantiate and Load Model Weights ---
print("Loading model and weights...")
model = AlexNet(num_classes=num_classes).to(device)

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Check if checkpoint directly contains state_dict or is nested
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded weights from 'model_state_dict' key.")
    else:
        # Assuming the checkpoint itself is the state_dict
        model.load_state_dict(checkpoint)
        print("Loaded weights directly from checkpoint file.")

    # You can optionally print loss history if needed, like in inference script
    # if "history" in checkpoint and "train_loss" in checkpoint["history"]:
    #    train_loss = checkpoint["history"]["train_loss"]
    #    print(f"Final training loss from checkpoint: {train_loss[-1]}")

except FileNotFoundError:
    print(f"ERROR: Checkpoint file not found at {checkpoint_path}. Cannot load weights.")
    exit()
except KeyError as e:
    print(f"Error loading checkpoint: Missing key {e}. Checkpoint structure might be different.")
    exit()
except RuntimeError as e:
    print(f"Error loading state_dict: {e}. Model definition might mismatch checkpoint.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading the checkpoint: {e}")
    exit()

model.eval() # Set model to evaluation mode
print("Model loaded successfully.")

# --- Data Loading and Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading dataset...")
try:
    image_dataset = datasets.ImageFolder(DATASET_PATH, transform=preprocess)
    # Use num_workers and pin_memory from config
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    print(f"Dataset loaded with {len(image_dataset)} images.")
except FileNotFoundError:
    print(f"ERROR: Dataset not found at {DATASET_PATH}. Please set the correct path.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- Hook Setup ---
activation_storage = {'activation': None}
hook_handle = None

def get_activation_hook(name):
    """Hook function to capture output and store it."""
    def hook(module, input, output): # Correct arguments: module, input, output
        # Block output is the residual stream value for that block
        if isinstance(output, tuple):
             activation_storage[name] = output[0].detach().cpu()
        else:
             activation_storage[name] = output.detach().cpu()
    return hook

# Register the hook on the target block's *output*
try:
    if TARGET_BLOCK_INDEX < 0 or TARGET_BLOCK_INDEX >= n_layer:
         raise IndexError(f"TARGET_BLOCK_INDEX ({TARGET_BLOCK_INDEX}) out of range for {n_layer} layers.")
    # The target module is the specific Block instance within the Sequential container
    target_module = model.transformer.blocks[TARGET_BLOCK_INDEX]
    hook_handle = target_module.register_forward_hook(get_activation_hook('activation'))
    print(f"Hook registered on output of transformer block {TARGET_BLOCK_INDEX}.")
except AttributeError:
     print("Error: Could not find 'model.transformer.blocks'. Adjust the path to the transformer blocks.")
     exit()
except IndexError as e:
     print(f"Error: {e}")
     exit()
except Exception as e:
     print(f"An unexpected error occurred during hook registration: {e}")
     exit()

# --- Activation Collection Loop ---
print("Starting activation collection...")
saved_count = 0
file_indices = {} # To keep track of indices per original filename/path

# Store original file paths if possible (ImageFolder provides this)
original_filepaths = [item[0] for item in image_dataset.samples]

with torch.no_grad():
    overall_idx = 0
    for batch_idx, (inputs, _) in enumerate(tqdm.tqdm(dataloader, desc="Processing Batches")):
        inputs = inputs.to(device)
        try:
            _ = model(inputs) # Run forward pass - triggers the hook
        except Exception as e:
            print(f"\nError during forward pass on batch {batch_idx}: {e}")
            print("Skipping this batch.")
            continue # Skip saving for this batch

        if activation_storage['activation'] is None:
            print(f"Warning: Hook did not capture any activation in batch {batch_idx}.")
            continue

        batch_activations = activation_storage['activation'] # Shape: [batch_size, 36, 256]

        # Save activation for each image in the batch separately
        current_batch_size = batch_activations.shape[0]
        for i in range(current_batch_size):
            # Calculate the true index in the dataset
            dataset_idx = batch_idx * dataloader.batch_size + i
            if dataset_idx >= len(original_filepaths):
                print(f"Warning: Calculated dataset index {dataset_idx} out of bounds.")
                continue

            single_activation = batch_activations[i] # Shape: [36, 256]

            # --- Naming Convention ---
            # Extract original filename (without extension) for clarity
            original_filename = os.path.basename(original_filepaths[dataset_idx])
            filename_base, _ = os.path.splitext(original_filename)

            # Keep track of multiple samples if filenames aren't unique (unlikely for ImageFolder)
            if filename_base not in file_indices:
                file_indices[filename_base] = 0
            else:
                file_indices[filename_base] += 1
            instance_idx = file_indices[filename_base]

            # Construct save path
            save_filename = f"activation_block{TARGET_BLOCK_INDEX}_{filename_base}_{instance_idx:02d}.pt"
            save_path = os.path.join(OUTPUT_DIR, save_filename)
            # -------------------------

            try:
                torch.save(single_activation, save_path)
                saved_count += 1
            except Exception as e:
                print(f"\nError saving activation for index {dataset_idx} ({save_filename}): {e}")

        # Clear storage for the next batch
        activation_storage['activation'] = None

# --- Cleanup ---
if hook_handle:
    hook_handle.remove()
    print("Hook removed.")

print(f"\nActivation collection complete.")
print(f"Successfully saved {saved_count} activation tensors to '{OUTPUT_DIR}'.")

# --- Optional: Verify saved file ---
if saved_count > 0:
    try:
        # List files and try loading the first one found
        saved_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt')]
        if saved_files:
            first_file = os.path.join(OUTPUT_DIR, saved_files[0])
            loaded_tensor = torch.load(first_file)
            print(f"Verified saving: Loaded '{saved_files[0]}' with shape {loaded_tensor.shape}")
            # Expected shape: [36, 256]
            if loaded_tensor.shape != torch.Size([36, n_embd]):
                 print(f"Warning: Loaded tensor shape {loaded_tensor.shape} does not match expected [36, {n_embd}]")
        else:
            print("Could not verify saving: No .pt files found in output directory.")

    except Exception as e:
        print(f"Error during verification load: {e}")


    print("\nRegistering hooks to capture activations...")

# --- 1. Define Helper Functions ---

    def plot_feature_maps(feature_maps, title, grid_size=(8, 12)):
        """Plots the feature maps from a convolutional layer."""
        if feature_maps is None or not isinstance(feature_maps, torch.Tensor):
            print(f"Skipping plot for {title}: Invalid feature map data.")
            return
        try:
            num_maps = feature_maps.shape[1] # Get the number of channels
            maps_to_plot = feature_maps.squeeze(0).cpu().numpy() # Squeeze batch, move to CPU, convert to numpy

            rows, cols = grid_size
            plot_count = min(num_maps, rows * cols) # Determine how many maps to actually plot

            if num_maps > plot_count:
                print(f"Warning for '{title}': Only plotting first {plot_count} of {num_maps} maps.")

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
            fig.suptitle(title, fontsize=16)

            for i, ax in enumerate(axes.flat):
                if i < plot_count:
                    # Handle potential variations in map shapes if necessary, though unlikely for Conv output
                    im = ax.imshow(maps_to_plot[i], cmap='viridis')
                    ax.set_title(f'Map {i+1}')
                    ax.axis('off')
                    # Optional: Add a colorbar for each map or for the figure
                    # fig.colorbar(im, ax=ax)
                else:
                    ax.axis('off') # Hide unused subplots

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # plt.show() # Display the plot immediately
        except Exception as e:
            print(f"Error plotting feature maps for {title}: {e}")
            # Print tensor shape for debugging
            print(f"Tensor shape was: {feature_maps.shape}")


    def plot_transformer_activation_heatmap(activation_tensor, title):
        """Plots transformer activations (MLP or Residual) as a heatmap."""
        if activation_tensor is None or not isinstance(activation_tensor, torch.Tensor):
            print(f"Skipping plot for {title}: Invalid activation data.")
            return
        try:
            # Expecting shape [batch, seq_len, features]
            # Squeeze batch dim (assuming batch_size=1), move to CPU, convert to numpy
            data_to_plot = activation_tensor.squeeze(0).cpu().numpy()

            fig, ax = plt.subplots(figsize=(10, 5)) # Adjust figsize as needed
            im = ax.imshow(data_to_plot.T, cmap='viridis', aspect='auto') # Transpose for features on y-axis

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Sequence Position (Patch)')
            ax.set_ylabel('Feature Dimension')
            fig.colorbar(im, ax=ax, label='Activation Value')

            plt.tight_layout()
            # plt.show() # Display the plot immediately
        except Exception as e:
            print(f"Error plotting heatmap for {title}: {e}")
            # Print tensor shape for debugging
            print(f"Tensor shape was: {activation_tensor.shape}")


# --- 2. Define Hooks and Storage ---

# Use a dictionary to store activations from various points
    activations = {}
    hook_handles = [] # Keep track of registered hooks to remove them later

    def get_activation_hook(name):
        """Generic hook function to store the output of a module."""
        def hook(model, input, output):
            # Store output. Detach to prevent gradient tracking.
            # Handle potential tuple outputs (common in transformers)
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook

# --- 3. Register Hooks ---

# --- AlexNet Hooks ---
# Hooking outputs after ReLU usually makes more sense for visualization
    try:
        # Early Conv Layer (after ReLU)
        hook_handles.append(
            model.features[1].register_forward_hook(get_activation_hook('alexnet_relu1'))
        )
        # Mid Conv Layer (after ReLU)
        hook_handles.append(
            model.features[9].register_forward_hook(get_activation_hook('alexnet_relu4')) # Corresponds to Conv2d at index 8
        )
        # Late Conv Layer (after ReLU)
        hook_handles.append(
            model.features[13].register_forward_hook(get_activation_hook('alexnet_relu6')) # Corresponds to Conv2d at index 12
        )
    except IndexError as e:
        print(f"Error registering AlexNet hooks: {e}. Check layer indices.")
    except AttributeError as e:
        print(f"Error registering AlexNet hooks: {e}. Model structure might be different.")


# --- Transformer Hooks ---
# Choose which block(s) and which part(s) to monitor
    monitor_block_indices = [0, n_layer // 2, n_layer - 1] # First, Middle, Last

    for block_idx in monitor_block_indices:
        if block_idx >= n_layer:
            print(f"Warning: Block index {block_idx} is out of bounds for {n_layer} layers. Skipping.")
            continue
        try:
            # --- MLP Output ---
            # Hook the output of the Sequential 'net' inside 'ffwd'
            mlp_output_module = model.transformer.blocks[block_idx].ffwd.net
            hook_handles.append(
                mlp_output_module.register_forward_hook(get_activation_hook(f'transformer_block{block_idx}_mlp_out'))
            )

            # --- Residual Stream Output ---
            # Hook the output of the entire Block
            block_module = model.transformer.blocks[block_idx]
            hook_handles.append(
                block_module.register_forward_hook(get_activation_hook(f'transformer_block{block_idx}_resid_post'))
            )
        except IndexError as e:
            print(f"Error registering Transformer hooks for block {block_idx}: {e}. Check layer indices.")
        except AttributeError as e:
            print(f"Error registering Transformer hooks for block {block_idx}: {e}. Model structure might be different.")


# --- 4. Run Model Forward Pass ---
    print("Running forward pass to trigger hooks...")
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
        print("Forward pass completed.")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        # Consider stopping if forward pass fails
        # raise e

# --- 5. Plot Activations ---
    print("Plotting captured activations...")

# --- Plot CNN Feature Maps ---
    plot_feature_maps(activations.get('alexnet_relu1'), title='AlexNet Layer 1 (after ReLU) Activations')
    plot_feature_maps(activations.get('alexnet_relu4'), title='AlexNet Layer 4 (after ReLU) Activations')
    plot_feature_maps(activations.get('alexnet_relu6'), title='AlexNet Layer 6 (after ReLU) Activations')


# --- Plot Transformer Activations (as Heatmaps) ---
    for block_idx in monitor_block_indices:
        if block_idx >= n_layer: continue # Skip if index was invalid

        # Plot MLP Output
        plot_transformer_activation_heatmap(
            activations.get(f'transformer_block{block_idx}_mlp_out'),
            title=f'Transformer Block {block_idx+1} MLP Output Activations'
        )

        # Plot Residual Stream Output
        plot_transformer_activation_heatmap(
            activations.get(f'transformer_block{block_idx}_resid_post'),
            title=f'Transformer Block {block_idx+1} Residual Stream Output Activations'
        )


# --- 6. Remove Hooks (Important!) ---
    print("\nRemoving hooks...")
    for handle in hook_handles:
        handle.remove()
    hook_handles.clear() # Clear the list
    activations.clear() # Clear the stored activations
    print("Hooks removed.")

# --- 7. Display all plots ---
# If you didn't call plt.show() inside the functions, call it once here
    plt.show()

# ===================================================================
    # ... [Your existing code for plotting loss] ...
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss)+1), train_loss, marker='o', label='Training Loss')
    # ... [rest of your plotting code] ...
    plt.show()
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params/1e6:.2f}M trainable parameters.")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss)+1), train_loss, marker='o', label='Training Loss')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()
