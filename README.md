# ModernAlexNet: AlexNet with a Transformer Head

This project implements a hybrid deep learning model that combines the classic AlexNet convolutional neural network (CNN) with a modern transformer-based architecture (similar to GPT) for image classification. The model is trained and evaluated on the Imagenette dataset, a smaller subset of ImageNet.

## Overview

The core idea is to leverage the powerful feature extraction capabilities of a CNN like AlexNet and combine them with the sequence-processing power of a transformer.

1.  **Feature Extraction:** An AlexNet-style CNN processes the input images to extract a rich set of hierarchical features.
2.  **Transformer Head:** The feature maps from the CNN are then treated as a sequence of "patches" and fed into a transformer-based head. This allows the model to learn complex relationships and dependencies between different parts of the image.
3.  **Classification:** A final linear layer maps the transformer's output to the number of classes for classification.

This hybrid approach aims to combine the best of both worlds: the spatial hierarchy of features from CNNs and the global attention mechanism of transformers.

## Model Architecture

The model consists of two main components:

1.  **AlexNet Backbone (`features`):**
    *   This is a modified version of the original AlexNet, containing a sequence of convolutional, ReLU, and max-pooling layers.
    *   It takes a 3-channel image as input and produces a set of feature maps.

2.  **GPT-based Transformer Head (`transformer`):**
    *   The output feature maps from the AlexNet backbone are reshaped from `[Batch, Channels, Height, Width]` to `[Batch, SequenceLength, EmbeddingDim]`. In this model, the sequence length is the flattened spatial dimensions (`Height * Width`), and the embedding dimension is the number of channels.
    *   This sequence is then processed by a series of transformer blocks, each consisting of:
        *   Multi-Head Self-Attention
        *   Feed-Forward Network
        *   Layer Normalization and Residual Connections
    *   A final linear layer (`lm_head`) maps the transformer's output to the final class predictions.

## Dataset

The model is trained on the **Imagenette** dataset, which is a small subset of the full ImageNet dataset. It contains 10 easily distinguishable classes, making it ideal for rapid prototyping and experimentation.

The 10 classes are:
- tench
- English springer
- cassette player
- chain saw
- church
- French horn
- garbage truck
- gas pump
- golf ball
- parachute

The script will automatically download the dataset when you run the training for the first time.

## Dependencies

The project is built using PyTorch. The main dependencies are:

- `torch`
- `torchvision`
- `tqdm`
- `Pillow`

You can install them using pip:
```bash
pip install torch torchvision tqdm Pillow
```

## Usage

The `model.py` script provides a command-line interface to train the model or run inference.

### Directory Structure

Make sure your project has the following structure:

```
/
├── data/                 # Directory for the Imagenette dataset
├── image/                # Directory for images to be used for inference
│   ├── image1.jpg
│   └── ...
├── alexnet_imagenette.pth  # Saved model checkpoint
├── model.py              # The main script
└── README.md
```

### Training

To train the model, run the following command:

```bash
python model.py --train True --epochs 20
```

- `--train True`: Specifies that you want to train the model.
- `--epochs <number>`: Sets the total number of epochs to train for.

The script will:
1.  Download the Imagenette dataset (if not already present).
2.  Initialize the model, optimizer, and loss function.
3.  Load a checkpoint if one exists, otherwise it will start training from scratch.
4.  Train the model for the specified number of epochs.
5.  Save a checkpoint (`alexnet_imagenette.pth`) whenever the validation loss improves.

### Inference

To run inference on a set of images, place them in the `image/` directory and run:

```bash
python model.py --inference True
```

The script will:
1.  Load the trained model from the `alexnet_imagenette.pth` checkpoint.
2.  Load all images from the `image/` directory.
3.  Preprocess the images and run them through the model.
4.  Print the predicted class for each image.

## Configuration

The main hyperparameters and configuration settings are located at the top of the `model.py` script. You can modify these to experiment with different settings.

```python
# -----------Config for alexnet + transformer - decoder-block --------------

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 64
num_classes = 10
num_epochs = 4
train_data_path = "data"
checkpoint_path = "alexnet_imagenette.pth"
num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
pin_memory = False if device.type == 'mps' else True
weight_decay = 0.0005
learning_rate = 1e-4
n_embd = 256
n_head = 16
n_layer = 16
dropout = 0.5
inference_data_path = "image"
image_size = 224
```
