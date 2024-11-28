# Copyright (c) 2024, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import os

import numpy as np
import torch
from torch.utils.data import DataLoader


def pad_sequence(batch):
    """Makes all tensor in a batch the same length by padding with zeros."""
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    """Collate function for variable-length batches."""
    # Iterate
    tensors, targets, vids = [], [], []
    for waveform, label, vid in batch:
        tensors += [waveform]
        targets += [torch.tensor(label)]
        vids += [torch.tensor(vid)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    vids = torch.stack(vids)

    return tensors, targets, vids


def train_model(model, train_loader, criterion, optimizer, device, nb_epochs):
    """Trains the given model with the provided parameters."""
    loss_list = []
    for e in range(nb_epochs):
        # Accumulated loss
        acc_loss = 0

        wandb.watch(model)
        # Iterate by mini-batches
        for inputs, targets in train_loader:
            # Forward pass
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.forward(inputs)

            # Compute loss
            loss = criterion(
                outputs.squeeze(), targets
            )  # Sure about the squeeze ? W/o there is a bug
            acc_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()  # Zero grad
            loss.backward()  # Backpropagation
            optimizer.step()  # SGD optimizer

        # Log
        wandb.log({"loss": acc_loss, "epoch": e})


def test_model(model, test_loader, device):
    """Test trained model and prints out test error and accuracy."""

    nb_test_errors, nb_test_samples = 0, 0
    model.eval()

    for input, targets in iter(test_loader):
        input, targets = input.to(device), targets.to(device)
        output = model(input)

        wta = torch.argmax(output.data, 1).view(-1)

        for i in range(targets.size(0)):
            nb_test_samples += 1
            if wta[i] != targets[i]:
                nb_test_errors += 1

    test_error = 100 * nb_test_errors / nb_test_samples
    print(f"Test error: {test_error:.02f}% ({nb_test_errors}/{nb_test_samples})")
    print(
        f"Accuracy: {round((nb_test_samples-nb_test_errors)*100/nb_test_samples, 2)}% ({nb_test_samples-nb_test_errors}/{nb_test_samples})"
    )


def manual_reflect_pad(wav, max_len):
    """
    Pads an audio waveform tensor with reflective padding to a specified maximum length.

    This function ensures that the audio waveform is symmetrically padded on both sides
    to reach the desired maximum length. If the waveform is already longer than or equal
    to the maximum length, no padding is applied. The padding applied is reflective, meaning
    the waveform is mirrored at the edges for padding, creating a smooth transition.

    Parameters:
    - wav (torch.Tensor): A 1D tensor representing the audio waveform to be padded.
    - max_length (int): The target length for the padded waveform. If the original waveform
    is longer than this value, it will not be shortened.

    Returns:
    - torch.Tensor: The padded waveform as a 1D tensor of length `max_length`.

    Examples:
    >>> waveform = torch.tensor([1, 2, 3, 4, 5])
    >>> padded_waveform = manual_reflect_pad(waveform, 10)
    >>> print(padded_waveform)
    tensor([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])

    Note:
    This function is useful for preparing batches of audio data for neural network
    models, where inputs need to be of uniform size.
    """
    current_len = wav.shape[0]
    if current_len >= max_len:
        return wav[:max_len]

    while len(wav) < max_len:
        pad_len = max_len - len(wav)
        # Reflect pad on both sides
        if len(wav) > 1:  # Normal case
            pad_amount = min(len(wav) - 1, pad_len // 2)
            pad_left = wav[1 : 1 + pad_amount].flip(dims=[0])
            pad_right = wav[-pad_amount - 1 : -1].flip(dims=[0])
            wav = torch.cat([pad_left, wav, pad_right], dim=0)
        else:  # Edge case for very short waveforms
            wav = wav.repeat((pad_len + 1,))

    # Ensure the waveform is exactly max_len
    return wav[:max_len]


def manual_repeat_pad(wav, max_length):
    """
    Pads an audio waveform tensor by repeating its content to a specified maximum length.

    This function ensures that the audio waveform is padded by repeating its content from
    the start to reach the desired maximum length. If the waveform is already longer than
    or equal to the maximum length, no padding is applied. The repetition is seamless and
    maintains the original order of the waveform.

    Parameters:
    - wav (torch.Tensor): A 1D tensor representing the audio waveform to be padded.
    - max_length (int): The target length for the padded waveform. If the original waveform
      is longer than this value, it will not be shortened.

    Returns:
    - torch.Tensor: The padded waveform as a 1D tensor of length `max_length`.

    Examples:
    >>> waveform = torch.tensor([1, 2, 3, 4, 5])
    >>> padded_waveform = manual_repeat_pad(waveform, 10)
    >>> print(padded_waveform)
    tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

    Note:
    This function is useful for preparing batches of audio data for neural network
    models, where inputs need to be of uniform size. It repeats the waveform to fill
    the required length, which may alter the natural end of the audio if not aligned
    with the maximum length.
    """
    current_length = wav.shape[0]
    if current_length >= max_length:
        return wav

    repeats = max_length // current_length
    extra = max_length % current_length

    wav_repeated = wav.repeat(repeats)
    wav_padded = torch.cat([wav_repeated, wav[:extra]])

    return wav_padded


def get_statistics_stack(datamodule, save_path):
    """Computes the mean and std over the train DataLoader."""
    # Get train dataset
    datamodule.setup()
    train_dataset = datamodule.train_dataset

    # Pass entire dataset
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=True,
        collate_fn=collate_fn,
    )
    x, _, _ = next(iter(train_dataloader))

    # Compute statistics
    means = x.mean(dim=0, keepdim=True)
    stds = x.std(dim=0, keepdim=True)

    # Stack
    stack = {
        "means": means.numpy(),
        "stds": stds.numpy(),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **stack)

    del x
    del train_dataset
    del train_dataloader
    del stack
