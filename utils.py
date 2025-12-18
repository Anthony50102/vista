"""
Shared utilities for HW2D surrogate modeling experiments.
"""
import torch
import torch.nn as nn
import numpy as np
import h5py
import os
import json
import datetime
import random
from torch.utils.data import Dataset, DataLoader


# ============== Data Processing ==============

def process_state_data(density, potential):
    """Process density and potential into state tensor."""
    return np.stack([density, potential], axis=1)  # (T, 2, H, W)


def process_derived_data(gamma_n, gamma_c):
    """Process gamma_n and gamma_c into derived tensor."""
    return np.stack([gamma_n, gamma_c], axis=1)  # (T, 2)


def load_h5_data(filepath, truncation=1.0):
    """Load data from HDF5 file with optional truncation."""
    with h5py.File(filepath, 'r') as f:
        end_idx = int(f['density'].shape[0] * truncation)
        data = {
            'density': f['density'][:end_idx],
            'potential': f['phi'][:end_idx],
            'gamma_n': f['gamma_n'][:end_idx],
            'gamma_c': f['gamma_c'][:end_idx],
        }
    return data


# ============== Datasets ==============

class StateTrajectoryDataset(Dataset):
    """Dataset for state-only trajectory data (for state model training)."""
    
    def __init__(self, data_path, mode='train', train_split=0.8, val_split=0.5):
        if isinstance(data_path, str) and data_path.endswith('.npz'):
            self.data = np.load(data_path)['data']
        else:
            # Assume it's a directory - find all train_*.npz files
            files = [os.path.join(data_path, f) for f in os.listdir(data_path)
                     if f.startswith('train_') and f.endswith('.npz') and 'output' not in f]
            self.data = np.concatenate([np.load(f)['data'] for f in files], axis=0)
        
        total = len(self.data)
        train_end = int(total * train_split)
        val_end = train_end + int((total - train_end) * val_split)
        
        if mode == 'train':
            self.data = self.data[:train_end]
        elif mode == 'val':
            self.data = self.data[train_end:val_end]
        elif mode == 'test':
            self.data = self.data[val_end:]
    
    def __len__(self):
        return 1  # Return whole trajectory
    
    def __getitem__(self, idx):
        return {'data': torch.from_numpy(self.data).float()}


class StateDerivedDataset(Dataset):
    """Dataset for state → derived quantity mapping."""
    
    def __init__(self, data_path, mode='train', train_split=0.8, val_split=0.5):
        data = np.load(data_path)
        state = data['state']
        derived = data['derived']
        
        total = len(state)
        train_end = int(total * train_split)
        val_end = train_end + int((total - train_end) * val_split)
        
        if mode == 'train':
            self.state, self.derived = state[:train_end], derived[:train_end]
        elif mode == 'val':
            self.state, self.derived = state[train_end:val_end], derived[train_end:val_end]
        elif mode == 'test':
            self.state, self.derived = state[val_end:], derived[val_end:]
    
    def __len__(self):
        return len(self.state)
    
    def __getitem__(self, idx):
        return {
            'state': torch.from_numpy(self.state[idx]).float(),
            'derived': torch.from_numpy(self.derived[idx]).float()
        }


class TrajectoryWithDerivedDataset(Dataset):
    """Dataset for full trajectories with derived quantities."""
    
    def __init__(self, data_path, mode='train'):
        files = [os.path.join(data_path, f) for f in os.listdir(data_path)
                 if f.startswith('output_train_') and f.endswith('.npz')]
        
        train_end = int(len(files) * 0.8)
        self.files = files[:train_end] if mode == 'train' else files[train_end:]
    
    def __len__(self):
        return max(len(self.files), 1)
    
    def __getitem__(self, idx):
        if len(self.files) == 0:
            raise RuntimeError("No trajectory files found")
        data = np.load(self.files[idx % len(self.files)])
        return {
            'state': torch.from_numpy(data['state']).float(),
            'derived': torch.from_numpy(data['derived']).float()
        }


# ============== Model Save/Load ==============

def save_model(model, name, save_dir='.', metadata=None):
    """
    Save model with metadata.
    
    Args:
        model: PyTorch model
        name: Model name (e.g., 'state_model', 'derived_fno', 'derived_cnn')
        save_dir: Directory to save to
        metadata: Optional dict with training info
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': timestamp,
        'name': name,
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    # Save
    filename = f"{name}_{timestamp}.pt"
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    
    # Also save as 'latest' for easy loading
    latest_path = os.path.join(save_dir, f"{name}_latest.pt")
    torch.save(checkpoint, latest_path)
    
    print(f"Saved: {filepath}")
    return filepath


def load_model(model, name, save_dir='.', version='latest'):
    """
    Load model from checkpoint.
    
    Args:
        model: PyTorch model (architecture must match)
        name: Model name
        save_dir: Directory to load from
        version: 'latest' or specific timestamp
    """
    if version == 'latest':
        filepath = os.path.join(save_dir, f"{name}_latest.pt")
    else:
        filepath = os.path.join(save_dir, f"{name}_{version}.pt")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded: {filepath}")
    if 'metadata' in checkpoint:
        print(f"  Metadata: {checkpoint['metadata']}")
    
    return checkpoint.get('metadata', {})


def list_checkpoints(name, save_dir='.'):
    """List all available checkpoints for a model."""
    files = [f for f in os.listdir(save_dir) if f.startswith(name) and f.endswith('.pt')]
    return sorted(files)


# ============== Training Utilities ==============

def train_epoch_single_step(model, loader, optimizer, device):
    """Train one epoch on single-step predictions."""
    model.train()
    total_loss, n_samples = 0, 0
    
    for batch in loader:
        traj = batch['data'].to(device)
        B, T, C, H, W = traj.shape
        
        # All consecutive pairs
        inputs = traj[:, :-1].reshape(-1, C, H, W)
        targets = traj[:, 1:].reshape(-1, C, H, W)
        
        # Mini-batch through trajectory
        perm = torch.randperm(inputs.shape[0])
        for i in range(0, len(perm), 32):
            idx = perm[i:i+32]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(inputs[idx]), targets[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
            n_samples += len(idx)
    
    return total_loss / n_samples


def train_epoch_rollout(model, loader, optimizer, device, rollout_len, ss_prob=0.0):
    """Train one epoch with rollout and optional scheduled sampling."""
    model.train()
    total_loss, n_rollouts = 0, 0
    
    for batch in loader:
        traj = batch['data'].to(device)
        B, T, C, H, W = traj.shape
        
        max_start = T - rollout_len - 1
        if max_start < 0:
            continue
        
        start = random.randint(0, max_start)
        optimizer.zero_grad()
        
        state = traj[:, start]
        loss = 0
        
        for step in range(rollout_len):
            pred = model(state)
            target = traj[:, start + step + 1]
            loss += nn.functional.mse_loss(pred, target)
            
            # Scheduled sampling
            if random.random() < ss_prob:
                state = pred.detach()
            else:
                state = target
        
        loss = loss / rollout_len
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_rollouts += 1
    
    return total_loss / max(n_rollouts, 1)


def train_epoch_direct(model, loader, optimizer, device):
    """Train one epoch for direct mapping (state → derived)."""
    model.train()
    total_loss, n_samples = 0, 0
    
    for batch in loader:
        state = batch['state'].to(device)
        derived = batch['derived'].to(device)
        
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(state), derived)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * state.size(0)
        n_samples += state.size(0)
    
    return total_loss / n_samples


@torch.no_grad()
def validate_direct(model, loader, device):
    """Validate direct mapping model."""
    model.eval()
    total_loss, n_samples = 0, 0
    
    for batch in loader:
        state = batch['state'].to(device)
        derived = batch['derived'].to(device)
        loss = nn.functional.mse_loss(model(state), derived)
        total_loss += loss.item() * state.size(0)
        n_samples += state.size(0)
    
    return total_loss / n_samples


# ============== Inference ==============

@torch.no_grad()
def rollout_state(model, initial_state, num_steps, device):
    """
    Autoregressive state rollout.
    
    Args:
        model: State prediction model
        initial_state: (C, H, W) numpy array
        num_steps: Number of steps to predict
        device: torch device
    
    Returns:
        (T, C, H, W) numpy array of predicted states
    """
    model.eval()
    C, H, W = initial_state.shape
    recon = np.zeros((num_steps, C, H, W))
    recon[0] = initial_state
    
    state = torch.from_numpy(initial_state).float().to(device)
    
    for t in range(1, num_steps):
        state = model(state.unsqueeze(0)).squeeze(0)
        recon[t] = state.cpu().numpy()
    
    return recon


@torch.no_grad()
def rollout_combined(state_model, derived_model, initial_state, num_steps, device):
    """
    Combined rollout: predict both states and derived quantities.
    
    Returns:
        state_recon: (T, C, H, W)
        derived_recon: (T, 2)
    """
    state_model.eval()
    derived_model.eval()
    
    C, H, W = initial_state.shape
    state_recon = np.zeros((num_steps, C, H, W))
    derived_recon = np.zeros((num_steps, 2))
    
    state_recon[0] = initial_state
    state = torch.from_numpy(initial_state).float().to(device)
    
    # Initial derived prediction
    derived_recon[0] = derived_model(state.unsqueeze(0)).squeeze(0).cpu().numpy()
    
    for t in range(1, num_steps):
        state = state_model(state.unsqueeze(0)).squeeze(0)
        state_recon[t] = state.cpu().numpy()
        derived_recon[t] = derived_model(state.unsqueeze(0)).squeeze(0).cpu().numpy()
    
    return state_recon, derived_recon


# ============== Evaluation ==============

def compute_mse_over_time(ground_truth, reconstruction):
    """Compute MSE at each timestep."""
    T = ground_truth.shape[0]
    return np.array([np.mean((ground_truth[t] - reconstruction[t])**2) for t in range(T)])


def compute_mae_over_time(ground_truth, reconstruction):
    """Compute MAE at each timestep."""
    T = ground_truth.shape[0]
    return np.array([np.mean(np.abs(ground_truth[t] - reconstruction[t])) for t in range(T)])


def print_summary(name, mse_array):
    """Print summary statistics for an MSE array."""
    print(f"{name}:")
    print(f"  Initial: {mse_array[0]:.6f}")
    print(f"  Final:   {mse_array[-1]:.6f}")
    print(f"  Mean:    {mse_array.mean():.6f}")
    print(f"  Max:     {mse_array.max():.6f}")
