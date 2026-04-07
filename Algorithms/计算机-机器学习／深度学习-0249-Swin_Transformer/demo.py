"""Swin Transformer minimal runnable MVP on sklearn digits.

This script implements key Swin ideas directly in source code:
- patch embedding
- window-based multi-head self-attention (W-MSA)
- shifted window attention (SW-MSA) with attention mask
- MLP + residual + LayerNorm blocks

No external dataset download is required.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset



def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Split (B, H, W, C) into windows: (num_windows*B, ws*ws, C)."""
    bsz, height, width, channels = x.shape
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError(
            f"H/W must be divisible by window_size, got H={height}, W={width}, ws={window_size}"
        )
    x = x.view(
        bsz,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    # Rearrange so each contiguous chunk is one local window.
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size * window_size, channels)
    return windows



def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reverse window_partition back to (B, H, W, C)."""
    num_windows_per_image = (height // window_size) * (width // window_size)
    bsz = windows.shape[0] // num_windows_per_image
    channels = windows.shape[-1]
    x = windows.view(
        bsz,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(bsz, height, width, channels)
    return x



def build_shifted_window_mask(
    input_resolution: Tuple[int, int],
    window_size: int,
    shift_size: int,
) -> Optional[torch.Tensor]:
    """Create attention mask for SW-MSA; shape (nW, ws*ws, ws*ws)."""
    if shift_size == 0:
        return None

    height, width = input_resolution
    img_mask = torch.zeros((1, height, width, 1))

    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )

    cnt = 0
    for h_sl in h_slices:
        for w_sl in w_slices:
            img_mask[:, h_sl, w_sl, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size=window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
    attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches by Conv2d stride=patch_size."""

    def __init__(
        self,
        img_size: int = 8,
        patch_size: int = 2,
        in_channels: int = 1,
        embed_dim: int = 48,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")
        self.grid_size = img_size // patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)  # (B, C, H', W')
        height, width = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, L, C)
        return x, (height, width)


class WindowAttention(nn.Module):
    """Window based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table: (2*ws-1)^2 offsets for each head.
        relative_coords_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_coords_size, num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.flatten(1)  # (2, N)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (num_windows*B, N, C)
        bsz_windows, num_tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(bsz_windows, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # (3, BW, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (BW, heads, N, N)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            num_tokens, num_tokens, self.num_heads
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if attn_mask is not None:
            num_windows = attn_mask.shape[0]
            attn = attn.view(
                bsz_windows // num_windows,
                num_windows,
                self.num_heads,
                num_tokens,
                num_tokens,
            )
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, num_tokens, num_tokens)

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(bsz_windows, num_tokens, channels)
        out = self.proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwinBlock(nn.Module):
    """One Swin block: (W-MSA/SW-MSA) + MLP with pre-norm residuals."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        height, width = input_resolution
        if min(height, width) <= window_size:
            window_size = min(height, width)
            shift_size = 0
        if shift_size >= window_size:
            raise ValueError(
                f"shift_size must be smaller than window_size, got {shift_size} >= {window_size}"
            )

        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)

        attn_mask = build_shifted_window_mask(
            input_resolution=input_resolution,
            window_size=window_size,
            shift_size=shift_size,
        )
        if attn_mask is None:
            self.register_buffer("attn_mask", None)
        else:
            self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H*W, C)
        bsz, num_tokens, channels = x.shape
        height, width = self.input_resolution
        if num_tokens != height * width:
            raise ValueError(
                f"token count mismatch: got {num_tokens}, expected {height * width}"
            )

        shortcut = x
        x = self.norm1(x)
        x = x.view(bsz, height, width, channels)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, attn_mask=self.attn_mask)

        shifted_x = window_reverse(
            attn_windows,
            window_size=self.window_size,
            height=height,
            width=width,
        )

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        x = x.view(bsz, height * width, channels)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class TinySwinClassifier(nn.Module):
    """Two-block tiny Swin classifier for 8x8 grayscale digit images."""

    def __init__(
        self,
        img_size: int = 8,
        patch_size: int = 2,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 48,
        num_heads: int = 4,
        window_size: int = 2,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        grid_h = img_size // patch_size
        grid_w = img_size // patch_size
        input_resolution = (grid_h, grid_w)

        self.block1 = SwinBlock(
            dim=embed_dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=4.0,
        )
        self.block2 = SwinBlock(
            dim=embed_dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,
            mlp_ratio=4.0,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.patch_embed(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average over tokens
        x = self.head(x)
        return x



def load_dataset(
    test_ratio: float = 0.25,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")

    digits = load_digits()
    x = digits.images.astype(np.float32) / 16.0  # (N, 8, 8)
    y = digits.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y,
    )

    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )



def build_dataloaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            total_correct += int((pred == yb).sum().item())
            total_count += yb.numel()
            total_loss += float(loss.item() * yb.numel())

    mean_loss = total_loss / max(total_count, 1)
    mean_acc = total_correct / max(total_count, 1)
    return mean_loss, mean_acc



def majority_class_baseline(y_train: torch.Tensor, y_test: torch.Tensor) -> Tuple[int, float]:
    majority_class = int(torch.mode(y_train).values.item())
    baseline_acc = float((y_test == majority_class).float().mean().item())
    return majority_class, baseline_acc



def main() -> None:
    set_global_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, x_test, y_test = load_dataset(test_ratio=0.25, random_state=42)
    train_loader, test_loader = build_dataloaders(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        batch_size=64,
    )

    model = TinySwinClassifier(
        img_size=8,
        patch_size=2,
        in_channels=1,
        num_classes=10,
        embed_dim=48,
        num_heads=4,
        window_size=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    epochs = 20
    print(f"device: {device}")
    print(f"train shape: X={tuple(x_train.shape)}, y={tuple(y_train.shape)}")
    print(f"test  shape: X={tuple(x_test.shape)}, y={tuple(y_test.shape)}")
    print(f"model: TinySwinClassifier (params={sum(p.numel() for p in model.parameters())})")
    print(f"optimizer: AdamW(lr=2e-3, weight_decay=1e-4), epochs={epochs}")

    final_train_acc = 0.0
    final_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        test_loss, test_acc = run_epoch(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        final_train_acc = train_acc
        final_test_acc = test_acc

        if epoch == 1 or epoch % 4 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
            )

    majority_class, baseline_acc = majority_class_baseline(y_train, y_test)
    print(f"majority baseline: class={majority_class}, acc={baseline_acc:.4f}")

    model.eval()
    with torch.no_grad():
        sample_logits = model(x_test[:8].to(device))
        sample_prob = torch.softmax(sample_logits, dim=1).cpu().numpy()
        sample_pred = np.argmax(sample_prob, axis=1)

    print("sample predictions (first 8 test samples):")
    for i in range(8):
        top_prob = float(np.max(sample_prob[i]))
        print(
            f"  idx={i:02d} true={int(y_test[i].item())} "
            f"pred={int(sample_pred[i])} top_prob={top_prob:.3f}"
        )

    print(f"final train_acc={final_train_acc:.4f}, final test_acc={final_test_acc:.4f}")

    if not np.isfinite(final_test_acc):
        raise RuntimeError("test accuracy is not finite")
    if final_test_acc < 0.88:
        raise RuntimeError(f"test accuracy too low: {final_test_acc:.4f} < 0.88")
    if final_test_acc < baseline_acc + 0.35:
        raise RuntimeError(
            "model did not beat majority baseline by >= 0.35: "
            f"test={final_test_acc:.4f}, baseline={baseline_acc:.4f}"
        )

    print("All checks passed.")


if __name__ == "__main__":
    main()
