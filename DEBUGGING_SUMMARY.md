# MinCatFlow Debugging Summary

## Overview

This document consolidates findings from comprehensive code analysis covering both **hyperparameter issues** and **implementation bugs**.

---

## 📊 Analysis Files

1. **PERFORMANCE_ANALYSIS.md** - Hyperparameter and configuration issues
2. **FLOW_MATCHING_BUGS.md** - Implementation bugs in flow matching code
3. **claude.md** - Complete project documentation

---

## 🎯 Top Priority Fixes (By Impact)

### 1. **DNG Mode Mask Bug** 🔴 CRITICAL
**File**: `src/module/effcat_module.py:194-270`
**Type**: Implementation Bug
**Impact**: Breaks DNG mode completely

The code creates two different masks that should be identical:
- `prim_slab_atom_mask`: Dynamically sampled (each sample has different atom count)
- `feats["prim_slab_atom_pad_mask"]`: From original data (all samples same pattern)

The network uses the wrong mask, causing incorrect attention and loss computation.

**Fix**:
```python
# Line 270, replace with:
feats["prim_slab_atom_pad_mask"] = prim_slab_atom_mask
feats["prim_slab_token_pad_mask"] = prim_slab_atom_mask
# Remove lines 216-269 (don't adjust/expand separately)
```

---

### 2. **Learning Rate Scheduler Not Implemented** 🔴 CRITICAL
**File**: `src/module/effcat_module.py:1006-1013`
**Type**: Configuration Bug
**Impact**: Prevents convergence and fine-tuning

The scheduler is defined in config but never used. Model trains with constant LR.

**Fix**:
```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        params=self.parameters(),
        lr=self.training_args["lr"],
        weight_decay=self.training_args["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val/total_loss",
        }
    }
```

---

### 3. **Supercell Prior Ignores Learned Statistics** 🔴 HIGH
**File**: `src/data/prior.py:127-133`
**Type**: Implementation Bug
**Impact**: Makes optimization much harder

Uses identity + noise instead of learned distribution from training data.

**Fix**:
```python
# Uncomment lines 127-129:
supercell_matrix_0_normalized = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
supercell_matrix_0 = self.denormalize_supercell(supercell_matrix_0_normalized)
# Delete lines 131-133
```

---

### 4. **Loss Weight Imbalance** 🔴 HIGH
**File**: `configs/model/default.yaml:62-69`
**Type**: Hyperparameter Issue
**Impact**: Model ignores adsorbates and angles

Current weights severely under-weight important predictions:
```yaml
ads_coord_loss_weight: 0.1      # 10x too low
angle_loss_weight: 0.01          # 100x too low
prim_slab_element_loss_weight: 5.0  # 5x too high (DNG mode)
```

**Fix**:
```yaml
prim_slab_coord_loss_weight: 1.0
ads_coord_loss_weight: 1.0       # Increase from 0.1
length_loss_weight: 1.0
angle_loss_weight: 0.1           # Increase from 0.01
supercell_matrix_loss_weight: 1.0
scaling_factor_loss_weight: 1.0
prim_slab_element_loss_weight: 1.0  # Reduce from 5.0
```

---

### 5. **Weight Decay = 0** 🟡 MEDIUM
**File**: `configs/model/default.yaml:72`
**Type**: Configuration Issue
**Impact**: Overfitting on large model

**Fix**:
```yaml
weight_decay: 0.01  # Change from 0.0
```

---

## 📋 Complete Fix Checklist

### Immediate (Do First)

- [ ] Fix DNG mask bug (effcat_module.py:270)
- [ ] Implement LR scheduler (effcat_module.py:1006)
- [ ] Use learned supercell prior (prior.py:127)
- [ ] Rebalance loss weights (configs/model/default.yaml:62-69)
- [ ] Set weight_decay to 0.01 (configs/model/default.yaml:72)

### High Priority

- [ ] Increase validation frequency to every epoch (configs/model/default.yaml:75)
- [ ] Add gradient accumulation = 2 (configs/train/default.yaml:25)
- [ ] Reduce gradient clipping to 1.0 (configs/train/default.yaml:27)
- [ ] Verify adsorbate coordinate statistics (check preprocessing)

### Medium Priority

- [ ] Add LR warmup (1000-5000 steps)
- [ ] Disable deterministic mode (configs/train/default.yaml:4)
- [ ] Increase ODE epsilon from 1e-5 to 1e-3 (flow.py:711)
- [ ] Consider element prediction index fix (layers.py:281)

### Optional

- [ ] Enable positional encoding (configs/model/default.yaml:13)
- [ ] Improve lattice angle normalization
- [ ] Add better ODE solver (RK4 or adaptive)

---

## 🧪 Validation After Fixes

### Metrics to Monitor

```python
# Add to training loop:
self.log("debug/grad_norm", grad_norm)
self.log("debug/lr", self.trainer.optimizers[0].param_groups[0]['lr'])

# For DNG mode specifically:
self.log("debug/atom_counts", batch["prim_slab_atom_pad_mask"].sum(dim=1).float().mean())
self.log("debug/prim_validity", val_prim_structural_validity)

# Loss magnitudes before weighting:
self.log("debug/prim_slab_loss_raw", prim_slab_coord_loss.mean())
self.log("debug/ads_loss_raw", ads_coord_loss.mean())
self.log("debug/angle_loss_raw", angle_loss.mean())
```

### Expected Improvements

1. **Convergence**: Should see steady decrease in validation loss
2. **Adsorbate accuracy**: Significant improvement from balanced weights
3. **Lattice angles**: Better predictions from higher weight
4. **Supercell matrix**: More accurate predictions with proper prior
5. **DNG mode**: Actually works (if using dng=True)
6. **Training stability**: No gradient explosions, smooth loss curves

### Warning Signs

- NaN losses → Check gradient clipping and numerical stability
- Validation loss plateau → Verify LR scheduler is working
- Zero structural validity (DNG) → Mask bug not fixed properly
- All supercell matrices near identity → Prior not fixed

---

## 💡 Recommended Training Workflow

### Step 1: Fix Critical Bugs
Start with the 5 top priority fixes above. Test on a small subset first.

### Step 2: Baseline Training (dng=False)
```bash
# Train without DNG mode first
python src/run.py \
    model.flow_model_args.dng=false \
    train.pl_trainer.max_epochs=50 \
    logging.wandb.name="baseline_fixes"
```

Monitor:
- Loss components are balanced
- LR decreases when stuck
- Structural validity > 0.5
- RMSD improves over time

### Step 3: Enable DNG Mode
```bash
# After baseline works
python src/run.py \
    model.flow_model_args.dng=true \
    logging.wandb.name="dng_mode_fixed"
```

Monitor:
- Different samples have different atom counts
- Element prediction loss is reasonable (not dominating)
- Structural validity remains high

### Step 4: Full Training
```bash
# Train to convergence
python src/run.py \
    model.flow_model_args.dng=true \
    train.pl_trainer.max_epochs=1000 \
    train.pl_trainer.accumulate_grad_batches=2
```

---

## 🔍 Debugging Tips

### If model still underperforms:

1. **Check data quality**
   ```python
   # Add to DataModule:
   def on_after_batch_transfer(self, batch, dataloader_idx):
       print(f"Coords range: {batch['prim_slab_cart_coords'].min():.2f} to {batch['prim_slab_cart_coords'].max():.2f}")
       print(f"Lattice range: {batch['lattice'].min():.2f} to {batch['lattice'].max():.2f}")
       return batch
   ```

2. **Verify statistics**
   ```python
   # Recompute and compare:
   prim_coords_mean = train_data['prim_slab_cart_coords'].mean(dim=(0,1))
   prim_coords_std = train_data['prim_slab_cart_coords'].std(dim=(0,1))
   # Should match config values
   ```

3. **Log intermediate activations**
   ```python
   # In AtomAttentionEncoder.forward():
   self.log("debug/encoder_norm", h_atoms.norm(dim=-1).mean())

   # In AtomAttentionDecoder.forward():
   self.log("debug/decoder_norm", x_global.norm(dim=-1).mean())
   ```

4. **Visualize samples**
   ```python
   # Save generated structures and compare to ground truth
   from ase import Atoms
   atoms = Atoms(numbers=elements, positions=coords, cell=lattice, pbc=True)
   atoms.write(f"sample_{i}.cif")
   ```

---

## 📖 Related Documentation

- **claude.md**: Full project documentation and architecture guide
- **PERFORMANCE_ANALYSIS.md**: All hyperparameter issues (12 total)
- **FLOW_MATCHING_BUGS.md**: All implementation bugs (5 total)
- **configs/**: Configuration files with inline comments

---

## ✅ Success Criteria

After applying all fixes, you should see:

1. ✅ Training loss steadily decreases (no plateaus)
2. ✅ Validation loss follows training loss (no overfitting)
3. ✅ Learning rate automatically decreases when stuck
4. ✅ Structural validity rate > 0.7 (ideally > 0.9)
5. ✅ Match rate > 0.5 for RMSD metrics (if dng=False)
6. ✅ Diverse atom counts in DNG mode
7. ✅ Supercell matrices not all near-identity
8. ✅ Gradient norms stable (0.1 - 10 range)
9. ✅ No NaN or Inf losses
10. ✅ Adsorbate positions meaningful (not all zeros)

---

## 🚀 Quick Start

```bash
# 1. Apply the 5 critical fixes (see files above)

# 2. Test with small dataset
python src/run.py \
    data.batch_size.train=8 \
    train.pl_trainer.max_epochs=10 \
    train.pl_trainer.fast_dev_run=true

# 3. Full training
python src/run.py
```

---

**Last Updated**: 2026-01-06
**Total Issues Found**: 17 (5 implementation bugs + 12 hyperparameter issues)
**Critical Issues**: 5
**High Priority**: 4
**Medium Priority**: 8
