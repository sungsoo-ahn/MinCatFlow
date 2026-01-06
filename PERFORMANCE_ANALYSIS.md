# MinCatFlow Performance Issues Analysis

## Executive Summary

After analyzing the MinCatFlow codebase, I've identified **12 critical issues** that could significantly impact model performance. These range from configuration mismatches to architectural choices and training instabilities.

---

## 🔴 Critical Issues

### 1. **Learning Rate Scheduler Not Used**
**Location**: `src/module/effcat_module.py:1006-1013`

**Problem**:
```python
def configure_optimizers(self):
    # TODO: Add af3 scheduler
    optimizer = torch.optim.AdamW(
        params=self.parameters(),
        lr=self.training_args["lr"],
        weight_decay=self.training_args["weight_decay"],
    )
    return {"optimizer": optimizer}  # ❌ No scheduler returned
```

The code defines a learning rate scheduler in `configs/optim/default.yaml`:
```yaml
lr_scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  mode: "min"
  factor: 0.5
  patience: 10
  min_lr: 1e-6
```

**But this scheduler is completely ignored!** The model trains with constant learning rate (1e-4), which can:
- Prevent fine-tuning in later epochs
- Get stuck in local minima
- Fail to converge properly

**Impact**: High - This severely limits the model's ability to refine predictions
**Fix**: Implement the scheduler in `configure_optimizers()`:
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

### 2. **Weight Decay Configuration Mismatch**
**Locations**:
- `configs/model/default.yaml:72` → `weight_decay: 0.0`
- `configs/optim/default.yaml:8` → `weight_decay: 0.01`

**Problem**:
The model uses `training_args["weight_decay"]` (0.0) instead of the optim config (0.01). Weight decay is a crucial regularization technique.

**Impact**: Medium - No regularization leads to potential overfitting
**Fix**: Use weight decay = 0.01 or higher, especially for large models (768 dim × 12 layers)

---

### 3. **Extreme Loss Weight Imbalance**
**Location**: `configs/model/default.yaml:59-69`

```yaml
training_args:
  prim_slab_coord_loss_weight: 1.0
  ads_coord_loss_weight: 0.1        # ❌ 10x lower!
  length_loss_weight: 1.0
  angle_loss_weight: 0.01            # ❌ 100x lower!
  supercell_matrix_loss_weight: 1.0
  scaling_factor_loss_weight: 1.0
  prim_slab_element_loss_weight: 5.0  # ❌ 5x higher!
```

**Problems**:
1. **Adsorbate severely under-weighted** (0.1): The model barely learns to predict adsorbate positions
2. **Angles nearly ignored** (0.01): Lattice angles are critical for crystal structure but weighted 100× less
3. **Element loss dominates** (5.0 in DNG mode): This can overshadow geometric learning

**Impact**: High - Poor adsorbate placement and lattice angle prediction
**Fix**: Re-balance weights based on loss magnitudes:
```yaml
prim_slab_coord_loss_weight: 1.0
ads_coord_loss_weight: 1.0          # Increase to equal weight
length_loss_weight: 1.0
angle_loss_weight: 0.1              # Increase from 0.01
prim_slab_element_loss_weight: 1.0  # Reduce from 5.0
```

---

### 4. **Supercell Matrix Prior Ignores Learned Statistics**
**Location**: `src/data/prior.py:127-133`

```python
# ❌ Commented out: learned distribution
# supercell_matrix_0_normalized = torch.randn(batch_size, 3, 3, ...)
# supercell_matrix_0 = self.denormalize_supercell(supercell_matrix_0_normalized)

# ✅ Actually used: identity + tiny noise
identity = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
noise = torch.randn(batch_size, 3, 3, device=device, dtype=dtype) * 0.1
supercell_matrix_0 = identity + noise
```

**Problem**:
The config provides carefully computed statistics from the training set:
```yaml
supercell_mean: [
  [-0.026726, 0.078970, -0.065390],
  [-0.133529, 0.188983, -0.132132],
  [-0.013621, -0.254989, 0.140161]
]
supercell_std: [
  [1.796400, 1.213596, 0.877900],
  [1.223385, 1.383904, 1.337392],
  [1.373271, 1.452034, 1.045282]
]
```

But the prior uses identity + noise, making flow matching much harder since the prior doesn't match the data distribution.

**Impact**: High - Poor supercell matrix predictions, harder optimization
**Fix**: Uncomment the learned prior or use a better initialization

---

### 5. **No Learning Rate Warmup**
**Location**: Training configuration

**Problem**:
Starting training at full learning rate (1e-4) without warmup can cause:
- Early training instability
- Poor initialization adaptation
- Suboptimal convergence trajectory

**Impact**: Medium
**Fix**: Add warmup for first 1000-5000 steps

---

### 6. **High Element Loss Weight in DNG Mode**
**Location**: `configs/model/default.yaml:69`

```yaml
prim_slab_element_loss_weight: 5.0  # Used when dng=True
```

**Problem**:
When DNG mode is enabled (`dng: true`), the element prediction loss is weighted 5× higher than coordinate losses. This can cause:
- Model focuses too much on element prediction
- Geometric structure learning suffers
- Imbalanced gradient magnitudes

**Impact**: Medium-High when using DNG mode
**Fix**: Reduce to 1.0 or lower, monitor relative loss magnitudes

---

### 7. **Adsorbate Coordinate Statistics Suspect**
**Location**: `configs/model/default.yaml:47-48`

```yaml
ads_coord_mean: [0.007402, -0.702398, 0.934607]
ads_coord_std: [8.247430, 9.035212, 9.979235]  # ❌ Very high!
```

**Problem**:
Compared to prim_slab coordinates:
```yaml
prim_slab_coord_mean: [3.298159, 4.591314, 5.311862]
prim_slab_coord_std: [2.842205, 3.540208, 4.057026]
```

Adsorbate std is **~3× larger**, suggesting either:
- Data preprocessing issues
- Outliers in the dataset
- Reference frame problems (adsorbates should be relative to slab)

**Impact**: Medium - Makes adsorbate prediction harder
**Fix**:
1. Verify adsorbate coordinates are in the correct reference frame
2. Check for outliers in dataset
3. Consider recomputing statistics after cleaning

---

### 8. **Validation Only Every 5 Epochs**
**Location**: `configs/model/default.yaml:75`

```yaml
validation_args:
  sample_every_n_epochs: 5  # ❌ Too infrequent
```

**Problem**:
- Can't detect early overfitting
- Learning rate scheduler (if implemented) updates too slowly
- Harder to debug training issues

**Impact**: Medium
**Fix**: Set to 1 for better monitoring, especially early in training

---

### 9. **Gradient Clipping May Be Suboptimal**
**Location**: `configs/train/default.yaml:27`

```yaml
gradient_clip_val: 10.0
gradient_clip_algorithm: "norm"
```

**Problem**:
The value 10.0 is arbitrary without knowing typical gradient norms. Could be:
- Too high → doesn't prevent gradient explosions
- Too low → excessive clipping hurts learning

**Impact**: Low-Medium
**Fix**:
1. Log gradient norms during training
2. Adjust based on observed values (typically 0.5-5.0 for well-normalized models)

---

### 10. **No Gradient Accumulation Used**
**Location**: `configs/train/default.yaml:25`

```yaml
accumulate_grad_batches: 1  # ❌ No accumulation
```

**Problem**:
With batch size 32, the effective batch size is small for a large model. This can cause:
- Noisy gradient estimates
- Training instability
- Slower convergence

**Impact**: Medium
**Fix**: Set to 2-4 to effectively double/quadruple batch size without OOM

---

## 🟡 Moderate Issues

### 11. **Positional Encoding Disabled**
**Location**: `configs/model/default.yaml:13`

```yaml
atom_encoder_positional_encoding: false  # positional encoding not used
```

**Problem**:
Transformer models benefit from positional information. Without it, the model is permutation-invariant, which might not be ideal for spatially-organized structures.

**Impact**: Low-Medium
**Fix**: Try enabling positional encoding (sinusoidal or learned)

---

### 12. **Deterministic Mode Enabled**
**Location**: `configs/train/default.yaml:4`

```yaml
deterministic: true
```

**Problem**:
From `src/run.py:60-61`:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # ❌ Disables optimizations
```

This disables cuDNN optimizations, making training **slower** (10-30% slower) without improving reproducibility much in modern PyTorch.

**Impact**: Low (performance, not accuracy)
**Fix**: Set `deterministic: false` and use fixed seed only

---

## 📊 Recommendations Summary

### Immediate Fixes (High Impact)
1. ✅ **Implement learning rate scheduler** - Critical for convergence
2. ✅ **Rebalance loss weights** - Especially increase adsorbate and angle weights
3. ✅ **Use learned supercell prior** - Match training distribution
4. ✅ **Add weight decay** - Prevent overfitting

### Medium Priority
5. ✅ **Reduce element loss weight** (if using DNG mode)
6. ✅ **Verify adsorbate coordinate statistics** - Check preprocessing
7. ✅ **Increase validation frequency** to every epoch
8. ✅ **Add gradient accumulation** for larger effective batch size

### Low Priority
9. ✅ **Add learning rate warmup**
10. ✅ **Tune gradient clipping** based on observed norms
11. ✅ **Experiment with positional encoding**
12. ✅ **Disable deterministic mode** for faster training

---

## 🔬 Debugging Checklist

To diagnose performance issues, log these metrics:

```python
# Add to training_step or on_train_epoch_end
self.log("debug/grad_norm", grad_norm)
self.log("debug/prim_slab_loss_raw", prim_slab_loss.mean())
self.log("debug/ads_loss_raw", ads_loss.mean())
self.log("debug/element_loss_raw", element_loss.mean())
self.log("debug/lr", self.trainer.optimizers[0].param_groups[0]['lr'])
```

### Key Metrics to Monitor
1. **Loss magnitude ratios**: Are they balanced after weighting?
2. **Gradient norms**: Are they stable? (should be 0.1-10)
3. **Validation metrics**: RMSD, match rate, structural validity
4. **Learning rate**: Is it decreasing when stuck?

---

## 📝 Example Fixed Configuration

```yaml
# configs/model/default.yaml (partial)
training_args:
  lr: 1e-4
  weight_decay: 0.01  # Changed from 0.0
  train_multiplicity: 1
  loss_type: "l2"
  prim_slab_coord_loss_weight: 1.0
  ads_coord_loss_weight: 1.0      # Changed from 0.1
  length_loss_weight: 1.0
  angle_loss_weight: 0.1          # Changed from 0.01
  supercell_matrix_loss_weight: 1.0
  scaling_factor_loss_weight: 1.0
  prim_slab_element_loss_weight: 1.0  # Changed from 5.0

validation_args:
  sample_every_n_epochs: 1  # Changed from 5

# configs/train/default.yaml (partial)
pl_trainer:
  accumulate_grad_batches: 2  # Changed from 1
  gradient_clip_val: 1.0      # Changed from 10.0

deterministic: false  # Changed from true
```

---

## 🎯 Expected Improvements

After implementing these fixes:

1. **Convergence**: Faster and more stable with scheduler + warmup
2. **Adsorbate accuracy**: Significant improvement from balanced loss weights
3. **Lattice angles**: Better prediction from 10× higher weight
4. **Generalization**: Less overfitting with weight decay
5. **Supercell matrix**: More accurate predictions with proper prior

---

## Additional Investigation Needed

1. **Dataset quality**: Check for corrupted samples, outliers
2. **Loss scale calibration**: Log raw loss values to verify weighting
3. **Architecture ablations**: Test smaller/larger models
4. **Sampling steps**: Try different num_sampling_steps (16, 50, 100)
5. **Prior distributions**: Validate that statistics were computed correctly

---

**Generated**: 2026-01-06
**Analyzer**: Claude Code Analysis
