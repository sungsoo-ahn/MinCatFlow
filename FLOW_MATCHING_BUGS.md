# Flow Matching Implementation Bugs

## Executive Summary

After deep analysis of the flow matching implementation, I found **3 critical bugs** and **2 moderate issues** that affect model correctness and performance.

---

## 🔴 Critical Bugs

### 1. **DNG Mode: Mask Inconsistency Between Sample Creation and Network Input**
**Location**: `src/module/effcat_module.py:194-270`
**Severity**: CRITICAL - Breaks DNG mode entirely

**Problem**:

When `dng=True`, the code creates TWO different masks that should be identical but aren't:

```python
# Line 195-197: Create mask #1 based on sampled atom counts
prim_slab_atom_mask = torch.zeros((batch_size * multiplicity_flow_sample, max_n), dtype=torch.bool, device=self.device)
for i, n in enumerate(sampled_n_atoms):  # Each sample has DIFFERENT n
    prim_slab_atom_mask[i, :n] = True  # ✅ Correct: varies per sample

# Line 216-223: Adjust mask #2 from original data
if max_n > original_n:
    padding = torch.zeros((batch_size, max_n - original_n), ...)
    feats["prim_slab_atom_pad_mask"] = torch.cat([feats["prim_slab_atom_pad_mask"], padding], dim=1)
    # ❌ BUG: Uses ORIGINAL data mask, not sampled mask!

# Line 265-266: Expand mask #2
feats["prim_slab_atom_pad_mask"] = feats["prim_slab_atom_pad_mask"].repeat_interleave(multiplicity_flow_sample, 0)
# ❌ BUG: All samples get SAME mask pattern (from original data)

# Line 292: Pass mask #1 to sampler
self.structure_module.sample(
    prim_slab_atom_mask=prim_slab_atom_mask,  # Uses sampled mask
    ...
)

# But inside the network (via feats), it uses mask #2!
# The network sees different masks than the sampler!
```

**Why This is Critical**:

1. **Mask #1** (prim_slab_atom_mask): Each of the `batch_size * multiplicity` samples has a different number of atoms (sampled from histogram)
   - Sample 0: 10 atoms (mask = [1,1,1,1,1,1,1,1,1,1,0,0,...])
   - Sample 1: 15 atoms (mask = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,...])
   - Sample 2: 8 atoms (mask = [1,1,1,1,1,1,1,1,0,0,...])

2. **Mask #2** (feats["prim_slab_atom_pad_mask"]): All samples have the SAME pattern from the original data
   - All samples: [1,1,1,1,1,1,1,1,1,1,1,1,0,0,...] (original data had 12 atoms)

3. The sampler uses Mask #1 to initialize coordinates, but the network uses Mask #2 to compute attention and loss!

**Impact**:
- Network computes loss on wrong atoms
- Attention masks are incorrect
- Padding positions are treated as valid
- DNG mode completely broken

**Fix**:
Replace lines 216-270 with:
```python
# In DNG mode, REPLACE feats mask with dynamically created mask
feats["prim_slab_atom_pad_mask"] = prim_slab_atom_mask
feats["prim_slab_token_pad_mask"] = prim_slab_atom_mask  # Assuming 1-to-1 mapping

# Don't adjust or expand separately - use the already-expanded sampled mask
```

---

### 2. **Supercell Matrix Prior Ignores Learned Distribution**
**Location**: `src/data/prior.py:127-133`
**Severity**: HIGH - Significantly degrades performance

**Problem**:

The config provides carefully computed statistics from the training set:
```yaml
supercell_mean: [[-0.026726, 0.078970, -0.065390], ...]
supercell_std: [[1.796400, 1.213596, 0.877900], ...]
```

But the code ignores them and uses a naive prior:
```python
# ❌ Commented out: learned distribution
# supercell_matrix_0_normalized = torch.randn(batch_size, 3, 3, ...)
# supercell_matrix_0 = self.denormalize_supercell(supercell_matrix_0_normalized)

# ✅ Actually used: identity + tiny noise
identity = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
noise = torch.randn(batch_size, 3, 3, device=device, dtype=dtype) * 0.1
supercell_matrix_0 = identity + noise  # Prior is always near-identity!
```

**Why This is a Problem**:

In flow matching, the prior p_0 should match the data distribution as closely as possible. Using identity + 0.1*noise means:
- Prior is always near [[1,0,0], [0,1,0], [0,0,1]]
- But real supercell matrices have mean ≈ [[0, 0.08, -0.07], [-0.13, 0.19, -0.13], [-0.01, -0.25, 0.14]]
- The flow has to learn to transform from identity to very different structures
- Makes optimization much harder

**Impact**:
- Poor supercell matrix predictions
- Harder optimization (need larger gradients)
- Slower convergence

**Fix**:
Uncomment lines 127-129 to use the learned distribution:
```python
supercell_matrix_0_normalized = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
supercell_matrix_0 = self.denormalize_supercell(supercell_matrix_0_normalized)
```

---

### 3. **Discrete Flow Matching: Potential Index Mismatch in Element Prediction**
**Location**: `src/models/layers.py:281` and `src/module/flow.py:730`
**Severity**: MEDIUM-HIGH - Subtle bug in DNG mode

**Problem**:

The decoder outputs NUM_ELEMENTS (100) logits for element prediction:
```python
# src/models/layers.py:279-282
self.feats_to_prim_slab_element = nn.Sequential(
    nn.LayerNorm(atom_s),
    LinearNoBias(atom_s, NUM_ELEMENTS)  # 100 classes (indices 0-99)
)
```

During sampling, we sample from these logits and add 1:
```python
# src/module/flow.py:729-730
x_1_probs = F.softmax(pred_element_logits, dim=-1)  # (B, N, 100)
x_1 = torch.distributions.Categorical(x_1_probs).sample() + 1  # Add 1 → 1-100
```

This means:
- Logit index 0 → Element 1 (Hydrogen)
- Logit index 1 → Element 2 (Helium)
- ...
- Logit index 99 → Element 100

But what about element 0 (padding)? The network can't predict it!

**Why This Might Be OK**:

Looking at the loss (line 556-559):
```python
target_indices = (true_element - 1).long()  # Convert 1-100 to 0-99
# Padding (true_element=0) becomes -1, which is ignored by ignore_index=-1
```

So padding is handled correctly in the loss. The network never needs to predict padding because it's masked out.

**However**, during sampling (line 730), after sampling and adding 1, we get elements 1-100. But if the network predicts index 0 with high probability, we'd get element 1, even if we wanted padding!

**Impact**:
- Probably minor if masks are correct
- But could cause issues at boundaries

**Fix**:
Consider outputting NUM_ELEMENTS_WITH_MASK (101) logits and have index 0 represent padding, then use the mask to force padding positions to index 0.

---

## 🟡 Moderate Issues

### 4. **Potential Numerical Instability in ODE Solver Near t=1**
**Location**: `src/module/flow.py:711-715`
**Severity**: LOW-MEDIUM

**Problem**:

The vector field computation divides by (1 - t):
```python
flow_prim_slab_coords = (pred_prim_slab_coords_1 - prim_slab_coords_t) / (1 - t + 1e-5)
```

When t approaches 1, (1 - t) approaches 0, making the flow very large. The epsilon (1e-5) helps, but:
- At t=0.9999, (1-t) = 0.0001, so flow is amplified 10,000×!
- This can cause gradient explosions
- Last step has dt = 1/num_steps ≈ 0.02 for 50 steps, so the last update is ≈ 200× the difference!

**Impact**:
- Training instability near t=1
- Gradient explosions
- Poor final predictions

**Fix**:
1. Use larger epsilon (1e-4 or 1e-3)
2. Or clip the flow magnitude
3. Or use a better ODE solver (RK4, adaptive step size)

---

### 5. **Lattice Angle Normalization: Degrees vs Radians Confusion**
**Location**: `src/module/flow.py:277-280, 304-306`
**Severity**: LOW - But could cause confusion

**Problem**:

The code converts lattice angles to radians for the network:
```python
# Input processing
l_noisy[:, 3:] = (torch.pi / 180.0) * noised_lattice[:, 3:]  # degrees → radians

# Output processing
denoised_lattice[:, 3:] = (180.0 / torch.pi) * denoised_lattice[:, 3:]  # radians → degrees
```

This is fine, but the network operates on values in range [π/3, 2π/3] (60° to 120°) which is roughly [1.05, 2.09]. This is a small range and might not use the network's capacity well.

**Impact**: Minor - angles might be harder to predict than necessary

**Suggestion**: Consider normalizing angles to [0, 1] range or using sin/cos representation

---

## ✅ Things That Are Correct

Despite the bugs, several aspects are implemented correctly:

1. **Flow Matching Formulation**: The network correctly predicts x_1, and the vector field is correctly computed as (x_1 - x_t) / (1 - t)

2. **ODE Integration**: Euler's method is correctly implemented

3. **Loss Computation**: The loss correctly compares predicted x_1 to true x_1

4. **Normalization Pipeline**: Coordinates are correctly normalized/denormalized (except supercell prior issue)

5. **Element Prediction Loss**: Cross-entropy is correctly computed with proper ignore_index for padding

6. **Mask Handling in Non-DNG Mode**: Works correctly when dng=False

---

## 🔧 Summary of Fixes

### Immediate Fixes (Critical)

1. **Fix DNG mask inconsistency** (effcat_module.py:216-270):
```python
# Replace the mask adjustment code with:
feats["prim_slab_atom_pad_mask"] = prim_slab_atom_mask
feats["prim_slab_token_pad_mask"] = prim_slab_atom_mask
# Remove lines 216-270 (don't adjust/expand separately)
```

2. **Use learned supercell prior** (prior.py:127-133):
```python
# Uncomment:
supercell_matrix_0_normalized = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
supercell_matrix_0 = self.denormalize_supercell(supercell_matrix_0_normalized)
# Remove lines 131-133
```

### Recommended Fixes (Moderate)

3. **Increase numerical stability epsilon** (flow.py:711-715):
```python
# Change 1e-5 to 1e-3:
flow_prim_slab_coords = (pred_prim_slab_coords_1 - prim_slab_coords_t) / (1 - t + 1e-3)
```

4. **Add gradient clipping** (already in config, but reduce from 10.0 to 1.0):
```yaml
gradient_clip_val: 1.0  # Reduce from 10.0
```

---

## 🧪 Testing the Fixes

After applying fixes, monitor these metrics:

1. **DNG mode validation** (if using dng=True):
   - Check that `val/prim_structural_validity_rate` is non-zero
   - Check that different samples have different atom counts
   - Log `batch["prim_slab_atom_pad_mask"].sum(dim=1)` to verify varying sizes

2. **Supercell matrix**:
   - Log `train/supercell_matrix_det_invalid_ratio` - should decrease
   - Check that predicted matrices aren't all near-identity

3. **Training stability**:
   - Monitor gradient norms - should be < 10 and not spike near end of training
   - Check that losses don't have NaN or sudden jumps

---

## 📝 Additional Investigation

If issues persist after fixes:

1. **Verify data preprocessing**:
   - Check that supercell statistics were computed correctly
   - Verify adsorbate coordinates are in correct reference frame
   - Check for outliers/corrupted samples

2. **Check histogram file** (DNG mode):
   - Verify `primitive_atom_distribution.json` exists and is valid
   - Check that probabilities sum to 1.0
   - Ensure histogram covers expected atom count range

3. **Architecture ablation**:
   - Try smaller model (atom_s=256, depths reduced)
   - Test with dng=False first to isolate DNG bugs
   - Verify attention masks are computed correctly

---

**Generated**: 2026-01-06
**Analysis Type**: Flow Matching Implementation Review
