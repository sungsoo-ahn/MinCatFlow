# DNG Mask Bug: Detailed Explanation

## The Bug in One Sentence

**The code creates two different masks for the same atoms: one based on dynamically sampled atom counts, and another from the original data batch - then the network uses the wrong one.**

---

## Step-by-Step Walkthrough

Let me trace through what happens during validation with `dng=True`:

### Setup
```python
# Example validation batch
batch_size = 2
multiplicity_flow_sample = 3  # Generate 3 samples per input
original_atoms = 12  # Original data has 12 atoms per structure

# Original mask from data batch
batch["prim_slab_atom_pad_mask"] = torch.ones(2, 12)  # All 12 atoms valid
```

---

### Step 1: Sample Atom Counts from Histogram
**Location**: `src/module/effcat_module.py:184-189`

```python
sampled_n_atoms = np.random.choice(
    n_atoms_indices,
    size=batch_size * multiplicity_flow_sample,  # 2 * 3 = 6 samples
    p=self.n_prim_slab_atoms_hist,
    replace=True
).tolist()

# Example result:
# sampled_n_atoms = [10, 15, 8, 12, 11, 9]
```

Each of the 6 samples gets a **different** number of atoms:
- Sample 0 (from original 0): 10 atoms
- Sample 1 (from original 0): 15 atoms
- Sample 2 (from original 0): 8 atoms
- Sample 3 (from original 1): 12 atoms
- Sample 4 (from original 1): 11 atoms
- Sample 5 (from original 1): 9 atoms

---

### Step 2: Create Dynamic Mask #1
**Location**: `src/module/effcat_module.py:192-197`

```python
max_n = max(sampled_n_atoms)  # max_n = 15

# Create prim_slab_atom_mask based on sampled counts
prim_slab_atom_mask = torch.zeros((6, 15), dtype=torch.bool)
for i, n in enumerate(sampled_n_atoms):
    prim_slab_atom_mask[i, :n] = True
```

**Result** - `prim_slab_atom_mask` shape `(6, 15)`:
```
Sample 0: [T T T T T T T T T T F F F F F]  ← 10 atoms
Sample 1: [T T T T T T T T T T T T T T T]  ← 15 atoms
Sample 2: [T T T T T T T T F F F F F F F]  ← 8 atoms
Sample 3: [T T T T T T T T T T T T F F F]  ← 12 atoms
Sample 4: [T T T T T T T T T T T F F F F]  ← 11 atoms
Sample 5: [T T T T T T T T T F F F F F F]  ← 9 atoms
```

✅ **This mask is CORRECT** - each sample has different atom count!

---

### Step 3: Adjust Original Data Mask #2
**Location**: `src/module/effcat_module.py:216-223`

```python
# Adjust feats["prim_slab_atom_pad_mask"] to match max_n
# Original shape: (2, 12)
if max_n > original_n:  # 15 > 12
    padding = torch.zeros((batch_size, max_n - original_n), ...)  # (2, 3)
    feats["prim_slab_atom_pad_mask"] = torch.cat([
        feats["prim_slab_atom_pad_mask"],  # (2, 12) - all True
        padding  # (2, 3) - all False
    ], dim=1)
    # Result: (2, 15)
```

**Result** - `feats["prim_slab_atom_pad_mask"]` shape `(2, 15)`:
```
Original 0: [T T T T T T T T T T T T F F F]  ← 12 atoms (from data)
Original 1: [T T T T T T T T T T T T F F F]  ← 12 atoms (from data)
```

---

### Step 4: Expand Original Mask for Multiplicity
**Location**: `src/module/effcat_module.py:265-266`

```python
# Expand to match multiplicity
feats["prim_slab_atom_pad_mask"] = feats["prim_slab_atom_pad_mask"].repeat_interleave(
    multiplicity_flow_sample, 0
)  # (2, 15) → (6, 15)
```

**Result** - `feats["prim_slab_atom_pad_mask"]` shape `(6, 15)`:
```
Sample 0: [T T T T T T T T T T T T F F F]  ← From original 0: 12 atoms
Sample 1: [T T T T T T T T T T T T F F F]  ← From original 0: 12 atoms
Sample 2: [T T T T T T T T T T T T F F F]  ← From original 0: 12 atoms
Sample 3: [T T T T T T T T T T T T F F F]  ← From original 1: 12 atoms
Sample 4: [T T T T T T T T T T T T F F F]  ← From original 1: 12 atoms
Sample 5: [T T T T T T T T T T T T F F F]  ← From original 1: 12 atoms
```

❌ **This mask is WRONG!**
- All samples from original 0 get 12 atoms (but should be 10, 15, 8)
- All samples from original 1 get 12 atoms (but should be 12, 11, 9)

---

### Step 5: Compare the Two Masks

**Mask #1** (`prim_slab_atom_mask`) - CORRECT:
```
Sample 0: [T T T T T T T T T T F F F F F]  ← 10 atoms ✓
Sample 1: [T T T T T T T T T T T T T T T]  ← 15 atoms ✓
Sample 2: [T T T T T T T T F F F F F F F]  ← 8 atoms ✓
Sample 3: [T T T T T T T T T T T T F F F]  ← 12 atoms ✓
Sample 4: [T T T T T T T T T T T F F F F]  ← 11 atoms ✓
Sample 5: [T T T T T T T T T F F F F F F]  ← 9 atoms ✓
```

**Mask #2** (`feats["prim_slab_atom_pad_mask"]`) - WRONG:
```
Sample 0: [T T T T T T T T T T T T F F F]  ← 12 atoms ✗ (should be 10)
Sample 1: [T T T T T T T T T T T T F F F]  ← 12 atoms ✗ (should be 15)
Sample 2: [T T T T T T T T T T T T F F F]  ← 12 atoms ✗ (should be 8)
Sample 3: [T T T T T T T T T T T T F F F]  ← 12 atoms ✓
Sample 4: [T T T T T T T T T T T T F F F]  ← 12 atoms ✗ (should be 11)
Sample 5: [T T T T T T T T T T T T F F F]  ← 12 atoms ✗ (should be 9)
```

---

## Where Each Mask is Used

### Mask #1 (`prim_slab_atom_mask`) - Used For:

1. **Prior sampling** (`flow.py:649-656`):
```python
sampler_data = {
    "prim_slab_atom_mask": prim_slab_atom_mask,  # ✓ Correct mask
    ...
}
priors = self.prior_sampler.sample(sampler_data)
```

2. **Zeroing padding** during ODE steps (`flow.py:745-746`):
```python
prim_slab_coords_t = prim_slab_coords_t * prim_slab_atom_mask.unsqueeze(-1)  # ✓ Correct
```

### Mask #2 (`feats["prim_slab_atom_pad_mask"]`) - Used For:

1. **Encoder attention mask** (`layers.py:107-187`):
```python
prim_slab_mask = feats["prim_slab_atom_pad_mask"].bool()  # ✗ Wrong mask!
joint_mask = torch.cat([prim_slab_mask, ads_mask], dim=1)
q = self.atom_encoder(q, t, joint_mask)  # Attends to wrong atoms!
```

2. **Decoder global pooling** (`layers.py:322-323`):
```python
num_prim_slab_atoms = prim_slab_mask.sum(dim=1)  # ✗ Wrong count!
x_global = torch.sum(x_prim_slab * prim_slab_mask[..., None], dim=1) / num_prim_slab_atoms
# Averages over 12 atoms when should be 10, 15, 8, etc.
```

3. **Validation metrics** (`effcat_module.py:442`):
```python
return_dict = {
    "prim_slab_atom_mask": batch["prim_slab_atom_pad_mask"],  # ✗ Uses original!
    ...
}
# Metrics computed on wrong atom positions!
```

---

## Concrete Impact

Let's trace Sample 0 (should have 10 atoms, but mask says 12):

### During Network Forward:

**What should happen:**
- Network attends to atoms 0-9 (10 atoms)
- Global pooling averages over atoms 0-9
- Atoms 10-14 are ignored

**What actually happens:**
- Network attends to atoms 0-11 (12 atoms) ← includes 2 padding atoms!
- Global pooling averages over atoms 0-11 ← diluted by 2 zeros
- Atoms 10-11 are treated as valid but have random/zero values

### During Loss Computation:

The network predicts coordinates for 15 positions (max_n), but:
- Positions 0-9 should be valid (10 atoms)
- **Positions 10-11 are incorrectly treated as valid** (from wrong mask)
- Positions 12-14 are correctly treated as padding

So the loss is computed over 12 atoms instead of 10, including 2 bogus positions!

---

## Why This Breaks DNG Mode

The whole point of DNG (Dynamic Number Generation) is to generate structures with **varying atom counts**. But:

1. **Network never sees varying counts** - it always gets the same mask pattern (12 atoms) from the original data
2. **Attention is wrong** - attends to padding positions that should be masked
3. **Pooling is wrong** - averages over wrong number of atoms
4. **Loss is wrong** - trained on incorrect atom positions
5. **Validation is wrong** - metrics compare wrong atoms

In effect, **DNG mode doesn't work** because the model never actually learns to handle varying atom counts. All samples from the same original structure get treated identically!

---

## The Fix

**Location**: `src/module/effcat_module.py:265-270`

Replace:
```python
# Lines 216-270: Don't adjust and expand separately
if "prim_slab_atom_pad_mask" in feats:
    feats["prim_slab_atom_pad_mask"] = feats["prim_slab_atom_pad_mask"].repeat_interleave(multiplicity_flow_sample, 0)
```

With:
```python
# Simply use the dynamically created mask!
feats["prim_slab_atom_pad_mask"] = prim_slab_atom_mask  # Already has shape (6, 15)
feats["prim_slab_token_pad_mask"] = prim_slab_atom_mask  # Same for tokens
```

This ensures the network uses the **correct mask** that matches the sampled atom counts.

---

## Verification Test

After the fix, add this assertion:
```python
# After line 270
assert torch.equal(feats["prim_slab_atom_pad_mask"], prim_slab_atom_mask), \
    "Masks must be identical!"

# Also verify varying counts
atom_counts = prim_slab_atom_mask.sum(dim=1)
assert atom_counts.unique().numel() > 1, \
    f"Should have varying atom counts, got all {atom_counts[0]}"
```

And log during validation:
```python
self.log("debug/min_atoms", prim_slab_atom_mask.sum(dim=1).float().min())
self.log("debug/max_atoms", prim_slab_atom_mask.sum(dim=1).float().max())
self.log("debug/mean_atoms", prim_slab_atom_mask.sum(dim=1).float().mean())
```

If working correctly, you should see:
- `min_atoms` < `max_atoms` (varying counts)
- Different values across samples

If broken, you'd see:
- `min_atoms` == `max_atoms` == 12 (all same)

---

## Why This Bug is So Insidious

1. **Code runs without errors** - shapes match, no crashes
2. **Losses decrease** - model still learns something, just wrong
3. **Metrics show "results"** - but they're measuring the wrong things
4. **Only affects DNG mode** - works fine when `dng=False`
5. **Hard to notice** - need to carefully track which mask is used where

The bug essentially makes DNG mode fall back to fixed-size prediction, defeating its entire purpose!

---

**Generated**: 2026-01-06
