# MinCatFlow: Flow Matching for Catalyst Structure Generation

## Overview

MinCatFlow is a machine learning framework for generating catalyst structures using flow matching (continuous normalizing flows). The model generates complete catalyst systems including:
- **Primitive slab structures** (bulk crystal structures)
- **Adsorbate positions** (molecules adsorbed on the surface)
- **Lattice parameters** (unit cell dimensions and angles)
- **Supercell matrices** (periodic replication)
- **Scaling factors** (size adjustments)

The framework is built on PyTorch Lightning and uses transformer-based architecture with DiT (Diffusion Transformer) blocks.

## Key Features

### 1. **Flow Matching Architecture**
- Continuous normalizing flow-based generative model
- Jointly generates coordinates, lattice parameters, and supercell configurations
- Supports both fixed and dynamic number of atoms (DNG mode)

### 2. **Dynamic Number Generation (DNG)**
- **Mode**: `dng=True` enables variable atom count generation
- Samples number of atoms from a learned histogram distribution
- Uses discrete flow matching with masking for atom types
- Allows generating structures with different sizes than training data

### 3. **Transformer-Based Architecture**
Three main components:
- **AtomAttentionEncoder**: Encodes atom-level features with attention
- **TokenTransformer**: Processes token-level representations with DiT blocks
- **AtomAttentionDecoder**: Decodes to coordinates, lattice, and supercell predictions

### 4. **Validation Metrics**
- **RMSD matching** (when `dng=False`): Structural similarity to ground truth
- **Structural validity**: Checks for valid crystal structures
- **Adsorption energy** (optional): Energy calculations using UMA calculator
- **Match rates**: Percentage of generated structures matching references

## Project Structure

```
MinCatFlow/
├── configs/               # Hydra configuration files
│   ├── default.yaml      # Main configuration
│   ├── data/             # Dataset configurations
│   ├── model/            # Model architecture configs
│   ├── optim/            # Optimizer settings
│   ├── train/            # Training configurations
│   └── logging/          # Logging and W&B settings
├── src/
│   ├── data/
│   │   ├── datamodule.py      # Lightning DataModule with LMDB support
│   │   ├── lmdb_dataset.py    # LMDB dataset implementation
│   │   ├── prior.py           # Prior samplers for flow matching
│   │   └── pad.py             # Dynamic padding utilities
│   ├── models/
│   │   ├── transformers.py    # DiT blocks and transformer layers
│   │   ├── layers.py          # Attention encoders/decoders
│   │   ├── utils.py           # Model utilities
│   │   └── loss/              # Loss functions and validation
│   ├── module/
│   │   ├── effcat_module.py   # Main Lightning module
│   │   └── flow.py            # Flow matching implementation
│   ├── run.py                 # Training script entry point
│   └── utils.py               # General utilities
├── scripts/
│   ├── assemble.py            # Structure assembly utilities
│   ├── eval.py                # Evaluation scripts
│   ├── refine_sc_mat.py       # Supercell matrix refinement
│   └── primitive_atom_distribution.py  # Atom count analysis
├── bash_scripts/
│   ├── train.sh               # Training launch script
│   ├── sample.sh              # Sampling script
│   └── relaxation.sh          # Structure relaxation
├── environment.yml            # Conda environment specification
├── save_valid_samples.py      # Sample validation and saving
├── validate_relaxation.py     # Relaxation validation
├── validate_step.py           # Step-by-step validation
└── val_relax_gen.py          # Validation with relaxation

```

## Installation

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate mincatflow
```

### 2. Key Dependencies
- Python 3.10
- PyTorch 2.8.0+ (with CUDA 12.6 support)
- PyTorch Lightning 2.6.0
- PyTorch Geometric 2.7.0
- Hydra 1.3.2
- Weights & Biases (wandb) for logging
- ASE (Atomic Simulation Environment)
- Pymatgen for crystal structure analysis
- fairchem-core for energy calculations

## Usage

### Training

#### Basic Training
```bash
python src/run.py
```

#### Training with Custom Config
```bash
python src/run.py \
    train.pl_trainer.max_epochs=100 \
    data.datamodule.batch_size.train=32 \
    model.flow_model_args.dng=True
```

#### Using Bash Script
```bash
bash bash_scripts/train.sh
```

### Configuration

The project uses Hydra for configuration management. Main config file: `configs/default.yaml`

#### Key Configuration Options

**Model Configuration** (`configs/model/default.yaml`):
```yaml
flow_model_args:
  atom_s: 256                    # Atom feature dimension
  token_s: 512                   # Token feature dimension
  atom_encoder_depth: 4          # Encoder transformer depth
  token_transformer_depth: 8     # Token transformer depth
  atom_decoder_depth: 4          # Decoder depth
  dng: false                     # Enable dynamic number generation
```

**Training Configuration** (`configs/train/default.yaml`):
```yaml
training_args:
  lr: 0.0001                     # Learning rate
  train_multiplicity: 1          # Samples per training input
  loss_type: "l2"                # Loss function (l1 or l2)
  prim_slab_coord_loss_weight: 1.0
  ads_coord_loss_weight: 1.0
  length_loss_weight: 1.0
  angle_loss_weight: 1.0
  supercell_matrix_loss_weight: 1.0
  scaling_factor_loss_weight: 1.0
```

**Data Configuration** (`configs/data/default.yaml`):
```yaml
datamodule:
  _target_: src.data.datamodule.LMDBDataModule
  train_lmdb_path: "path/to/train.lmdb"
  val_lmdb_path: "path/to/val.lmdb"
  batch_size:
    train: 32
    val: 16
  preload_to_ram: true
```

### Sampling/Generation

Generate structures from a trained model:

```python
# In prediction mode
trainer.predict(model=model, datamodule=datamodule)
```

Sample validation script:
```bash
python save_valid_samples.py \
    --checkpoint path/to/checkpoint.ckpt \
    --output_dir generated_samples/ \
    --num_samples 100
```

### Evaluation

Validate generated structures:
```bash
python validate_relaxation.py \
    --samples_dir generated_samples/ \
    --reference_dir reference_structures/
```

## Architecture Details

### Flow Matching Process

1. **Prior Sampling (t=0)**:
   - Coordinates sampled from Gaussian distributions
   - Lattice parameters from learned priors
   - Supercell matrices from identity or learned distributions
   - Element types start as MASK tokens (DNG mode)

2. **Flow ODE Integration**:
   - Euler method with configurable steps (default: 16)
   - Interpolates between prior (t=0) and target (t=1)
   - Network predicts x_1 (final state) at each timestep
   - Flow computed as: `flow = (pred_x_1 - x_t) / (1 - t)`

3. **Discrete Flow Matching (DNG mode)**:
   - Element types use masked token approach
   - Rate-based unmasking: `unmask_rate = dt / (1 - t)`
   - Categorical sampling from predicted logits
   - Cross-entropy loss for element prediction

### Model Components

#### AtomAttentionEncoder (src/models/layers.py)
- Input embedding of coordinates, lattice, time, and elements
- Multi-head self-attention over atoms
- Separate processing for primitive slab and adsorbate atoms
- Positional encoding options: sinusoidal, RoPE, or learned

#### TokenTransformer (src/models/transformers.py)
- DiT (Diffusion Transformer) blocks with adaLN-Zero conditioning
- Time-conditioned modulation of attention and MLP layers
- Processes aggregated token-level representations
- Optional activation checkpointing for memory efficiency

#### AtomAttentionDecoder (src/models/layers.py)
- Separate output heads for:
  - Primitive slab coordinates (3D)
  - Adsorbate coordinates (3D)
  - Lattice parameters (6D: a, b, c, α, β, γ)
  - Supercell matrix (3×3)
  - Scaling factor (scalar)
  - Element logits (DNG mode, NUM_ELEMENTS classes)

### Loss Functions

**Coordinate Losses** (L1 or L2):
- Primitive slab coordinate loss (masked by valid atoms)
- Adsorbate coordinate loss (masked by valid atoms)

**Lattice Losses**:
- Length loss (a, b, c in Angstroms)
- Angle loss (α, β, γ in degrees)

**Supercell & Scaling Losses**:
- Supercell matrix loss (9 elements)
- Scaling factor loss (scalar)

**Element Loss (DNG mode)**:
- Cross-entropy loss for element classification
- Ignores padding positions with `ignore_index=-1`

### Validation Workflow

The validation process (`on_validation_epoch_end` in src/module/effcat_module.py:455):

1. **Sample Generation**: Generate multiple structures per input (configurable via `flow_samples`)

2. **Parallel Processing**:
   - Uses `ProcessPool` for parallel structure analysis
   - Configurable timeout and worker count
   - Progress tracking with tqdm

3. **Metrics Computed**:
   - **Prim RMSD** (DNG=False): Match primitive slab to reference
   - **Slab RMSD** (DNG=False): Match full assembled structure
   - **Prim Structural Validity**: Check primitive cell validity
   - **Slab Structural Validity**: Check full structure validity
   - **Adsorption Energy** (optional): UMA calculator predictions

4. **DDP Considerations**:
   - Validation runs only on rank 0
   - Other ranks wait at synchronization barrier
   - Adsorption computation disabled by default in DDP to avoid timeout

## Data Format

### LMDB Dataset Structure

The project uses LMDB (Lightning Memory-Mapped Database) for efficient data loading:

```python
# Each sample contains:
{
    'prim_slab_cart_coords': (N, 3),      # Primitive slab coordinates
    'ads_cart_coords': (M, 3),             # Adsorbate coordinates
    'lattice': (6,),                       # [a, b, c, α, β, γ]
    'supercell_matrix': (3, 3),           # Supercell transformation
    'scaling_factor': float,               # Scaling factor
    'ref_prim_slab_element': (N,),        # Element types (1-indexed)
    'ref_ads_element': (M,),              # Adsorbate element types
    'prim_slab_atom_pad_mask': (N,),      # Valid atom mask
    'ads_atom_pad_mask': (M,),            # Valid adsorbate mask
    'prim_slab_atom_to_token': (N, N),    # Atom-to-token mapping
    'ads_atom_to_token': (M, M),          # Adsorbate atom-to-token
}
```

### Dynamic Padding

The datamodule (src/data/datamodule.py) implements dynamic padding:
- Pads sequences to maximum length in each batch
- Reduces memory usage compared to global max padding
- Maintains mask tensors for valid positions

## Training Tips

### 1. **Memory Optimization**
- Use activation checkpointing for large models: `activation_checkpointing=True`
- Enable mixed precision: `train.pl_trainer.precision=16-mixed`
- Adjust batch size with `max_batch_units` for variable-sized data
- Preload small datasets to RAM: `preload_to_ram=True`

### 2. **Deterministic Training**
- Set `train.deterministic=True` for reproducibility
- Fixed random seed: `train.random_seed=42`
- Disables cudnn benchmarking for determinism

### 3. **Validation Configuration**
- Adjust `sample_every_n_epochs` to reduce validation overhead
- Disable adsorption in DDP: `compute_adsorption=False`
- Set appropriate timeout for structure matching: `timeout=300`
- Configure parallel workers: `num_workers=8`

### 4. **DNG Mode Training**
- Requires histogram file: `n_prim_slab_atoms_histogram_path`
- Enable element loss: `prim_slab_element_loss_weight=1.0`
- Set `dng=True` in `flow_model_args`
- RMSD metrics disabled automatically when DNG=True

### 5. **Checkpoint Management**
- Saves every epoch: `SaveEveryEpochCheckpoint` callback
- Monitors validation loss: `monitor_metric=val/total_loss`
- Automatic resume from last checkpoint in output directory

## Logging and Monitoring

### Weights & Biases Integration
```yaml
logging:
  wandb:
    project: "mincatflow"
    name: "experiment_name"
    mode: "online"  # or "offline" for local logging
```

### Logged Metrics

**Training Metrics**:
- Loss components (per type)
- Stratified losses by timestep
- Batch size and samples/second
- Learning rate (with LearningRateMonitor)

**Validation Metrics**:
- Match rates (prim/slab)
- Average RMSD values
- Structural validity rates
- Adsorption energies (if enabled)
- Supercell matrix determinant checks

## Advanced Features

### 1. **Supercell Matrix Refinement**
Located in `scripts/refine_sc_mat.py`:
- Rounds predicted supercell matrices to nearest integers
- Ensures valid crystallographic transformations
- Can be applied post-generation

### 2. **Structure Assembly**
Located in `scripts/assemble.py`:
- Combines primitive slab + adsorbate
- Applies supercell transformations
- Scales structures by scaling factor
- Outputs ASE Atoms objects or structure files

### 3. **Relaxation Validation**
Located in `validate_relaxation.py`:
- Performs DFT relaxation on generated structures
- Compares relaxed vs. unrelaxed energies
- Validates structural stability

### 4. **Custom Prior Samplers**
Located in `src/data/prior.py`:
- `CatPriorSampler`: Catalyst-specific prior distributions
- Gaussian priors for coordinates
- Learned distributions for lattice parameters
- Customizable for different material systems

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size or `max_batch_units`
   - Enable activation checkpointing
   - Use gradient accumulation
   - Decrease model dimensions (`atom_s`, `token_s`)

2. **NCCL Timeout in DDP**
   - Disable adsorption computation: `compute_adsorption=False`
   - Reduce validation frequency: `sample_every_n_epochs=5`
   - Increase NCCL timeout: `export NCCL_TIMEOUT_MS=1800000`

3. **Slow Validation**
   - Reduce `flow_samples` (number of samples per input)
   - Increase `num_workers` for parallel processing
   - Decrease `sampling_steps` (may reduce quality)

4. **NaN Losses**
   - Check data preprocessing and normalization
   - Reduce learning rate
   - Enable gradient clipping: `train.pl_trainer.gradient_clip_val=1.0`
   - Check for invalid structures in dataset

## Code Entry Points

### Training
- **Main script**: `src/run.py:main()` (line 252)
- **Training loop**: `src/run.py:run()` (line 155)
- **Model definition**: `src/module/effcat_module.py:EffCatModule` (line 33)

### Model Forward Pass
- **Training forward**: `src/module/flow.py:AtomFlowMatching.forward()` (line 320)
- **Sampling**: `src/module/flow.py:AtomFlowMatching.sample()` (line 582)
- **Network call**: `src/module/flow.py:FlowModule.forward()` (line 100)

### Data Loading
- **DataModule**: `src/data/datamodule.py:LMDBDataModule` (line 194)
- **Dataset**: `src/data/lmdb_dataset.py:LMDBCachedDataset`
- **Collate function**: `src/data/lmdb_dataset.py:collate_fn_with_dynamic_padding`

## Citation

If you use MinCatFlow in your research, please cite the relevant papers:

```bibtex
# Add citation information here when available
```

## Contributing

When contributing to this project:
1. Follow the existing code style (uses Black formatter)
2. Add type hints to function signatures
3. Document new features in this file
4. Update configuration examples as needed
5. Test changes with `pytest` (if tests are added)

## License

[Add license information]

## Contact

[Add contact information or links to issues/discussions]

---

**Last Updated**: 2026-01-06
**Version**: Based on commit a661321
