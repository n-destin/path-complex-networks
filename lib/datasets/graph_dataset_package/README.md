# Structured Graph Dataset Generator

This package refactors your single-file cospectral loader into a small multi-file generator/loader pipeline for controlled graph families.

## Files

- `lib/utils/generators/config.py`
  - dataclasses for dataset/family/split config
- `lib/utils/generators/families.py`
  - graph family builders
- `lib/utils/generators/features.py`
  - node feature generation regimes
- `lib/utils/generators/splits.py`
  - IID, size extrapolation, distribution-shift splits
- `lib/utils/generators/io.py`
  - save/load generated datasets under `raw/<name>/`
- `lib/utils/generators/build_dataset.py`
  - top-level dataset construction
- `lib/data/datasets/structured_graph_dataset.py`
  - `InMemoryComplexDataset` wrapper that converts saved graphs to path complexes

## Example usage

```python
from lib.utils.generators import DatasetConfig, FamilySpec, SplitConfig, build_and_save_dataset

cfg = DatasetConfig(
    name='triangles_vs_holes',
    task_type='graph',
    feature_regime='ones',
    seed=7,
    families=[
        FamilySpec(name='triangle_rich', num_graphs=200, n_min=20, n_max=30, label=0, params={'k': 6, 'rewire_p': 0.03}),
        FamilySpec(name='hole_rich', num_graphs=200, n_min=20, n_max=30, label=1, params={'num_cycles': 1, 'min_cycle_len': 8}),
    ],
    split=SplitConfig(scheme='iid', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1),
)

build_and_save_dataset(raw_root='/path/to/dataset_root/raw', cfg=cfg)
```

For size extrapolation:

```python
cfg.split = SplitConfig(
    scheme='size_extrapolation',
    train_max_nodes=30,
    val_max_nodes=40,
)
```

Then load from your training pipeline using:

```python
from lib.data.datasets import StructuredGraphDataset

dataset = StructuredGraphDataset(root='/path/to/dataset_root', name='triangles_vs_holes', max_dim=3)
print(len(dataset.train_ids), len(dataset.val_ids), len(dataset.test_ids))
```

## Notes

- `task_type='graph'`: scalar graph label
- `task_type='node'`: node labels (e.g. clique membership / cycle membership)
- `task_type='edge'`: edge labels (e.g. cycle-edge membership)
- feature regimes currently included: `ones`, `degree`, `degree_triangle`, `random`
- families currently included: `er`, `triangle_rich`, `planted_clique`, `hole_rich`

## Important path fix

When calling `build_and_save_dataset`, pass the dataset root's **raw directory parent** as `raw_root`, e.g.
`/path/to/dataset_root/raw`.
The resulting files will be saved to `/path/to/dataset_root/raw/<dataset_name>/...`.
