from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional, Sequence, Tuple

TaskType = Literal['graph', 'node', 'edge']
FeatureRegime = Literal['ones', 'degree', 'degree_triangle', 'random']
SplitScheme = Literal['iid', 'size_extrapolation', 'distribution_shift']
FamilyName = Literal['er', 'triangle_rich', 'planted_clique', 'hole_rich']


@dataclass
class FamilySpec:
    name: FamilyName
    num_graphs: int
    n_min: int
    n_max: int
    params: Dict = field(default_factory=dict)
    label: Optional[int] = None


@dataclass
class SplitConfig:
    scheme: SplitScheme = 'iid'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Used for size_extrapolation
    train_max_nodes: Optional[int] = None
    val_max_nodes: Optional[int] = None

    # Used for distribution_shift
    train_families: Optional[Sequence[str]] = None
    val_families: Optional[Sequence[str]] = None
    test_families: Optional[Sequence[str]] = None


@dataclass
class DatasetConfig:
    name: str
    families: List[FamilySpec]
    task_type: TaskType = 'graph'
    feature_regime: FeatureRegime = 'ones'
    split: SplitConfig = field(default_factory=SplitConfig)
    seed: int = 0
    random_feature_dim: int = 8

    # Only used by family-specific graph generation logic when labels are not
    # fully determined by the family id.
    default_graph_label: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)
