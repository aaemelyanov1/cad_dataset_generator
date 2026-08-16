from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from pathlib import Path

@dataclass
class GeneratorConfig:
    complexity_levels: List[str] = field(default_factory=lambda: ["easy", "medium", "hard", "expert"])
    easy_ops_range: Tuple[int, int] = (3, 6)
    medium_ops_range: Tuple[int, int] = (7, 12)
    hard_ops_range: Tuple[int, int] = (13, 20)
    expert_ops_range: Tuple[int, int] = (20, 35)
    max_holes: int = 5
    max_fillets: int = 10
    max_chamfers: int = 10
    max_booleans: int = 10
    max_patterns: int = 5
    max_shell: int = 1
    max_splits: int = 5
    max_twist_extrudes: int = 10
    max_depth: int = 14
    max_operations: int = 40
    binary_probability: float = 0.4
    min_volume: float = 0.01
    max_volume: float = 1e6
    min_bbox_diag: float = 0.1
    max_bbox_diag: float = 1000.0
    fillet_radius_ratio: float = 0.1
    shell_thickness_ratio: float = 0.02
    backtrack_attempts: int = 60
    max_seed_retries: int = 50
    sample_timeout: float = 150.0
    # таймаут зависшего сэмпла по сложности: лёгкие уровни живут ~attempt_timeout*4
    # (easy 32s, medium 60s, hard 80s, expert 120s), ждать 150s на easy бессмысленно
    sample_timeouts: Dict[str, float] = field(default_factory=lambda: {
        "easy": 40.0,
        "medium": 75.0,
        "hard": 100.0,
        "expert": 150.0,
    })
    # --- вероятности «попробовать» дорогих/рискованных модификаторов ---
    modifier_probabilities: Dict[str, float] = field(default_factory=lambda: {
        "shell": 0.3,
        "fillet": 0.6,
        "chamfer": 0.6,
        "rarray": 0.6,
        "polarArray": 0.6,
        "scatter": 0.6,
        "hole": 0.5,
        "split": 0.6,
    })
    # --- веса выбора унарных модификаторов (влияют на разнообразие) ---
    unary_choice_weights: Dict[str, float] = field(default_factory=lambda: {
        "fillet": 5.0,
        "chamfer": 5.0,
        "shell": 3.5,
        "hole": 5.0,
        "split": 3.5,
        "translate": 2.5,
        "rotate": 2.5,
        "mirror": 2.5,
        "rarray": 3.0,
        "polarArray": 3.0,
        "scatter": 3.0,
    })
    # --- веса выбора бинарных операций ---
    binary_choice_weights: Dict[str, float] = field(default_factory=lambda: {
        "union": 5.0,
        "cut": 5.0,
        "intersect": 2.0,
    })
    # --- веса выбора листовых (терминальных) операций ---
    leaf_choice_weights: Dict[str, float] = field(default_factory=lambda: {
        "box": 1.0,
        "cylinder": 1.0,
        "sphere": 1.0,
        "cone": 1.0,
        "wedge": 1.0,
        "torus": 1.0,
        "extrude": 1.2,
        "revolve": 1.2,
        "twist_extrude": 1.2,
        "loft": 2.5,
        "sweep": 2.5,
    })
    split_by_complexity: bool = True
    save_formats: List[str] = field(default_factory=lambda: ["program_py"])
    pointcloud_samples: int = 4096
    render_resolution: Tuple[int, int] = (512, 512)
    global_seed: int = 42
    output_dir: Path = Path("output")