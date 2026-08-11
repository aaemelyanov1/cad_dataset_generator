from dataclasses import dataclass, field
from typing import List, Tuple
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
    save_formats: List[str] = field(default_factory=lambda: [
        "step", "stl", "pointcloud", "mesh", "program_py", "program_txt",
        "ast_json", "metadata_json", "render"
    ])
    pointcloud_samples: int = 4096
    render_resolution: Tuple[int, int] = (512, 512)
    global_seed: int = 42
    output_dir: Path = Path("output")