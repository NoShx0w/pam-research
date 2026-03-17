from dataclasses import dataclass
from typing import Dict, Tuple

GridKey = Tuple[float, float]

@dataclass
class CellValue:
    coverage: float = 0.0
    curvature: float = 0.0
    piF_tail: float = 0.0
    h_joint_mean: float = 0.0
    present: bool = False

@dataclass
class ObservatoryState:
    dataset_progress: int
    dataset_total: int
    selected_r: float
    selected_alpha: float
    color_mode: str = "coverage"
    embedding_mode: str = "r"
    probe_mode: str = "fan"
    r_values: list | None = None
    alpha_values: list | None = None
    cells: Dict[GridKey, CellValue] | None = None

    def get_cell(self,r,a):
        if not self.cells: return CellValue()
        return self.cells.get((r,a), CellValue())

    def cycle_color_mode(self):
        modes=["coverage","curvature","piF_tail","h_joint_mean"]
        self.color_mode = modes[(modes.index(self.color_mode)+1)%len(modes)]

    def cycle_embedding_mode(self):
        modes=["r","alpha","curvature"]
        self.embedding_mode = modes[(modes.index(self.embedding_mode)+1)%len(modes)]

    def cycle_probe_mode(self):
        modes=["fan","path"]
        self.probe_mode = modes[(modes.index(self.probe_mode)+1)%len(modes)]

    def selected_indices(self):
        r_idx=self.r_values.index(self.selected_r)
        a_idx=self.alpha_values.index(self.selected_alpha)
        return r_idx,a_idx

    def move_selection(self,dr=0,da=0):
        r,a=self.selected_indices()
        r=max(0,min(len(self.r_values)-1,r+dr))
        a=max(0,min(len(self.alpha_values)-1,a+da))
        self.selected_r=self.r_values[r]
        self.selected_alpha=self.alpha_values[a]