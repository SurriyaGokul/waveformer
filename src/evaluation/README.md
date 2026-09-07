# `src/evaluation/`

Training diagnostics — used to inspect a run *during or after* training, as opposed to
`scripts/evaluate_*.py`, which score a finished checkpoint end-to-end.

| File | Key functions | Use |
|---|---|---|
| `diagnose_training.py` | `diagnose_physics_gradients`, `diagnose_loss_balance`, `diagnose_data_distribution` | Inspect gradient magnitudes / loss-term balance / input data statistics for a given run |
| `visualize_issues.py` | `parse_training_log`, `create_analysis_figure`, `create_loss_component_analysis` | Parses a `outputs/<run>/train_*.log` file and plots per-loss-component curves to spot collapse/instability |
| `generate_visualizations.py` | `prepare_data_for_model`, `iq_to_displacement`, `plot_frame_comparison` | Generates frame-by-frame prediction-vs-GT comparison figures |

## Usage

These are library functions, typically called from a small driver script or notebook
rather than run directly, e.g.:

```python
from src.evaluation.visualize_issues import parse_training_log, create_analysis_figure

log_data = parse_training_log("outputs/wavepinn_v10b/train_wavepinn_v10b.log")
create_analysis_figure(log_data, out_path="outputs/wavepinn_v10b/loss_analysis.png")
```
