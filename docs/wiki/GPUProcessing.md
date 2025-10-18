# GPU Processing

GPU acceleration is handled using **CuPy**, mirroring NumPy’s API.

---

### Core GPU Operations

| Operation | Module | Description |
|------------|---------|-------------|
| Normalization | `audio_samples.py` | Centering and scaling waveform |
| Correlation | `correlator/` | Async and sync moving window comparisons |
| Memory Pool | `cp.get_default_memory_pool()` | Recycles GPU memory blocks |

---

### Performance Notes

- GPU memory pool cleared after each comparison.
- Async correlation uses coarse steps; sync correlation refines within detected range.
- Avoid full-length files — use 30–90 s context per episode for best results.
