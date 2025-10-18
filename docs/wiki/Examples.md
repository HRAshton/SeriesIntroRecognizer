# Examples

## Using Audio Files

```python
from series_intro_recognizer.config import Config
from series_intro_recognizer.processors.audio_files import recognise_from_audio_files

cfg = Config()
files = ['ep1.wav', 'ep2.wav', 'ep3.wav']
intervals = recognise_from_audio_files(iter(files), cfg)
print(intervals)
```

---

## Using Raw NumPy Arrays

```python
import numpy as np
from series_intro_recognizer.config import Config
from series_intro_recognizer.processors.audio_samples import recognise_from_audio_samples

cfg = Config()
episodes = [np.random.randn(48000*60) for _ in range(3)]  # dummy 1-min samples
intervals = recognise_from_audio_samples(iter(episodes), cfg)
print(intervals)
```
