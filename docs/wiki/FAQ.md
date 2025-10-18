# FAQ

**Q: Does it only detect fixed intro lengths?**  
No. It detects any segment length between `min_segment_length_sec` and `max_segment_length_sec`.

**Q: Can I process long files (like 1 h episodes)?**  
Possible but inefficient; crop to first ~90 s for better results.

**Q: Is GPU mandatory?**  
Yes. The correlator depends on CuPy.

**Q: How accurate is it?**  
Precision depends on `precision_secs`, input quality, and number of episodes.
