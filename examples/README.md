#Runnable demos for 1D/2D adaptive Kalman filters.
## Quickstart

### Install
```bash
pip install -r requirements.txt

### Diagnostics (NIS)
When measurement noise increases, the fixed-R Kalman filter tends to underestimate uncertainty, causing NIS to spike.
The adaptive-R (IAE) version adjusts R(t) online, making NIS more stable and closer to a reasonable range.
