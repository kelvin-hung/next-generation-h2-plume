# VE Forward Predictor (Streamlit)

Forward VE plume prediction tool:
- Upload NPZ with `phi` and `k` 2D arrays
- Upload CSV schedule with columns `t` and `q` (q>0 injection, q<0 production)
- Run forward simulation → view Sg maps + area/r_eq time series
- Download outputs ZIP

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
