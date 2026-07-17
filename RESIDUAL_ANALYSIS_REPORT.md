# Residual Analysis Report

Phase 3 of the Linear Regression Engine diagnostics suite -- per-target residual audit (`linear_regression.residual_diagnostics`) over each target's held-out test-split predictions: normality (Shapiro-Wilk / Jarque-Bera / Q-Q), heteroscedasticity (Breusch-Pagan), autocorrelation (Durbin-Watson), and a market-regime error breakdown.

| Target | Mean resid | Std resid | Normal (5%) | Heteroscedastic (5%) | Durbin-Watson |
|---|---:|---:|:---:|:---:|---:|
| next_close | -0.0002 | 0.0006 | no | no | 1.2498 |
| next_high | -0.0002 | 0.0006 | no | no | 1.3022 |
| next_low | -0.0002 | 0.0005 | no | no | 1.1890 |
| next_return | -0.0001 | 0.0005 | no | no | 1.5537 |
| expected_pip_movement | -1.1740 | 6.1792 | no | no | 1.5532 |
| future_volatility | -0.0000 | 0.0001 | no | no | 1.3476 |
| maximum_favorable_excursion | -0.0001 | 0.0006 | no | no | 1.6929 |
| maximum_adverse_excursion | 0.0000 | 0.0003 | no | yes | 1.1477 |

## Per-target detail

### `next_close`

- Shapiro-Wilk: stat=0.6344, p=0.0000; Jarque-Bera: stat=7888.5014, p=0.0000 -> **non-normal at 5%**
- Q-Q correlation: 0.7886 (n=113)
- Breusch-Pagan: LM=0.0086, p=0.9260 -> **homoscedastic at 5%**
- Durbin-Watson: 1.2498 (~2.0 = no autocorrelation, <2 = positive, >2 = negative)
- Regime breakdown (MAE / RMSE):
  - trending: n=34, MAE=0.0003, RMSE=0.0004
  - ranging: n=79, MAE=0.0005, RMSE=0.0007
  - high_volatility: n=3 (insufficient samples)
  - low_volatility: n=0 (insufficient samples)
  - london_session: n=36, MAE=0.0006, RMSE=0.0010
  - newyork_session: n=48, MAE=0.0005, RMSE=0.0009
  - asian_session: n=36, MAE=0.0004, RMSE=0.0004
  - sydney_session: n=41, MAE=0.0004, RMSE=0.0004

### `next_high`

- Shapiro-Wilk: stat=0.6049, p=0.0000; Jarque-Bera: stat=9502.9773, p=0.0000 -> **non-normal at 5%**
- Q-Q correlation: 0.7693 (n=113)
- Breusch-Pagan: LM=0.0080, p=0.9286 -> **homoscedastic at 5%**
- Durbin-Watson: 1.3022 (~2.0 = no autocorrelation, <2 = positive, >2 = negative)
- Regime breakdown (MAE / RMSE):
  - trending: n=34, MAE=0.0003, RMSE=0.0004
  - ranging: n=79, MAE=0.0005, RMSE=0.0007
  - high_volatility: n=3 (insufficient samples)
  - low_volatility: n=0 (insufficient samples)
  - london_session: n=36, MAE=0.0005, RMSE=0.0010
  - newyork_session: n=48, MAE=0.0005, RMSE=0.0009
  - asian_session: n=36, MAE=0.0003, RMSE=0.0004
  - sydney_session: n=41, MAE=0.0004, RMSE=0.0004

### `next_low`

- Shapiro-Wilk: stat=0.7597, p=0.0000; Jarque-Bera: stat=2741.9524, p=0.0000 -> **non-normal at 5%**
- Q-Q correlation: 0.8653 (n=113)
- Breusch-Pagan: LM=0.1716, p=0.6787 -> **homoscedastic at 5%**
- Durbin-Watson: 1.1890 (~2.0 = no autocorrelation, <2 = positive, >2 = negative)
- Regime breakdown (MAE / RMSE):
  - trending: n=34, MAE=0.0003, RMSE=0.0004
  - ranging: n=79, MAE=0.0004, RMSE=0.0006
  - high_volatility: n=3 (insufficient samples)
  - low_volatility: n=0 (insufficient samples)
  - london_session: n=36, MAE=0.0005, RMSE=0.0008
  - newyork_session: n=48, MAE=0.0005, RMSE=0.0007
  - asian_session: n=36, MAE=0.0003, RMSE=0.0004
  - sydney_session: n=41, MAE=0.0003, RMSE=0.0004

### `next_return`

- Shapiro-Wilk: stat=0.6189, p=0.0000; Jarque-Bera: stat=8902.3249, p=0.0000 -> **non-normal at 5%**
- Q-Q correlation: 0.7784 (n=113)
- Breusch-Pagan: LM=0.0212, p=0.8843 -> **homoscedastic at 5%**
- Durbin-Watson: 1.5537 (~2.0 = no autocorrelation, <2 = positive, >2 = negative)
- Regime breakdown (MAE / RMSE):
  - trending: n=34, MAE=0.0002, RMSE=0.0002
  - ranging: n=79, MAE=0.0004, RMSE=0.0006
  - high_volatility: n=3 (insufficient samples)
  - low_volatility: n=0 (insufficient samples)
  - london_session: n=36, MAE=0.0005, RMSE=0.0009
  - newyork_session: n=48, MAE=0.0004, RMSE=0.0008
  - asian_session: n=36, MAE=0.0003, RMSE=0.0003
  - sydney_session: n=41, MAE=0.0002, RMSE=0.0003

### `expected_pip_movement`

- Shapiro-Wilk: stat=0.6192, p=0.0000; Jarque-Bera: stat=8880.2531, p=0.0000 -> **non-normal at 5%**
- Q-Q correlation: 0.7786 (n=113)
- Breusch-Pagan: LM=0.0221, p=0.8819 -> **homoscedastic at 5%**
- Durbin-Watson: 1.5532 (~2.0 = no autocorrelation, <2 = positive, >2 = negative)
- Regime breakdown (MAE / RMSE):
  - trending: n=34, MAE=2.0585, RMSE=2.8294
  - ranging: n=79, MAE=4.3866, RMSE=7.2899
  - high_volatility: n=3 (insufficient samples)
  - low_volatility: n=0 (insufficient samples)
  - london_session: n=36, MAE=5.6268, RMSE=9.8885
  - newyork_session: n=48, MAE=4.6591, RMSE=8.6543
  - asian_session: n=36, MAE=3.0335, RMSE=3.6915
  - sydney_session: n=41, MAE=2.4666, RMSE=3.0994

### `future_volatility`

- Shapiro-Wilk: stat=0.5209, p=0.0000; Jarque-Bera: stat=14892.9909, p=0.0000 -> **non-normal at 5%**
- Q-Q correlation: 0.7122 (n=113)
- Breusch-Pagan: LM=0.6900, p=0.4062 -> **homoscedastic at 5%**
- Durbin-Watson: 1.3476 (~2.0 = no autocorrelation, <2 = positive, >2 = negative)
- Regime breakdown (MAE / RMSE):
  - trending: n=34, MAE=0.0000, RMSE=0.0001
  - ranging: n=79, MAE=0.0001, RMSE=0.0001
  - high_volatility: n=3 (insufficient samples)
  - low_volatility: n=0 (insufficient samples)
  - london_session: n=36, MAE=0.0001, RMSE=0.0002
  - newyork_session: n=48, MAE=0.0001, RMSE=0.0002
  - asian_session: n=36, MAE=0.0000, RMSE=0.0001
  - sydney_session: n=41, MAE=0.0000, RMSE=0.0001

### `maximum_favorable_excursion`

- Shapiro-Wilk: stat=0.4121, p=0.0000; Jarque-Bera: stat=24221.6018, p=0.0000 -> **non-normal at 5%**
- Q-Q correlation: 0.6324 (n=113)
- Breusch-Pagan: LM=0.3432, p=0.5580 -> **homoscedastic at 5%**
- Durbin-Watson: 1.6929 (~2.0 = no autocorrelation, <2 = positive, >2 = negative)
- Regime breakdown (MAE / RMSE):
  - trending: n=34, MAE=0.0002, RMSE=0.0002
  - ranging: n=79, MAE=0.0003, RMSE=0.0007
  - high_volatility: n=3 (insufficient samples)
  - low_volatility: n=0 (insufficient samples)
  - london_session: n=36, MAE=0.0004, RMSE=0.0009
  - newyork_session: n=48, MAE=0.0003, RMSE=0.0008
  - asian_session: n=36, MAE=0.0003, RMSE=0.0003
  - sydney_session: n=41, MAE=0.0002, RMSE=0.0003

### `maximum_adverse_excursion`

- Shapiro-Wilk: stat=0.9322, p=0.0000; Jarque-Bera: stat=47.3320, p=0.0000 -> **non-normal at 5%**
- Q-Q correlation: 0.9644 (n=113)
- Breusch-Pagan: LM=8.2017, p=0.0042 -> **heteroscedastic at 5%**
- Durbin-Watson: 1.1477 (~2.0 = no autocorrelation, <2 = positive, >2 = negative)
- Regime breakdown (MAE / RMSE):
  - trending: n=34, MAE=0.0001, RMSE=0.0002
  - ranging: n=79, MAE=0.0002, RMSE=0.0003
  - high_volatility: n=3 (insufficient samples)
  - low_volatility: n=0 (insufficient samples)
  - london_session: n=36, MAE=0.0003, RMSE=0.0003
  - newyork_session: n=48, MAE=0.0003, RMSE=0.0003
  - asian_session: n=36, MAE=0.0002, RMSE=0.0002
  - sydney_session: n=41, MAE=0.0001, RMSE=0.0001

