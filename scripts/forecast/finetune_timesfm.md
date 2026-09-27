# Fine-tuning TimesFM 2.5 on your own minute bars

This is a how-to, not a script JevTrader ships and runs for you: fine-tuning needs a GPU (or a
patient CPU) and network access to Hugging Face, neither of which this repo assumes you have on
the machine running the rest of the stack. Run it on your own machine, adapting the paths below.

It walks through adapting Google's own LoRA fine-tuning example
(`timesfm-forecasting/examples/finetuning/finetune_lora.py` in the TimesFM repo,
github.com/google-research/timesfm) to JevTrader's cached minute bars, and evaluating the
result with `jevtrader.forecast.evaluate` **before** wiring it into anything that trades.

Only fine-tune TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`, Apache-2.0). Never fine-tune or
distribute a checkpoint derived from TimesFM 3.0 (`google/timesfm-3.0-pytorch`) outside
research/backtest use -- its weights are licensed non-commercial, non-production only; see
`docs/FORECASTING.md#license`.

## 0. Prerequisites

- A machine with a GPU (recommended) or a lot of patience on CPU, and network access to
  huggingface.co (this container has neither -- that's why this is a doc, not a script run
  here).
- Clone `google-research/timesfm` and its fine-tuning example:

  ```bash
  git clone https://github.com/google-research/timesfm.git
  cd timesfm/timesfm-forecasting/examples/finetuning
  pip install transformers accelerate peft pandas pyarrow scikit-learn
  ```

- Your own bars, e.g. `data/cache/btcusd_bitstamp_1min_2025_2026.csv` (or your own symbol/venue)
  -- columns `timestamp, open, high, low, close, volume`.

## 1. Decide what you're fine-tuning on

Per the honest prior in `docs/FORECASTING.md`: fine-tune on **volatility or volume**, not raw
price/returns, unless you have a specific, tested reason to believe your instrument's short-
horizon returns are *not* close to a random walk. Concretely, feed the model one of:

- `jevtrader.forecast.targets.realized_vol_series(closes, window)` -- rolling realized vol, or
- `jevtrader.forecast.targets.log_volume_series(volumes)` -- log volume,

rather than raw prices or log returns. Both are the honest baselines' native targets too
(`EWMAVolForecaster`, `GARCH11Forecaster`, `SeasonalNaiveForecaster`), so the fine-tune is
scored against exactly the same target the baselines already forecast well.

## 2. Export a training series

From a `jev` (JevTrader) checkout, with the venv active:

```bash
python - <<'PY'
import numpy as np
import pandas as pd
from jevtrader.forecast.targets import realized_vol_series

df = pd.read_csv("data/cache/btcusd_bitstamp_1min_2025_2026.csv")
closes = df["close"].to_numpy(dtype=float)
rv = realized_vol_series(closes, window=15)          # 15-bar realized vol, your forecasting horizon's ballpark
rv = rv[~np.isnan(rv)]
np.save("rv_series.npy", rv.astype(np.float32))
print(f"{rv.size} points, mean={rv.mean():.6f}")
PY
```

For multiple symbols/venues, repeat per series and keep them as a `list[np.ndarray]` -- exactly
the shape `finetune_lora.py`'s `TimeSeriesRandomWindowDataset` expects (see step 3).

**Chronological split, no leakage.** Do not shuffle bars across the train/val boundary. Cut a
single, fixed timestamp (e.g. the last 20% of your date range) and put everything before it in
train, everything after it in val -- exactly what `jevtrader.forecast.evaluate`'s rolling-origin
functions already enforce for evaluation, and just as important during fine-tuning:

```python
cut = int(len(rv) * 0.8)
train_series, val_series = [rv[:cut]], [rv[cut:]]
```

If you have several symbols, split each one at its own 80% timestamp, not by shuffling series
together -- you want every val window's context to end before that symbol's own cutoff.

## 3. Adapt `finetune_lora.py`

The example script hard-codes a retail-sales Parquet download in `load_retail_sales`. Replace
that one function; the rest of the script (LoRA setup, training loop, CLI) works unchanged:

```python
def load_jevtrader_series(context_len, horizon_len, num_samples, seed):
    import numpy as np
    from finetune_lora import TimeSeriesRandomWindowDataset, TimeSeriesLastWindowDataset

    rv = np.load("rv_series.npy")
    cut = int(len(rv) * 0.8)
    train_arr, val_arr = rv[:cut], rv[cut:]

    train_ds = TimeSeriesRandomWindowDataset(
        [train_arr], context_len, horizon_len, num_samples=num_samples, seed=seed
    )
    val_ds = TimeSeriesLastWindowDataset([val_arr], context_len, horizon_len)
    return train_ds, val_ds
```

Then swap the call in `train()` (`load_retail_sales(...)` -> `load_jevtrader_series(...)`).
Everything else -- the `TimesFm2_5ModelForPrediction` load, LoRA config, training loop -- needs
no changes; it already does not externally normalize the series (TimesFM 2.5 does its own RevIN
internally), which matters for a vol series whose scale can drift across regimes.

Pick `--context_len`/`--horizon_len` to match how you'll actually call it from
`jevtrader.forecast.timesfm_backend.TimesFMForecaster` (e.g. `context_len=390` for a trading
day of 1-minute bars, `horizon_len=15` for a 15-minute-ahead vol forecast).

## 4. Train

```bash
python finetune_lora.py \
    --context_len 390 --horizon_len 15 \
    --epochs 15 --batch_size 32 --lr 5e-5 \
    --lora_r 8 --lora_alpha 16 \
    --num_samples 8000 \
    --output_dir jevtrader-btc-vol-lora
```

This saves a small PEFT adapter directory (a few MB, not a full 200M-parameter checkpoint) to
`--output_dir`. Keep the base model (`google/timesfm-2.5-200m-pytorch`, Apache-2.0) and this
adapter as two artifacts; you load them together at inference time.

## 5. Evaluate the fine-tune BEFORE using it -- against the baselines, not against itself

The whole point of `jevtrader.forecast.evaluate` is that a fine-tune only earns a place in the
stack if it beats `EWMAVolForecaster`/`GARCH11Forecaster` on your own out-of-sample bars, not
because it trained without error. Wrap the fine-tuned model with the standard `Forecaster`
protocol (`forecast(series, horizon) -> dict[str, QuantileForecast]`) -- same shape
`TimesFMForecaster` already provides for the un-tuned base checkpoint -- then:

```python
from jevtrader.forecast.baselines import EWMAVolForecaster, GARCH11Forecaster
from jevtrader.forecast.evaluate import evaluate_volatility, to_markdown

results = {
    "timesfm25_lora": {h: evaluate_volatility(val_bars, my_finetuned_forecaster, horizon=h) for h in (5, 15, 60)},
    "ewma": {h: evaluate_volatility(val_bars, EWMAVolForecaster(), horizon=h) for h in (5, 15, 60)},
    "garch": {h: evaluate_volatility(val_bars, GARCH11Forecaster(), horizon=h) for h in (5, 15, 60)},
}
print(to_markdown(results, title="Vol forecast: fine-tune vs. baselines"))
```

Run this on the **val** split from step 2 -- bars the fine-tune never saw during training -- and
only wire the fine-tune into `jevtrader.forecast.features.forecast_features` (or
`ForecastAdvisor`) if its QLIKE/MSE beats both baselines by a margin you'd trust with real size,
not by noise. If it doesn't win, the honest answer is to keep using the baseline; that's exactly
what the scoreboard is for.
