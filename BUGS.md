# Issues found in the provided starter notebook

For pasting into the submission form's "bugs you spotted" field.

## Real bugs / things that break or mislead

**1. Dead EODHD API token.**
The hardcoded `api_token = '68efdceed38ec8.15967744'` returns 401
Unauthorized. Anyone running the starter as-is hits the error
`RuntimeError: Erreur 401` immediately on the first cell — the saved
notebook output even shows this exact traceback.

**2. `np.random.seed(42)` set once globally before a 252-iteration MC backtest.**
The backtest loop runs 10,000-path Monte Carlo on each iteration without
re-seeding. All iterations share evolving RNG state. Re-runs are
reproducible end-to-end, but per-iteration determinism is implicit and
fragile to any change in iteration order. Better: pass an explicit
`np.random.default_rng()` and thread it through the simulation.

**3. `fetch_market_options_data()` is fake.**
The function name implies it fetches market option prices. It actually
generates synthetic random data:
```python
'call_price': [theoretical_call * (0.9 + 0.2 * np.random.rand()) for _ in strikes]
'put_price' : [theoretical_put  * (0.9 + 0.2 * np.random.rand()) for _ in strikes]
```
The downstream "trading recommendation" (BUY / SELL / HOLD the strangle)
compares the model's own theoretical price against random noise around
itself. The recommendation is meaningless.

**4. Undefined `sigma` in `theoretical_option_price`.**
The `else` branch references a variable `sigma` that is never defined in
any scope:
```python
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.randn(len(paths)))
```
Calling `theoretical_option_price(..., is_cyber=False)` raises NameError.
Dead branch, but it's still committed code.

## Brief / starter inconsistencies

**5. The brief claims the Colab has a helper function `evaluate(predictions)`.**
There is no such function in the notebook. The starter inlines the
metric computation (coverage / width / Winkler) inside
`backtest_confidence_intervals`. If a candidate trusts the brief and
looks for `evaluate()`, they don't find it.

**6. The brief claims "the starter Colab uses Binance by default."**
The starter actually uses EODHD (`https://eodhd.com/api/eod/{symbol}`)
for `USDCHF.FOREX` daily forex data. The Binance switch is the candidate's
job — not a default.

## Design choices in the starter that don't fit the assignment

**7. FIGARCH on 1-hour 1-step-ahead forecasts.**
FIGARCH's distinguishing property is fractional-integration "long memory"
in volatility. That property only manifests at long forecast horizons
(multi-day, multi-week). For a 1-step-ahead 1-hour forecast, FIGARCH and
GARCH(1,1) produce nearly identical predictions while FIGARCH adds an
extra fractional-differencing parameter `d` whose estimate is noisy on
small windows. I switched to EWMA-smoothed Garman-Klass — same family,
no MLE required.

**8. Non-standard "cyber" feature layer with five hardcoded hyperparameters.**
The starter bolts onto plain GBM:
- rolling Shannon entropy of residuals (`rolling_entropy`)
- a "crisis detection" rule on entropy and absolute-return magnitude
  (`crisis = (H_val > 0.8) or (M_val > 0.8)`)
- a "redundancy" multiplier from a variance ratio
- a binary `info_filter` based on entropy mean
- an online "parameter learning" loop (`update_params`) that mutates
  `gamma` during simulation
- five constants with no justification in the notebook:
  `alpha=0.5, delta=0.3, gamma=0.2, kappa=0.1, eta=1e-3`

None of this is in the standard volatility-modelling literature. The
brief itself says "everything else in the starter is plumbing — focus on
the three concepts." I read that as instruction to strip this layer and
did.

## Off-topic content

**9. ~150 lines of options pricing.**
The third cell computes strangle prices, Greeks, optimal strike/duration
selection, and fake "market" prices for a buy/sell recommendation. None
of this is relevant to the AlphaI assignment, which is about a 95%
interval forecast for the next hour. Removed in my submission.
