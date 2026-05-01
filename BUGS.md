# bugs i found in the starter colab

each one has the cell, the offending lines, how to reproduce, expected
behavior, and what actually happens.

---

## 1. dead EODHD api token (HARD BUG, blocks the whole notebook)

**where**: cell 2, lines defining the data fetch

**offending lines**:
```python
api_token = '68efdceed38ec8.15967744'
symbol    = 'USDCHF.FOREX'
end_date  = datetime.now().strftime('%Y-%m-%d')
start_date= (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
prices    = get_daily_data(symbol, start_date, end_date, api_token)
```

**repro**: open the notebook, run cell 2 as-is.

**expected**: `prices` populated with 10 years of USDCHF daily closes.

**actual** (visible in the saved notebook output):
```
RuntimeError: Erreur 401
---------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipython-input-3150741174.py in <cell line: 0>()
     27 end_date  = datetime.now().strftime('%Y-%m-%d')
     28 start_date= (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
---> 29 prices    = get_daily_data(symbol, start_date, end_date, api_token)
...
RuntimeError: Erreur 401
```

EODHD rejects the token (likely expired / over quota / never refreshed
since the notebook was authored). Anyone running the starter as-is is
blocked at the very first step.

**impact**: the entire notebook below this cell never executes.

---

## 2. NameError waiting in `theoretical_option_price` else branch

**where**: cell 3, function `theoretical_option_price`

**offending code**:
```python
def theoretical_option_price(S0, K, T, r, paths, option_type, is_cyber=True):
    t_index = int(T * 252)
    if is_cyber:
        ST = paths[:, t_index]
    else:
        ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*np.random.randn(len(paths)))
    ...
```

**repro**: extracted the function verbatim and called with is_cyber=False.

```python
import numpy as np

def theoretical_option_price(S0, K, T, r, paths, option_type, is_cyber=True):
    t_index = int(T * 252)
    if is_cyber:
        ST = paths[:, t_index]
    else:
        ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*np.random.randn(len(paths)))
    ...

theoretical_option_price(1.0, 1.0, 0.1, 0.05, np.zeros((100,100)),
                         'call', is_cyber=False)
```

**expected**: a price.

**actual** (just ran it):
```
NameError: name 'sigma' is not defined
```

neither the function arguments nor the surrounding module scope defines
a bare `sigma` (only `sigma_fig`, `sigma_bt`, `sigma2` exist).

**impact**: dormant. nothing in the notebook calls this branch (every
caller uses the default `is_cyber=True`). still a real latent bug.

---

## 3. `fetch_market_options_data` is fake AND ignores its arguments

**where**: cell 3

**offending code**:
```python
def fetch_market_options_data(symbol, expiration_date):
    strikes = np.linspace(K_put * 0.98, K_call * 1.02, 20)
    data = {
        'strike': strikes,
        'call_price': [theoretical_call * (0.9 + 0.2*np.random.rand()) for _ in strikes],
        'put_price' : [theoretical_put  * (0.9 + 0.2*np.random.rand()) for _ in strikes],
        'call_delta': [greeks_call['delta'] * (0.95 + 0.1*np.random.rand()) for _ in strikes],
        'put_delta' : [greeks_put['delta']  * (0.95 + 0.1*np.random.rand()) for _ in strikes],
    }
    return pd.DataFrame(data)
```

**repro**: extracted the function and called it with two totally
different (symbol, expiration_date) pairs under the same RNG seed.

```python
np.random.seed(0)
df_btc = fetch_market_options_data('BTCUSDT', '2026-12-01')
np.random.seed(0)
df_xyz = fetch_market_options_data('UTTER_NONSENSE', 'never')
```

**expected** (if the function actually fetched market data): different
prices for different symbols / expirations.

**actual** (just ran it):
```
Test 1 - synthetic data check:
  call_price range: [4.520, 5.464]
  observed: prices stay in 4.500-5.500 range (90-110% of theoretical_call=5.0)

Test 2 - arguments ignored check:
  same RNG seed, totally different (symbol, expiration_date) args
  observed: dataframes identical? True
```

confirmed: prices are bounded to 90-110% of the model's own theoretical
price (not real market behavior), and the function returns IDENTICAL
output for completely different symbols / expirations. it does not make
any network call.

**downstream impact**: the notebook prints a "BUY / SELL / HOLD" trading
recommendation by comparing model price to this fake "market" price.
the recommendation is the model arguing with random noise around itself.

---

## 4. brief claims `evaluate(predictions)` exists; it doesn't

**where**: brief vs. cell 2

**brief says** (Part A description):
> "The Colab has a helper function `evaluate(predictions)` that computes
> all three for you. You don't implement them yourself."

**repro**: search the notebook for `def evaluate` or any usage of
`evaluate(`. zero hits.

**actual**: metrics are inlined inside `backtest_confidence_intervals`:
```python
winkler = (width95 + (2/alpha)*(low95-actual)) if actual < low95 else \
          (width95 + (2/alpha)*(actual-high95)) if actual > high95 else \
          width95
```

**impact**: a candidate trusting the brief looks for `evaluate()`,
doesn't find it, and either rebuilds it from scratch or gives up.

---

## 5. brief says starter uses Binance; it uses EODHD

**where**: brief FAQ vs. cell 2

**brief FAQ says**:
> "Use https://data-api.binance.vision/api/v3/klines instead - same
> endpoint, no geo block, fully public. The starter Colab uses this
> by default."

**actual code in cell 2**:
```python
def get_daily_data(symbol, start_date, end_date, api_token):
    url = (
        f'https://eodhd.com/api/eod/{symbol}'
        f'?api_token={api_token}&from={start_date}&to={end_date}&fmt=json'
    )
    ...

api_token = '68efdceed38ec8.15967744'
symbol    = 'USDCHF.FOREX'
```

**impact**: not a runtime bug, a documentation contradiction. the
candidate has to swap the entire data layer (different vendor,
different schema, different time grain), not just toggle a flag.

---

## 6. `np.random.seed(42)` set once globally before a 252-iter MC backtest

**where**: cell 2

**offending sequence**:
```python
np.random.seed(42)         # set once, near the top
...
def simulate_cyber_gbm(...):
    ...
    Z = np.random.standard_t(nu) * np.sqrt((nu - 2) / nu)   # consumes global state
    ...

def backtest_confidence_intervals(prices, train=504, test=252):
    ...
    for i in tqdm(range(train, train + test)):    # 252 iterations
        ...
        paths_bt = simulate_mc(..., n_sims=10_000, n_days=1)   # 10k MC each iter
```

**repro**: simulate the starter's RNG pattern and try to reproduce
"iteration 137" in isolation.

```python
# scenario 1: full backtest from seeded start
np.random.seed(42)
all_iters = [np.random.standard_t(5) for _ in range(252)]
iter_137_in_full = all_iters[137]

# scenario 2: re-seed and try to re-run "just iteration 137"
np.random.seed(42)
iter_137_solo = np.random.standard_t(5)
```

**expected** (if iterations were independently reproducible): both
values equal.

**actual** (just ran it):
```
iter 137 inside the full run:  -0.088310
iter  0  if we re-seed alone:  +0.559634
different? True
```

252 iterations x 10,000 MC paths = 2,520,000 calls to
`np.random.standard_t` all sharing the same global RNG state. the script
is reproducible end-to-end (same seed, same total output) but a single
iteration cannot be reproduced in isolation. this hides bugs: if
iteration 137 produces weird coverage, you can't re-run just iteration
137 to debug.

**fix in our submission**: pass an explicit `np.random.default_rng(42)`
instance through the call chain (see `backtest.py`).

---

## 7. FIGARCH chosen for 1-hour 1-step-ahead forecasts

**where**: cell 2

**offending code**:
```python
am = arch_model(log_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
```

**why this is a design issue** (not a runtime bug, but worth filing):
FIGARCH's distinguishing property is fractional-integration "long
memory" - past vol shocks decay slowly across many bars. that property
only manifests at long forecast horizons (multi-day, multi-week).

at 1-step-ahead, FIGARCH and plain GARCH(1,1) produce nearly identical
forecasts. FIGARCH adds a fractional-differencing parameter `d` whose
estimate is noisy on small windows and adds nothing to a 1-hour
forecast.

the starter was originally written for daily forex (USDCHF) where
FIGARCH might be more defensible. for our task it's overkill.

**fix in our submission**: replaced FIGARCH with EWMA on Garman-Klass
per-bar variance.

---

## 8. non-standard "cyber-GBM" layer with 5 unjustified constants

**where**: cell 2, between the FIGARCH fit and the simulation

**offending code** (excerpt):
```python
def rolling_entropy(x, window=60, bins=20):
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)

H_series = rolling_entropy(resid)
M_series = log_ret.abs().rolling(60).mean()
redundancy = 1 + 0.1 * np.log1p(prices.rolling(5).var() / prices.rolling(20).var())
info_filter = (H_series > H_series.mean()).astype(float)

α0, δ0 = 0.5, 0.3
base_params = {'alpha': α0, 'delta': δ0, 'gamma': 0.2, 'kappa': 0.1, 'eta': 1e-3}

def update_params(p, sigma2, bar_sigma2, t):
    err = sigma2 - bar_sigma2
    lr  = p['eta'] / (1 + t**0.55)
    p['gamma'] = np.clip(p['gamma'] + lr * err, 0.01, 0.5)
    return p

def simulate_cyber_gbm(...):
    ...
    crisis  = (H_val > 0.8) or (M_val > 0.8)
    ...
    sigma2 = (
        sigma_fig.iloc[current]**2 * (1 + params['alpha'] * H_val + delta_t * M_val)
        + params['gamma'] * (bar_sigma2 - sigma2)
    )
    sigma2 *= max(1e-12, redundancy.iloc[current])
    sigma2 *= 1 + 0.5 * info_filter.iloc[current]
```

**why this is a design issue**:
- none of these mechanisms appear in standard volatility-modelling
  literature (no GARCH/HAR/realized-variance variant introduces rolling
  entropy of residuals)
- the five constants (alpha=0.5, delta=0.3, gamma=0.2, kappa=0.1,
  eta=1e-3) have no source, derivation, or justification in the notebook
- `redundancy` uses `prices.rolling(N).var()` - variance of raw prices,
  dominated by the price level/trend, NOT volatility. to measure a
  vol ratio you'd compute variance of *log returns*. as coded this is
  effectively a "trendiness" indicator masquerading as a vol ratio
- the brief itself says "everything else in the starter is plumbing -
  focus on the three concepts," which i read as license to strip this

**fix in our submission**: removed entirely.

---

## 9. ~150 lines of off-topic options pricing

**where**: cell 3 (the entire third cell)

**what's there**:
- `theoretical_option_price()`
- `optimal_strikes()`
- `calculate_greeks()`
- `optimal_duration()`
- `fetch_market_options_data()` (the fake one, see bug #3)
- a French print block titled "STRATEGIE OPTIONS POUR USDCHF.FOREX"
- `plot_option_strategy()` final chart

**why this is a bug for this assignment**: the AlphaI assignment is to
predict a 95% interval for BTC's next-hour close. options strikes,
greeks, strangle prices, and BUY/SELL/HOLD recommendations are not
asked for, not graded, and add ~150 lines of noise to the file. this is
~30% of the entire starter.

**fix in our submission**: removed entirely. cell 3 contributes nothing
to the assignment deliverable.
