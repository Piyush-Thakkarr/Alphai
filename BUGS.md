# bugs i found in the starter colab

things that just don't run

1. the eodhd api token in the starter is dead. line says
   `api_token = '68efdceed38ec8.15967744'` and the first cell throws 401
   immediately. the saved output in the notebook even shows the error
   traceback so it's clearly never been refreshed. anyone running the
   notebook as-is gets blocked at the data fetch.

2. `theoretical_option_price` has an `else` branch with a NameError
   waiting to happen. the line is

       ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.randn(len(paths)))

   `sigma` is never defined in any scope here. calling it with
   `is_cyber=False` blows up. the only reason the notebook doesn't
   crash on this is because nothing actually calls that branch.

3. `fetch_market_options_data` is fake. the function name suggests it
   pulls market option prices but it just returns

       'call_price': [theoretical_call * (0.9 + 0.2 * np.random.rand()) for _ in strikes]

   random noise around the model's own theoretical price. then the code
   downstream prints a recommendation of "BUY" / "SELL" / "HOLD" the
   strangle based on this fake "market" price. the recommendation is
   meaningless because it's the model arguing with random noise around
   itself.

stuff the brief got wrong

4. brief says the colab has a helper called `evaluate(predictions)`. it
   doesn't. metrics are inlined inside `backtest_confidence_intervals`.
   if you trust the brief and grep for `evaluate` you find nothing.

5. brief says the starter uses binance by default. it actually uses
   eodhd for `USDCHF.FOREX` daily data. swapping to binance was the
   whole point of the assignment for me, not a default i could just
   inherit.

6. `np.random.seed(42)` set once globally before a 252-iteration MC
   backtest loop. all 252 iterations share evolving RNG state. it's
   reproducible end-to-end but you can't reproduce a single iteration
   in isolation, which makes debugging weird coverage spikes harder.

design choices that don't fit a 1-hour-ahead forecast

7. FIGARCH on 1-hour 1-step. FIGARCH's whole reason for existing is
   long-memory volatility decay (vol shocks staying alive for weeks). at
   1 step ahead this property doesn't even get a chance to manifest.
   FIGARCH and plain GARCH(1,1) give almost identical 1-step forecasts,
   and FIGARCH adds a fractional-differencing parameter that's noisy on
   small windows. i replaced it with EWMA on garman-klass per-bar
   variance.

8. there's a whole "cyber-GBM" layer bolted on top of plain GBM:

   - rolling shannon entropy of residuals
   - a `crisis` flag = `(H_val > 0.8) or (M_val > 0.8)`
   - a "redundancy" multiplier from a 5-vs-20-bar variance ratio
   - an `info_filter` binary flag from entropy mean
   - `update_params` running an online learning loop on `gamma` during
     each simulation
   - five constants: alpha=0.5, delta=0.3, gamma=0.2, kappa=0.1,
     eta=1e-3, with no justification anywhere in the notebook

   none of this is in any standard volatility textbook. the brief itself
   says "everything else in the starter is plumbing — focus on the three
   concepts" so i read that as license to strip the layer.

9. the whole third cell (~150 lines) computes options strikes, greeks,
   strangle prices and a buy/sell recommendation. completely unrelated
   to a 95% interval forecast for the next hour. removed.
