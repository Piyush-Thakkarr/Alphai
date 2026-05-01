# Issues found in the provided starter notebook

For pasting into the submission form's "bugs you spotted" field.

## Real bugs / things that would prevent the code from running

**1. Dead EODHD API token.**
The hardcoded `api_token = '68efdceed38ec8.15967744'` returns 401
Unauthorized. Anyone running the starter as-is hits the error
`RuntimeError: Erreur 401` immediately on the first cell.

**2. Markdown auto-linkification breaks Python identifiers.**
The Colab as shared has several places where dotted method names have
been mangled into Markdown-style links by whatever rendered the notebook,
e.g. `[pd.to](http://pd.to)_datetime`, `[am.fit](http://am.fit)`,
`[plt.show](http://plt.show)`, `[stats.t.fit](http://stats.t.fit)`,
`[USDCHF.FOREX](http://USDCHF.FOREX)`, `[datetime.now](http://datetime.now)`.
These won't run as Python; they need to be `pd.to_datetime`, `am.fit`,
`plt.show`, etc.

**3. `np.random.seed(42)` set once globally before a 720-iteration MC backtest.**
The backtest loop runs 10,000-path Monte Carlo on each iteration without
re-seeding. All iterations share evolving RNG state. Re-runs are
reproducible end-to-end, but per-iteration determinism is implicit and
fragile to any change in iteration order. Better: pass a `default_rng`
explicitly.

## Brief / starter inconsistencies

**4. The brief claims the Colab has a helper function `evaluate(predictions)`.**
There is no such function. The starter inlines the metric computation
(coverage / width / Winkler) inside `backtest_confidence_intervals`. If a
candidate trusts the brief and looks for `evaluate()`, they don't find it.

**5. The brief claims "the starter Colab uses Binance by default."**
The starter actually uses EODHD (`eodhd.com`) for USDCHF.FOREX daily
forex data. The Binance switch is the candidate's job — not a default.

## Design choices in the starter that don't fit the assignment

**6. FIGARCH on 1-hour 1-step-ahead forecasts.**
FIGARCH's distinguishing property is fractional-integration "long memory"
in volatility. That property only manifests at long forecast horizons (multi-day,
multi-week). For a 1-step-ahead 1-hour forecast, FIGARCH and GARCH(1,1)
produce nearly identical predictions while FIGARCH adds an extra
fractional-differencing parameter `d` whose estimate is noisy on small
windows. Switched to GARCH-flavored EWMA on Garman-Klass instead.

**7. Non-standard "cyber" feature layer with five hardcoded hyperparameters.**
The starter bolts onto plain GBM:
- rolling Shannon entropy of residuals
- a "crisis detection" rule on entropy and absolute return magnitude
- a "redundancy" multiplier from a variance ratio
- a binary `info_filter` based on entropy mean
- an online "parameter learning" loop (`update_params`)
- five constants (alpha=0.5, delta=0.3, gamma=0.2, kappa=0.1, eta=1e-3)

None of these are in the standard volatility-modelling literature and the
five constants are not justified anywhere. The brief itself says
"everything else in the starter is plumbing — focus on the three concepts."
I read that as instruction to strip this layer and did.

**8. Subtle look-ahead risk in any "fit once on full sample" approach.**
Not a bug in the starter directly, but a trap I want to flag: any
parameter (e.g. Student-t df) that's fitted once on the full 720-bar
sample and then used to predict bars within that sample is using future
information. My implementation refits per window so the fit at bar N
sees only bars [..N-1].

## Options-pricing section

The starter's last ~150 lines compute strangle prices, Greeks, and
optimal duration. Completely off-topic for the AlphaI assignment. Removed
in my submission.
