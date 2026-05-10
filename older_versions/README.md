# older_versions/

Cod care a fost folosit în iterațiile anterioare ale KernelTrap și nu mai e activ în
codebase-ul curent. Păstrat aici pentru referință (audit, comparare în teză, posibilă
recuperare a unor idei). **Niciunul din aceste fișiere nu este importat / executat
de aplicația activă.**

## isolation_forest/

### `beth_iforest_v1.py`
Prima versiune a pipeline-ului de antrenare Isolation Forest pe BETH.
- 7 features numerice de bază (`processId`, `parentProcessId`, `userId`, `mountNamespace`,
  `eventId`, `argsNum`, `returnValue`).
- Un singur prag global, optimizat prin sweeping de percentile pentru F1-score
  față de label-ul `evil`.
- Citește tot CSV-ul în memorie (fără streaming).
- Hyperparametri: `n_estimators=300`, `contamination=0.01`.
- Output binar: `malicious_pred ∈ {0, 1}`.

### `beth_iforest_v2.py`
A doua versiune. Două inovații față de v1:
- **Praguri per-entitate** (per `userId` sau per `host:userId`).
- **Severitate în două trepte** (low + high).
- Streaming chunked (`chunksize=500_000`) cu `partial_fit` pe scaler.
- Reservoir sampling (max 2M rânduri benigne).
- Hyperparametri: `n_estimators=600`, `contamination=0.03`.

Înlocuită de `beth_iforest.py` din directorul activ, care:
- Extinde features de la 7 la 22 (derivate din `args`, `processName`, `timestamp`,
  `threadId`).
- Folosește per-`(host, userId)` thresholds (group_mode = `host_user`).
- Mărește `n_estimators` la 2000 și `train_sample` la 10M.

## central_server/

### `window.py`
Versiunea inițială a sliding window tracker-ului. Două diferențe față de
`window_v3.py` activ:
- `drain_pivots()` returnează `List[Tuple[str, int]]` (doar `(host, uid)`),
  nu `(host, uid, trigger_reason)`.
- Nu are pragul de rată (`MIN_SEV2_RATE`); declanșează doar pe count.

Înlocuită de `window_v3.py`, care:
- Adaugă trigger-ul dual: count ≥ `PIVOT_THRESHOLD` AND rate ≥ `MIN_SEV2_RATE`.
- Returnează tuple cu motivul declanșării pentru audit.
