# Data provenance & management

This repository is a **starter codebase** that runs on **synthetic data** out of the box
(`python -m src.train`). Real bike-share availability data is **not committed** — replace the
synthetic inputs with real GBFS `station_status` snapshots locally (see
[`data/README.md`](data/README.md)).

## Sources

| Layer | Location | In git? | Notes |
|:--|:--|:--|:--|
| **Synthetic data** | generated in code | n/a | default; reproducible from `SEED` |
| **Real GBFS status** | `data/status_villes/<city>/*.parquet` | **No** (git-ignored) | ≈180 MB; fetched per-city from GBFS `station_status`, under provider terms — single source of truth = the feeds via the collector |
| **Reference assets** | `data/reference/` | Yes | small reference inputs the starter ships with |

## Reproducibility

- No real third-party data is committed; only synthetic generation + small reference assets.
- **Code**: MIT (`LICENSE`). Real GBFS data inherits provider terms and is not redistributed.
