# AI-guided high-throughput discovery of iridium- and ruthenium-free palladium-oxide catalysts for durable acidic oxygen evolution

This repository contains data and code to reproduce the figures in the accompanying manuscript.

**Citation:** Incoming.

## Setup

`fig_5a.py` and `fig_5b.py` rasterise SVG to PNG with `cairosvg`, which depends on
the native **cairo** C library. Install it with your system package manager first:

```bash
# macOS (Homebrew)
brew install cairo

# Debian/Ubuntu
sudo apt-get install libcairo2

# Fedora
sudo dnf install cairo
```

Then install the Python dependencies (a virtual environment is recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, if `cairosvg` cannot find the library at runtime
(`no library called "cairo-2" was found`), point the loader at Homebrew's lib dir:

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/lib
```

## Reproducing figures

Each script is self-contained and reads its inputs from `data/`, writing outputs to `figures/`.
Run them from the repository root:

```bash
python scripts/fig_2c.py
python scripts/fig_3a.py
python scripts/fig_3b.py
python scripts/fig_3c.py
python scripts/fig_3d.py
python scripts/fig_3e.py
python scripts/fig_4i.py
python scripts/fig_5a.py
python scripts/fig_5b.py
```

## Figures

| Script      | Reads from `data/`                                                                                                         | Writes to `figures/` |
|-------------|----------------------------------------------------------------------------------------------------------------------------|----------------------|
| `fig_2c.py` | `oer_exp_data.csv`                                                                                                          | `fig_2c.png`         |
| `fig_3a.py` | `fig_3a-d.parquet`                                                                                                        | `fig_3a.png`         |
| `fig_3b.py` | `fig_3a-d.parquet`                                                                                                        | `fig_3b.png`         |
| `fig_3c.py` | `fig_3a-d.parquet`                                                                                                        | `fig_3c.png`         |
| `fig_3d.py` | `fig_3a-d.parquet`                                                                                                        | `fig_3d.png`         |
| `fig_3e.py` | `fig_3e.csv`                                                                                                                | `fig_3e.png`         |
| `fig_4i.py` | `fig_4i.csv`                                                                                                                | `fig_4i.png`         |
| `fig_5a.py` | `fig_5a_active_search.csv`, `fig_5a_seq_learning_agent.csv`, `fig_5a_bo.csv`, `fig_5a_llm.csv`, `fig_5a_random.csv`         | `fig_5a.png`         |
| `fig_5b.py` | `fig_5b_active_search.csv`, `fig_5b_random.csv`, `fig_5b_bo.csv`, `fig_5b_seq_learning_agent.csv`, `fig_5b_llm_with_feedback.csv`, `fig_5b_llm_no_feedback.csv` | `fig_5b.png`         |
