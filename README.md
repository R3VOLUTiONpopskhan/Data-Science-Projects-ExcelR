# Data Science Projects — ExcelR

Two completed statistics projects with executable notebooks, source data, visualizations, and saved results.

| Project | Questions and methods | Notebook |
| --- | --- | --- |
| [Sales statistics](01-basic-statistics-sales-discounts/) | Descriptive statistics, histograms, boxplots, category counts, z-score standardization, one-hot encoding | [View analysis](01-basic-statistics-sales-discounts/notebooks/sales_statistics.ipynb) |
| [Print-head durability](02-confidence-interval-print-head-durability/) | Two-sided 99% confidence intervals using Student t and known-sigma z methods | [View analysis](02-confidence-interval-print-head-durability/notebooks/confidence_intervals.ipynb) |

## Run locally

Use Python 3.12 or newer. From the repository root:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open either notebook and select **Restart Kernel and Run All Cells**. Notebook outputs are already saved for browsing on GitHub. Alternatively, change into either project folder and run `python src/analysis.py`. Results and chart files are regenerated in that project.

## Data and scope

The source material is the two supplied ExcelR assignment ZIPs. The sales CSV is preserved unchanged; print-head measurements are transcribed from the confidence-interval assignment. Monetary units are unspecified in the sales source. These are educational analyses, with assumptions and limitations documented in each notebook. No predictive performance or causal effect is claimed.
