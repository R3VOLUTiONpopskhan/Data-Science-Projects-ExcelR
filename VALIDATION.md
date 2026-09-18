# Validation

Both notebooks were executed from fresh kernels on Python 3.12. Every code cell completed without an error. `requirements-tested.txt` records the analysis environment versions.

- Source sales CSV matches the supplied CSV byte for byte.
- No missing sales values; date strings parse and agree with weekday labels.
- Gross sales, discount amount, and net sales reconcile to their defining arithmetic.
- Standardized columns have zero means and population-convention SDs of one within numerical tolerance.
- The feature matrix has 450 rows and 86 columns; all category indicators are binary and each row has one active level per categorical variable.
- Both interval formulas agree with SciPy interval functions. The mean, sample SD, and normal critical value also agree with independent standard-library calculations.

These checks verify the calculations, not population representativeness or suitability for an unspecified future predictive model.
