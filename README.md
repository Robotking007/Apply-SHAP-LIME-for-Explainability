# Assignment 9 · Apply SHAP/LIME for Explainability

Train a random forest on scikit-learn’s Wine dataset and interpret its predictions with Tree SHAP and LIME. The script uses Python 3.10 and a seeded, stratified 75/25 split. It downloads no data at runtime.

## Run in PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe explain.py
```

## How the explanations work

- **Global SHAP:** Mean absolute Tree SHAP values across all 45 held-out rows, for Cultivar 0. Tree SHAP uses the training sample counts stored along tree paths as its reference distribution. The forest’s native output for this class is a probability.
- **Local SHAP:** Signed contributions for the first test row, which the model predicted as Cultivar 0. The script verifies that the baseline plus the contributions reconstructs the class probability on every test row.
- **Local LIME:** A weighted linear surrogate fitted on 5,000 perturbed instances. The six signed coefficients describe feature intervals for the same row and class. The weighted local R² measures how well the surrogate fits the forest on the sampled neighborhood.

## Reference run

| Measurement | Value |
| --- | ---: |
| Training / test rows | 133 / 45 |
| Test accuracy | 1.0000 |
| Test macro F1 | 1.0000 |
| Cultivar 0 forest probability for dataset row 35 (zero-based) | 0.9212 |
| SHAP baseline + contributions | 0.3273 + 0.5939 |
| Maximum SHAP probability reconstruction error | 4.44e−16 |
| LIME surrogate prediction for the same row | 0.5911 |
| LIME weighted local R² | 0.1864 |

Proline ranked first across the test set for Cultivar 0. Flavanoids had the largest absolute local SHAP contribution and the largest local LIME coefficient for the selected row. The low local R² and the gap in probability call for caution when interpreting the LIME weights. The scores describe one small held-out split.

## Outputs

- [Machine-readable results](outputs/results.json)
- [Global SHAP importance chart](outputs/shap_global.png)
- [Local SHAP waterfall](outputs/shap_local.png)
- [Local LIME coefficients](outputs/lime_local.png)

## References

- [scikit-learn Wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
- [SHAP TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)
- [LIME Tabular API](https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_tabular.LimeTabularExplainer)
