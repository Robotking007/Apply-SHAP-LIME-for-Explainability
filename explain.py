"""Train a random forest and explain its test predictions with SHAP and LIME."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
OUTPUT = Path("outputs")


def save_figure(name):
    plt.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

    wine = load_wine()
    feature_names = [name.replace("_", " ") for name in wine.feature_names]
    class_names = [f"Cultivar {index}" for index in range(len(wine.target_names))]
    train, test, y_train, y_test, _, test_rows = train_test_split(
        wine.data.astype(np.float32), wine.target, np.arange(len(wine.target)),
        test_size=0.25, stratify=wine.target, random_state=SEED,
    )
    model = RandomForestClassifier(
        n_estimators=150, max_depth=5, random_state=SEED, n_jobs=1,
    )
    model.fit(train, y_train)
    predictions = model.predict(test)
    probabilities = model.predict_proba(test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    # Explain the predicted class for the first held-out row.
    row = 0
    class_index = int(np.argmax(probabilities[row]))
    class_id = int(model.classes_[class_index])
    probability = float(probabilities[row, class_index])
    tree_explainer = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent",
        model_output="raw", feature_names=feature_names,
    )
    shap_values = tree_explainer(test)[:, :, class_index]
    shap_totals = shap_values.base_values + shap_values.values.sum(axis=1)
    expected = probabilities[:, class_index]
    max_error = float(np.max(np.abs(shap_totals - expected)))
    np.testing.assert_allclose(shap_totals, expected, rtol=0, atol=1e-6)

    shap.plots.bar(shap_values, max_display=8, show=False)
    plt.title(f"Global SHAP importance for {class_names[class_id]}")
    save_figure("shap_global.png")
    shap.plots.waterfall(shap_values[row], max_display=8, show=False)
    save_figure("shap_local.png")

    lime_explainer = LimeTabularExplainer(
        train, feature_names=feature_names, class_names=class_names,
        mode="classification", discretize_continuous=True, random_state=SEED,
    )
    lime_result = lime_explainer.explain_instance(
        test[row], model.predict_proba, labels=(class_index,),
        num_features=6, num_samples=5000,
    )
    lime_weights = lime_result.as_list(label=class_index)
    fig, ax = plt.subplots(figsize=(8, 3.3), layout="constrained")
    conditions, weights = zip(*reversed(lime_weights))
    ax.barh(conditions, weights, color=["#1769aa" if w > 0 else "#b44136" for w in weights])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Signed local surrogate coefficient")
    ax.set_title(f"LIME explanation for {class_names[class_id]}")
    save_figure("lime_local.png")

    global_scores = np.abs(shap_values.values).mean(axis=0)
    global_order = np.argsort(-global_scores)
    local_order = np.argsort(-np.abs(shap_values.values[row]))
    result = {
        "samples": {"total": len(wine.data), "train": len(train), "test": len(test)},
        "features": len(feature_names),
        "class_names": class_names,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=model.classes_).tolist(),
        "test_row": row,
        "original_dataset_row": int(test_rows[row]),
        "actual_class": class_names[int(y_test[row])],
        "explained_class": class_names[class_id],
        "model_probability": probability,
        "shap_base": float(shap_values.base_values[row]),
        "shap_sum": float(shap_values.values[row].sum()),
        "shap_reconstructed": float(shap_totals[row]),
        "shap_max_additivity_error": max_error,
        "shap_global": {feature_names[i]: float(global_scores[i]) for i in global_order},
        "shap_local": {feature_names[i]: float(shap_values.values[row, i]) for i in local_order},
        "lime_local": [
            {"feature": feature_names[i], "condition": condition, "weight": float(weight)}
            for (i, _), (condition, weight) in zip(lime_result.local_exp[class_index], lime_weights)
        ],
        "lime_surrogate_prediction": float(np.asarray(lime_result.local_pred).item()),
        "lime_local_r2": float(lime_result.score),
    }
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("Assignment 9 | Apply SHAP/LIME for Explainability")
    print(f"Wine dataset: {len(wine.data)} rows, {len(feature_names)} features, {len(class_names)} classes")
    print(f"Train/test: {len(train)}/{len(test)} | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"Test row {row} (dataset row {test_rows[row]}): actual {result['actual_class']}, predicted {class_names[class_id]}")
    print(f"SHAP: {result['shap_base']:.4f} + ({result['shap_sum']:+.4f}) = {shap_totals[row]:.4f}")
    print(f"SHAP max probability reconstruction error: {max_error:.2e}")
    print(f"LIME: surrogate {result['lime_surrogate_prediction']:.4f} | Local R^2: {result['lime_local_r2']:.4f}")
    print("Saved: outputs/results.json and three explanation plots")


if __name__ == "__main__":
    main()
