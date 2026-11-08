#### Educational: core pandas data manipulation patterns
"""
A tour of everyday pandas operations on a small synthetic dataset of
model training runs:

  1. building a DataFrame and inspecting it
  2. cleaning: dtypes, missing values, duplicates
  3. filtering with boolean masks and .query()
  4. derived columns with vectorized ops and .assign()
  5. groupby + agg (named aggregations)
  6. pivot_table and melt (wide <-> long)
  7. merge (SQL-style joins)
  8. window ops: rolling and cumulative per group
  9. ranking within groups

Run:  python pandas_demo.py
"""

import numpy as np
import pandas as pd


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def make_runs(seed=0):
    rng = np.random.default_rng(seed)
    models = ["tiny", "small", "base"]
    rows = []
    for model in models:
        for lr in [1e-4, 3e-4, 1e-3]:
            for epoch in range(1, 6):
                base = {"tiny": 2.0, "small": 1.6, "base": 1.3}[model]
                loss = base / np.sqrt(epoch) + rng.normal(0, 0.05) + (0.2 if lr == 1e-3 else 0)
                rows.append({"model": model, "lr": lr, "epoch": epoch,
                             "val_loss": round(loss, 4),
                             "gpu_hours": str(round({"tiny": 0.5, "small": 2, "base": 6}[model] * epoch, 1))})
    df = pd.DataFrame(rows)
    # Inject some mess: a missing value and a duplicated row.
    df.loc[3, "val_loss"] = np.nan
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def main():
    df = make_runs()

    section("1. Inspect")
    print(df.head())
    print(df.dtypes)
    print("shape:", df.shape)

    section("2. Clean")
    df = df.drop_duplicates()
    df["gpu_hours"] = pd.to_numeric(df["gpu_hours"])        # str -> float
    df["model"] = df["model"].astype("category")
    print("missing per column:\n", df.isna().sum())
    # Fill a missing loss by interpolating within the same (model, lr) run.
    df["val_loss"] = df.groupby(["model", "lr"], observed=True)["val_loss"].transform(
        lambda s: s.interpolate(limit_direction="both"))
    print("after fill, missing:", int(df["val_loss"].isna().sum()))

    section("3. Filter")
    final = df[df["epoch"] == df["epoch"].max()]
    print(final.query("val_loss < 1.0 and lr <= 3e-4"))

    section("4. Derived columns")
    df = df.assign(
        perplexity=lambda d: np.exp(d["val_loss"]),
        loss_per_gpu_hour=lambda d: d["val_loss"] / d["gpu_hours"],
    )
    print(df[["model", "epoch", "val_loss", "perplexity"]].head())

    section("5. groupby + agg")
    summary = (df.groupby("model", observed=True)
                 .agg(best_loss=("val_loss", "min"),
                      mean_loss=("val_loss", "mean"),
                      total_gpu_hours=("gpu_hours", "sum"))
                 .sort_values("best_loss"))
    print(summary)

    section("6. pivot_table and melt")
    wide = df.pivot_table(index="model", columns="lr", values="val_loss",
                          aggfunc="min", observed=True)
    print(wide.round(3))
    long = wide.reset_index().melt(id_vars="model", var_name="lr", value_name="best_loss")
    print(long.head())

    section("7. merge")
    params = pd.DataFrame({"model": ["tiny", "small", "base", "large"],
                           "params_m": [10, 60, 220, 770]})
    joined = summary.reset_index().astype({"model": str}).merge(params, on="model", how="left")
    print(joined)

    section("8. Rolling / cumulative per group")
    df = df.sort_values(["model", "lr", "epoch"])
    grp = df.groupby(["model", "lr"], observed=True)
    df["loss_ma2"] = grp["val_loss"].transform(lambda s: s.rolling(2, min_periods=1).mean())
    df["cum_gpu_hours"] = grp["gpu_hours"].cumsum()
    print(df[["model", "lr", "epoch", "val_loss", "loss_ma2", "cum_gpu_hours"]].head(6))

    section("9. Rank within group")
    best = df[df["epoch"] == 5].copy()
    best["rank_in_model"] = best.groupby("model", observed=True)["val_loss"].rank(method="dense")
    print(best.loc[best["rank_in_model"] == 1, ["model", "lr", "val_loss"]])


if __name__ == "__main__":
    main()
