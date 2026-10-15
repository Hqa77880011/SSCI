import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .core import ForwardModel, ResponseService, invert, quantile


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def generate(args):
    rng = np.random.default_rng(args.seed)
    state = np.c_[rng.uniform(0, 20, args.records), rng.uniform(20, 80, args.records)]
    context = rng.uniform(-1, 1, (args.records, 1))
    features = np.c_[np.ones(args.records), state[:, 0] / 20, state[:, 1] / 100,
                     state[:, 0] * state[:, 1] / 2000, (state[:, 0] / 20) ** 2, context]
    coefficient = rng.normal(0, 1, (features.shape[1], args.channels))
    coefficient[2, :args.channels // 2] = coefficient[1, :args.channels // 2] * .9
    noise = .03 * rng.normal(size=(args.records, args.channels)) + .02 * rng.normal(size=(args.records, 1))
    response = features @ coefficient + noise
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, y=state, x=context, z=response, group=np.arange(args.records))


def prepare(args):
    data = read(args.input)
    data.setdefault("x", np.empty((len(data["y"]), 0)))
    data.setdefault("group", np.arange(len(data["y"])))
    if any(len(v) != len(data["y"]) or not np.isfinite(v).all() for v in data.values()):
        raise ValueError("All arrays must be finite with matching record counts")
    groups = np.unique(data["group"])
    first, second, third = [int(f * len(groups)) for f in [.5, .65, .8]]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, selected in zip(["train", "development", "calibration", "test"],
                              [groups[:first], groups[first:second], groups[second:third], groups[third:]]):
        mask = np.isin(data["group"], selected)
        if not mask.any():
            raise ValueError("Too few groups for four nonempty partitions")
        np.savez_compressed(output / f"{name}.npz", **{key: value[mask] for key, value in data.items()})


def train(args):
    root = Path(args.data)
    training, development = read(root / "train.npz"), read(root / "development.npz")
    domain = np.asarray(args.domain).reshape(-1, 2)
    if domain.shape[0] != training["y"].shape[1] or np.any(domain[:, 1] <= domain[:, 0]):
        raise ValueError("Domain must supply an increasing lower/upper pair for each state coordinate")
    start = time.perf_counter()
    forward = ForwardModel().fit(training["y"], training["x"], training["z"],
                    (development["y"], development["x"], development["z"]), args.ridge)
    count = min(args.baseline_channels, training["z"].shape[1])
    features = np.c_[training["z"][:, :count], training["x"]]
    rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=args.seed,
                               n_jobs=-1).fit(features, training["y"][:, 0])
    svr = make_pipeline(StandardScaler(), SVR(C=10, epsilon=.1)).fit(features, training["y"][:, 0])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"forward": forward, "domain": domain, "baseline_channels": count,
                 "rf": rf, "svr": svr, "training_seconds": time.perf_counter() - start}, output / "model.joblib")


def calibrate(args):
    bundle, data = joblib.load(args.model), read(args.input)
    model = bundle["forward"]
    residual = np.max(abs(data["z"] - model.predict(data["y"], data["x"])) / model.response_scale, axis=1)
    unit_scores = [residual[data["group"] == group].max() for group in np.unique(data["group"])]
    bundle["q"] = quantile(unit_scores, args.alpha)
    bundle["alpha"] = args.alpha
    count = bundle["baseline_channels"]
    features = np.c_[data["z"][:, :count], data["x"]]
    for name in ["rf", "svr"]:
        scores = abs(data["y"][:, 0] - bundle[name].predict(features))
        unit_scores = [scores[data["group"] == group].max() for group in np.unique(data["group"])]
        bundle[f"{name}_q"] = quantile(unit_scores, args.alpha)
    joblib.dump(bundle, args.output)


def evaluate(args):
    bundle, data = joblib.load(args.model), read(args.input)
    model, q = bundle["forward"], bundle["q"]
    records = []
    for index, context in enumerate(data["x"]):
        start = time.perf_counter()
        if args.variant in {"rf", "svr"}:
            count = bundle["baseline_channels"]
            features = np.r_[data["z"][index, :count], context][None]
            point = float(bundle[args.variant].predict(features)[0])
            radius = bundle[f"{args.variant}_q"]
            result = {"point": point, "interval": [point - radius, point + radius],
                      "components": [[point - radius, point + radius]], "accepted": radius <= args.epsilon,
                      "incompatible": False, "charged_cost": float(count), "channels": list(range(count)),
                      "retained_cells": 0, "gamma": None}
        else:
            service = ResponseService(data["z"][index])
            result = invert(model, context, service, bundle["domain"], args.subdivisions,
                            q, args.budget, args.epsilon, args.policy, args.seed + index, args.variant)
        result["latency_ms"] = 1000 * (time.perf_counter() - start)
        result["record"] = index
        result["group"] = int(data["group"][index])
        truth = float(data["y"][index, 0])
        result["reference"] = truth
        interval = result["interval"]
        result["coverage"] = bool(interval is not None and interval[0] <= truth <= interval[1])
        result["hull_width"] = float(interval[1] - interval[0]) if interval is not None else None
        result["error"] = result["point"] - truth if result["point"] is not None else None
        result["accepted_error"] = bool(result["accepted"] and abs(result["error"]) > args.epsilon)
        result["physical_interval"] = [interval[0] - args.reference_bound, interval[1] + args.reference_bound] if interval else None
        if interval is not None and not np.isfinite(interval).all():
            result["interval"] = [None if not np.isfinite(v) else v for v in interval]
            result["components"] = [[None if not np.isfinite(v) else v for v in c] for c in result["components"]]
            result["physical_interval"] = [None if not np.isfinite(v) else v for v in result["physical_interval"]]
            result["hull_width"] = None
        records.append(result)
    errors = [r["error"] for r in records if r["error"] is not None]
    accepted = [r for r in records if r["accepted"]]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "records.json", records)
    metrics = {"variant": args.variant, "records": len(records), "point_records": len(errors),
               "mae": float(np.mean(np.abs(errors))) if errors else None,
               "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
               "bias": float(np.mean(errors)) if errors else None,
               "coverage": float(np.mean([r["coverage"] for r in records])),
               "abstention_rate": float(np.mean([not r["accepted"] for r in records])),
               "incompatibility_rate": float(np.mean([r["incompatible"] for r in records])),
               "accepted_error_unconditional": float(np.mean([r["accepted_error"] for r in records])),
               "accepted_error_conditional": float(np.mean([r["accepted_error"] for r in accepted])) if accepted else None,
               "accepted_records": len(accepted), "training_seconds": bundle["training_seconds"]}
    for name in ["charged_cost", "latency_ms", "hull_width", "retained_cells", "gamma"]:
        values = [r[name] for r in records if r.get(name) is not None]
        metrics[name] = float(np.mean(values)) if values else None
    metrics["mean_components"] = float(np.mean([len(r["components"]) for r in records]))
    metrics["readout_fraction"] = float(np.mean([len(r["channels"]) / data["z"].shape[1] for r in records]))
    repeat_variances = []
    for group in np.unique(data["group"]):
        selected = [r for r in records if r["group"] == group and r["point"] is not None]
        if len(selected) > 1 and np.ptp([r["reference"] for r in selected]) < 1e-8:
            repeat_variances.append(np.var([r["point"] for r in selected], ddof=1))
    metrics["repeatability"] = float(np.sqrt(np.mean(repeat_variances))) if repeat_variances else None
    ranges = []
    for low, high in zip(args.range_edges[:-1], args.range_edges[1:]):
        selected = [r for r in records if low <= r["reference"] < high]
        if selected:
            errors_in_range = [r["error"] for r in selected if r["error"] is not None]
            widths = [r["hull_width"] for r in selected if r["hull_width"] is not None]
            ranges.append({"range": [low, high], "records": len(selected),
                "coverage": float(np.mean([r["coverage"] for r in selected])),
                "rmse": float(np.sqrt(np.mean(np.square(errors_in_range)))) if errors_in_range else None,
                "width": float(np.mean(widths)) if widths else None})
    write(output / "ranges.json", ranges)
    metrics["group_simultaneous_coverage"] = float(np.mean([
        all(r["coverage"] for r in records if r["group"] == group) for group in np.unique(data["group"])]))
    write(output / "metrics.json", metrics)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = list(Path(args.input).rglob("metrics.json"))
    if not paths:
        raise ValueError("No evaluated metrics found")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text()) for p in paths]
    labels = [str(p.parent.relative_to(args.input)) for p in paths]
    for key in ["mae", "rmse", "bias", "coverage", "hull_width", "abstention_rate", "charged_cost", "latency_ms"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, [np.nan if r[key] is None else r[key] for r in records], color="#4b8390")
        ax.set_ylabel(key)
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(output / f"{key}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(prog="ssci")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--output", required=True)
    p.add_argument("--records", type=int, default=2000)
    p.add_argument("--channels", type=int, default=14)
    p.add_argument("--seed", type=int, default=13)
    p = sub.add_parser("prepare")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--domain", nargs="+", type=float, default=[0, 20, 0, 100])
    p.add_argument("--baseline-channels", type=int, default=6)
    p.add_argument("--ridge", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=13)
    p = sub.add_parser("calibrate")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alpha", type=float, default=.05)
    p = sub.add_parser("evaluate")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=["outer", "plugin", "point_grid", "rf", "svr"], default="outer")
    p.add_argument("--policy", choices=["separation", "fixed", "random"], default="separation")
    p.add_argument("--subdivisions", nargs="+", type=int, default=[32, 16])
    p.add_argument("--budget", type=float, default=6)
    p.add_argument("--epsilon", type=float, default=.5)
    p.add_argument("--reference-bound", type=float, default=0.)
    p.add_argument("--seed", type=int, default=20013)
    p.add_argument("--range-edges", nargs="+", type=float, default=[0., 5., 15., 20.000001])
    p = sub.add_parser("plot")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
