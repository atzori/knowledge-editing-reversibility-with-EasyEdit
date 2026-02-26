from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tabulate import tabulate

def _to_float(x: Any) -> Optional[float]:
    """Best-effort conversion of numeric-like values to float."""
    if x is None:
        return None
    if isinstance(x, np.generic):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return None


def _value_to_str(v: Any) -> str:
    """Convert a metric leaf value into a compact string."""
    f = _to_float(v)
    if f is not None:
        return f"{f:.4f}"

    if isinstance(v, list):
        # If list has a single numeric element, print it as a single scalar
        # (common for acc=[1.0] or acc=[0.0]).
        if len(v) == 1:
            f1 = _to_float(v[0])
            if f1 is not None:
                return f"{f1:.4f}"

        # If list is numeric -> show mean and length.
        floats = [x for x in (_to_float(xi) for xi in v) if x is not None]
        if floats and len(floats) == len(v):
            mean = sum(floats) / len(floats)
            return f"mean={mean:.4f} (n={len(floats)})"
        # Otherwise, fallback to stringified list.
        return str(v)

    if isinstance(v, dict):
        # Flatten nested dict on a single line.
        parts = []
        for kk, vv in v.items():
            parts.append(f"{kk}={_value_to_str(vv)}")
        return ", ".join(parts) if parts else "{}"

    return str(v)


def metrics_to_str_dict(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Convert a metrics dict into key -> string values."""
    return {str(k): _value_to_str(v) for k, v in (metrics or {}).items()}


def _iter_metric_cases(metrics: Any) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Normalize metrics into a list of (case_label, pre_dict, post_dict).
    Supports:
      - list[{'pre':..., 'post':..., 'case_id':...}]
      - {'pre':..., 'post':...}
      - {'case_0': {'pre':..., 'post':...}, ...}
    """
    if metrics is None:
        return []

    # 1) list of cases
    if isinstance(metrics, list):
        out: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for i, m in enumerate(metrics):
            if isinstance(m, dict):
                label = str(m.get("case_id", m.get("case", i)))
                pre = m.get("pre", {}) if isinstance(m.get("pre", {}), dict) else {}
                post = m.get("post", {}) if isinstance(m.get("post", {}), dict) else {}
                out.append((label, pre, post))
            else:
                out.append((str(i), {}, {}))
        return out

    # 2) single dict with pre/post
    if isinstance(metrics, dict) and "pre" in metrics and "post" in metrics:
        pre = metrics.get("pre", {}) if isinstance(metrics.get("pre", {}), dict) else {}
        post = metrics.get("post", {}) if isinstance(metrics.get("post", {}), dict) else {}
        label = str(metrics.get("case_id", metrics.get("case", 0)))
        return [(label, pre, post)]

    # 3) dict of cases
    if isinstance(metrics, dict):
        out2: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for k, v in metrics.items():
            if isinstance(v, dict) and "pre" in v and "post" in v:
                pre = v.get("pre", {}) if isinstance(v.get("pre", {}), dict) else {}
                post = v.get("post", {}) if isinstance(v.get("post", {}), dict) else {}
                out2.append((str(k), pre, post))
        if out2:
            return out2

    # Fallback: unknown structure
    return [("0", {}, {})]


def _filter_metric_groups(metric_dict: Dict[str, Any], include_groups: Optional[Sequence[str]]) -> Dict[str, Any]:
    if not include_groups:
        return metric_dict

    groups = {g.strip().lower() for g in include_groups if str(g).strip()}
    out: Dict[str, Any] = {}

    if "rewrite" in groups:
        for k, v in metric_dict.items():
            if str(k).startswith("rewrite_"):
                out[k] = v

    if "rephrase" in groups:
        for k, v in metric_dict.items():
            if str(k).startswith("rephrase_"):
                out[k] = v

    if "neighborhood" in groups:
        loc = metric_dict.get("locality")
        if isinstance(loc, dict):
            loc_filtered = {k: v for k, v in loc.items() if str(k).startswith("neighborhood_")}
            if loc_filtered:
                out["locality"] = loc_filtered

    return out


def _keep_only_accuracy(metric_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in metric_dict.items():
        ks = str(k)
        if ks.endswith("_acc"):
            out[k] = v
        elif ks == "locality" and isinstance(v, dict):
            loc_acc = {lk: lv for lk, lv in v.items() if str(lk).endswith("_acc")}
            if loc_acc:
                out[k] = loc_acc
    return out


def print_metrics_table(
    metrics: Any,
    *,
    title: str,
    include_groups: Optional[Sequence[str]] = None,
    accuracy_only: bool = False,
) -> None:
    """Print PRE vs POST metrics in a readable table."""
    cases = _iter_metric_cases(metrics)
    print(f"\n=== {title} ===")

    if not cases:
        print("(no metrics)")
        return

    for case_label, pre_dict, post_dict in cases:
        pre_filtered = _filter_metric_groups(pre_dict, include_groups)
        post_filtered = _filter_metric_groups(post_dict, include_groups)
        if accuracy_only:
            pre_filtered = _keep_only_accuracy(pre_filtered)
            post_filtered = _keep_only_accuracy(post_filtered)

        pre = metrics_to_str_dict(pre_filtered)
        post = metrics_to_str_dict(post_filtered)

        keys = sorted(set(pre.keys()) | set(post.keys()))
        rows = [[k, pre.get(k, "-"), post.get(k, "-")] for k in keys]

        print(f"\n[CASE {case_label}]")
        print(tabulate(rows, headers=["metric", "pre", "post"], tablefmt="github"))


def print_hparams_table(hparams: Any) -> None:
    """Print hparams as a table: name | value."""
    print("\n=== HPARAMS ===")

    if hparams is None:
        print("(no hparams)")
        return

    if isinstance(hparams, dict):
        data = hparams
    elif hasattr(hparams, "__dict__"):
        data = vars(hparams)
    else:
        data = {"hparams": str(hparams)}

    rows = [[str(k), str(data[k])] for k in sorted(data.keys())]
    print(tabulate(rows, headers=["name", "value"], tablefmt="github"))


def print_color(text: str, color_name: str):
    # Simple function to print colored text in the terminal
    color_codes = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "gray": "90",
    }
    color_code = color_codes.get(color_name, "")
    print(f"\033[{color_code}m{text}\033[0m")
