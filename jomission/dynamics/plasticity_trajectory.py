from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Any
import numpy as np

from jomission.network.connectivity import HIERARCHY


@dataclass(frozen=True)
class EdgePartition:
    """Authoritative structural partition of network edges."""
    n_edges: int
    pre: np.ndarray
    post: np.ndarray
    receptor_index: np.ndarray
    src_areas: tuple[str, ...]
    tgt_areas: tuple[str, ...]
    area_pairs: tuple[str, ...]
    src_layers: tuple[str, ...]
    tgt_layers: tuple[str, ...]
    layer_pairs: tuple[str, ...]
    src_classes: tuple[str, ...]
    tgt_classes: tuple[str, ...]
    class_pairs: tuple[str, ...]
    projection_types: tuple[str, ...]  # 'recurrent', 'FF', 'FB'

    @classmethod
    def from_model(cls, model: Any) -> "EdgePartition":
        """Build edge partition from a Jomission model."""
        tbl = model.neuron_table()
        el = model.params["edge_list"]
        pre = np.asarray(el.pre, dtype=np.int32)
        post = np.asarray(el.post, dtype=np.int32)
        rec_idx = np.asarray(el.receptor_index, dtype=np.int32)
        n_edges = int(pre.shape[0])

        areas = [r["area"] for r in tbl]
        layers = [r["layer"] for r in tbl]
        cts = [r["cell_type"] for r in tbl]

        src_areas = []
        tgt_areas = []
        area_pairs = []
        src_layers = []
        tgt_layers = []
        layer_pairs = []
        src_classes = []
        tgt_classes = []
        class_pairs = []
        proj_types = []

        for e in range(n_edges):
            p = pre[e]
            q = post[e]
            sa, ta = areas[p], areas[q]
            sl, tl = layers[p], layers[q]
            sc, tc = cts[p], cts[q]

            src_areas.append(sa)
            tgt_areas.append(ta)
            area_pairs.append(f"{sa}->{ta}")
            src_layers.append(sl)
            tgt_layers.append(tl)
            layer_pairs.append(f"{sl}->{tl}")
            src_classes.append(sc)
            tgt_classes.append(tc)
            class_pairs.append(f"{sc}->{tc}")

            if sa == ta:
                pt = "recurrent"
            elif HIERARCHY.index(sa) < HIERARCHY.index(ta):
                pt = "FF"
            else:
                pt = "FB"
            proj_types.append(pt)

        partition = cls(
            n_edges=n_edges,
            pre=pre,
            post=post,
            receptor_index=rec_idx,
            src_areas=tuple(src_areas),
            tgt_areas=tuple(tgt_areas),
            area_pairs=tuple(area_pairs),
            src_layers=tuple(src_layers),
            tgt_layers=tuple(tgt_layers),
            layer_pairs=tuple(layer_pairs),
            src_classes=tuple(src_classes),
            tgt_classes=tuple(tgt_classes),
            class_pairs=tuple(class_pairs),
            projection_types=tuple(proj_types),
        )
        partition.validate()
        return partition

    def validate(this) -> None:
        """Verify that partitions are exhaustive and non-overlapping."""
        pt_counts = {"recurrent": 0, "FF": 0, "FB": 0}
        for pt in this.projection_types:
            if pt in pt_counts:
                pt_counts[pt] += 1
            else:
                raise ValueError(f"Unknown projection type: {pt}")
        total_pt = sum(pt_counts.values())
        if total_pt != this.n_edges:
            raise ValueError(f"Partition count {total_pt} != n_edgep {this.n_edges}")


def compute_subset_metrics(
    w_0: np.ndarray,
    w_t: np.ndarray,
    indices: np.ndarray,
    w_floor: float = 0.01,
    w_ceiling: float = 10.0,
    eps: float = 1e-4,
) -> dict[str, float]:
    """Compute exact plasticity trajectory metrics for a subset of edges."""
    n = len(indices)
    if n == 0:
        return {
            "n_edges": 0,
            "mean_w0": 0.0,
            "mean_wt": 0.0,
            "delta_w": 0.0,
            "gain": 1.0,
            "d2_displacement": 0.0,
            "correlation": 1.0,
            "frac_potentiated": 0.0,
            "frac_depressed": 0.0,
            "frac_unchanged": 1.0,
            "frac_at_floor": 0.0,
            "frac_at_ceiling": 0.0,
            "sign_changes": 0,
            "quantiles": {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0},
        }

    w0_sub = np.asarray(w_0)[indices]
    wt_sub = np.asarray(w_t)[indices]

    mag0 = np.abs(w0_sub)
    magt = np.abs(wt_sub)

    sum_mag0 = float(np.sum(mag0))
    sum_magt = float(np.sum(magt))
    gain = (sum_magt / sum_mag0) if sum_mag0 > 1e-12 else 1.0

    diff = wt_sub - w0_sub
    norm_diff = float(np.linalg.norm(diff))
    norm_w0 = float(np.linalg.norm(w0_sub))
    d2 = (norm_diff / norm_w0) if norm_w0 > 1e-12 else 0.0

    std0 = float(np.std(w0_sub))
    stdt = float(np.std(wt_sub))
    if std0 > 1e-12 and stdt > 1e-12:
        corr = float(np.corrcoef(w0_sub, wt_sub)[0, 1])
    else:
        corr = 1.0 if np.allclose(w0_sub, wt_sub) else 0.0

    delta_mag = magt - mag0
    potentiated = float(np.mean(delta_mag > eps))
    depressed = float(np.mean(delta_mag < -eps))
    unchanged = float(np.mean(np.abs(delta_mag) <= eps))

    at_floor = float(np.mean(magt <= (w_floor + 1e-6)))
    at_ceiling = float(np.mean(magt >= (w_ceiling - 1e-6)))
    sign_changes = int(np.sum((w0_sub * wt_sub) < 0))

    quantiles = {
        "p10": float(np.percentile(magt, 10)),
        "p25": float(np.percentile(magt, 25)),
        "p50": float(np.percentile(magt, 50)),
        "p75": float(np.percentile(magt, 75)),
        "p90": float(np.percentile(magt, 90)),
    }

    return {
        "n_edges": int(n),
        "mean_w0": float(np.mean(mag0)),
        "mean_wt": float(np.mean(magt)),
        "delta_w": float(np.mean(delta_mag)),
        "gain": float(gain),
        "d2_displacement": float(d2),
        "correlation": float(corr),
        "frac_potentiated": potentiated,
        "frac_depressed": depressed,
        "frac_unchanged": unchanged,
        "frac_at_floor": at_floor,
        "frac_at_ceiling": at_ceiling,
        "sign_changes": sign_changes,
        "quantiles": quantiles,
    }


def summarize_plasticity_trajectory(
    w_0: np.ndarray,
    w_t: np.ndarray,
    partition: EdgePartition,
    w_floor: float = 0.01,
    w_ceiling: float = 10.0,
    eps: float = 1e-4,
) -> dict[str, Any]:
    """Produce the complete hierarchical plasticity summary."""
    all_indices = np.arange(partition.n_edges)
    proj_arr = np.array(partition.projection_types)
    area_arr = np.array(partition.area_pairs)
    src_c_arr = np.array(partition.src_classes)
    tgt_c_arr = np.array(partition.tgt_classes)
    pair_c_arr = np.array(partition.class_pairs)

    summary: dict[str, Any] = {
        "global": compute_subset_metrics(w_0, w_t, all_indices, w_floor, w_ceiling, eps),
        "by_projection_type": {},
        "by_area_pair": {},
        "by_class_pair": {},
        "by_source_class": {},
        "by_target_class": {},
    }

    for pt in ("recurrent", "FF", "FB"):
        idx = np.where(proj_arr == pt)[0]
        summary["by_projection_type"][pt] = compute_subset_metrics(
            w_0, w_t, idx, w_floor, w_ceiling, eps
        )

    for ap in sorted(set(partition.area_pairs)):
        idx = np.where(area_arr == ap)[0]
        summary["by_area_pair"][ap] = compute_subset_metrics(
            w_0, w_t, idx, w_floor, w_ceiling, eps
        )

    for cp in sorted(set(partition.class_pairs)):
        idx = np.where(pair_c_arr == cp)[0]
        summary["by_class_pair"][cp] = compute_subset_metrics(
            w_0, w_t, idx, w_floor, w_ceiling, eps
        )

    for sc in ("E", "PV", "SST", "VIP"):
        idx = np.where(src_c_arr == sc)[0]
        summary["by_source_class"][sc] = compute_subset_metrics(
            w_0, w_t, idx, w_floor, w_ceiling, eps
        )

    for tc in ("E", "PV", "SST", "VIP"):
        idx = np.where(tgt_c_arr == tc)[0]
        summary["by_target_class"][tc] = compute_subset_metrics(
            w_0, w_t, idx, w_floor, w_ceiling, eps
        )

    return summary
