"""将 Nsight Systems SQLite 中的权重收集与对应层的 GPU 前向配对。"""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median


# NVTX 标的是 CPU 提交范围，真正用于比较的是 CUPTI 的 GPU 时间戳。
# CUDA correlationId 只在进程内唯一，globalTid 高位用于匹配进程。
KERNELS_BY_RANGE = """
WITH ranges AS (
    SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value) AS label
    FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId
    WHERE n.end IS NOT NULL
)
SELECT DISTINCT v.label, k.deviceId, k.streamId, k.start, k.end, names.value
FROM ranges v
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
    ON r.globalTid = v.globalTid AND r.start >= v.start AND r.start < v.end
JOIN CUPTI_ACTIVITY_KIND_KERNEL k
    ON k.correlationId = r.correlationId AND (k.globalPid >> 24) = (r.globalTid >> 24)
JOIN StringIds names ON names.id = k.shortName
WHERE v.label GLOB 'FSDP gather *' OR v.label GLOB 'FSDP demand *' OR v.label GLOB 'FSDP compute *'
ORDER BY k.start
"""


def analyze(database: Path, layers: list[str], world_size: int) -> dict:
    by_layer = defaultdict(lambda: defaultdict(list))
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        for label, device, stream, start, end, name in connection.execute(KERNELS_BY_RANGE):
            _, phase, layer = label.split(" ", 2)
            if phase == "gather" and "allgather" not in name.lower().replace("_", ""):
                continue
            by_layer[device, layer][phase].append((start, end, stream, name))

    ranks = []
    for device in range(world_size):
        layer_rows = []
        for layer in layers:
            ranges = by_layer[device, layer]
            missing = {"gather", "demand", "compute"} - ranges.keys()
            if missing:
                raise ValueError(f"GPU {device} {layer}: Nsight 缺少 {', '.join(sorted(missing))} GPU kernel")
            gather_start = min(row[0] for row in ranges["gather"])
            gather_end = max(row[1] for row in ranges["gather"])
            compute_start, _, compute_stream, compute_name = ranges["compute"][0]
            # 地标排在当前层的权重等待之前，可区分预取已就绪与计算流还要等权重两种情况。
            demand_ns = max(row[1] for row in ranges["demand"])
            if any(row[2] != compute_stream for row in ranges["demand"]):
                raise ValueError(f"GPU {device} {layer}: 需求地标与前向使用了不同的计算流")
            layer_rows.append(
                {
                    "layer": layer,
                    "gather_start_ns": gather_start,
                    "gather_end_ns": gather_end,
                    "gather_stream": ranges["gather"][0][2],
                    "gather_kernel": ranges["gather"][0][3],
                    "demand_ns": demand_ns,
                    "compute_start_ns": compute_start,
                    "compute_stream": compute_stream,
                    "compute_kernel": compute_name,
                    "gather_us": (gather_end - gather_start) / 1000,
                    "ready_lead_us": (demand_ns - gather_end) / 1000,
                    "unhidden_us": max(gather_end - demand_ns, 0) / 1000,
                    "gather_end_to_compute_us": (compute_start - gather_end) / 1000,
                }
            )
        gather_times = [row["gather_us"] for row in layer_rows]
        late = [row["unhidden_us"] for row in layer_rows if row["unhidden_us"] > 0]
        ranks.append(
            {
                "device": device,
                "layer_count": len(layer_rows),
                "ready_at_demand": len(layer_rows) - len(late),
                "ready_before_compute": sum(row["gather_end_to_compute_us"] >= 0 for row in layer_rows),
                "gather_us_min": min(gather_times),
                "gather_us_median": median(gather_times),
                "gather_us_max": max(gather_times),
                "unhidden_us_median_when_late": median(late) if late else 0,
                "unhidden_us_max": max(late, default=0),
                "layers": layer_rows,
            }
        )
    return {"ranks": ranks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("benchmark_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.benchmark_json.read_text(encoding="utf-8"))
    result = analyze(args.database.resolve(), run["forward_layer_order"], run["world_size"])
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{key: value for key, value in rank.items() if key != "layers"} for rank in result["ranks"]], indent=2))


if __name__ == "__main__":
    main()
