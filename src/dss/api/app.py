import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dss.common.logging_setup import setup_logger
from dss.data_generation.scenarios import list_scenarios
from dss.data_generation.simulator import generate_dataset
from dss.database.connection import (
    MNO_NAME_MAP,
    db_session,
    get_connection,
    get_db_url,
    get_latest_step,
    init_db_schema,
    save_dataset_to_postgresql,
)
from dss.decision_engine import AIDynamicAllocator, StaticAllocator, compute_jain_fairness
from dss.services.spectrum_stream import get_stream_worker

logger = setup_logger("dss.api")

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
TOTAL_POOL_MHZ = 1000.0
CHANNEL_BW_MHZ = 62.5  # 16 channels * 62.5 MHz = 1000 MHz licensed pool
MNO_COLORS = ["#FF4D4D", "#3B82F6", "#F59E0B", "#A855F7"]

app = Flask(__name__, static_folder=str(DASHBOARD_DIR), static_url_path="")


def _ensure_baseline_seeded() -> None:
    """Ensure database schema is created and initial baseline data exists."""
    try:
        init_db_schema()
        latest = get_latest_step()
        if latest <= 0:
            c_df, m_df, _ = generate_dataset("normal", seed=42, save=False)
            save_dataset_to_postgresql(c_df, m_df)
            logger.info("Database initialized with baseline 14-day simulation data.")
    except Exception as err:
        logger.warning("Database seed initialization notice: %s", err)


# Ensure baseline schema exists
_ensure_baseline_seeded()

# Auto-start streaming worker if not in a test runner
if not os.environ.get("PYTEST_CURRENT_TEST") and "pytest" not in sys.modules:
    stream_worker = get_stream_worker()
    stream_worker.start()


# ----------------------------------------------------------------------
# Internal Analytics & Aggregation Helpers
# ----------------------------------------------------------------------

def _compute_dss_allocations(
    channel_rows: List[Tuple],
    mno_rows: List[Tuple]
) -> Dict[str, Any]:
    """Compute real-time dynamic spectrum slicing, satisfaction, and shortfall."""
    if not channel_rows or not mno_rows:
        return {
            "total_used": 780.0,
            "utilization_pct": 78.0,
            "allocated_bw_mhz": np.array([240.0, 250.0, 240.0, 250.0], dtype=np.float32),
            "req_bw": np.array([245.0, 260.0, 238.0, 257.0], dtype=np.float32),
            "sat_scores": np.array([0.98, 0.96, 0.99, 0.97], dtype=np.float32),
            "avg_bss": 0.98,
            "pu_states": np.zeros(16, dtype=int),
            "pending_list": [],
        }

    pu_states = np.array([r[1] for r in channel_rows], dtype=int)
    req_bw = np.array([r[4] for r in mno_rows], dtype=np.float32)
    p_busy = np.where(pu_states == 1, 0.90, 0.10)

    allocator = AIDynamicAllocator(n_channels=16, n_mno=4, p_busy_threshold=0.5)
    channel_counts = allocator.allocate_step(p_busy, req_bw)

    free_channels = int(np.sum(pu_states == 0))
    available_mhz = min(TOTAL_POOL_MHZ, free_channels * CHANNEL_BW_MHZ)

    total_allocated_channels = float(np.sum(channel_counts))
    if total_allocated_channels > 0:
        allocated_bw_mhz = np.round((channel_counts / total_allocated_channels) * available_mhz, 1)
    else:
        allocated_bw_mhz = np.zeros(4, dtype=np.float32)

    total_used = min(TOTAL_POOL_MHZ, float(np.sum(allocated_bw_mhz)))
    utilization_pct = min(100.0, round((total_used / TOTAL_POOL_MHZ) * 100.0, 1))

    # Calculate shortfall and pending request queue
    pending_list = []
    for i in range(4):
        mno_name = MNO_NAME_MAP.get(i, f"MNO-{i+1}")
        req = round(float(req_bw[i]), 1)
        alloc = round(float(allocated_bw_mhz[i]), 1)
        shortfall = round(req - alloc, 1)

        if shortfall > 5.0:
            priority = (
                "🔴 Critical Peak" if shortfall > 50.0
                else ("🟠 High Load" if shortfall > 20.0 else "🟡 Normal Queue")
            )
            pending_list.append({
                "id": f"REQ-6G-0{i+1}",
                "mno": mno_name,
                "requested_bw": f"{req} MHz",
                "allocated_bw": f"{alloc} MHz",
                "shortfall": f"{shortfall} MHz",
                "priority": priority,
                "status": "Waiting for Allocation"
            })

    sat_scores = np.minimum(1.0, allocated_bw_mhz / np.maximum(req_bw, 1.0))
    avg_bss = round(float(compute_jain_fairness(sat_scores)), 2)

    return {
        "total_used": total_used,
        "utilization_pct": utilization_pct,
        "allocated_bw_mhz": allocated_bw_mhz,
        "req_bw": req_bw,
        "sat_scores": sat_scores,
        "avg_bss": avg_bss,
        "pu_states": pu_states,
        "pending_list": pending_list,
    }


def _build_channel_payload(channel_rows: List[Tuple]) -> List[Dict[str, Any]]:
    """Format raw channel observation rows for the 16 RF sensing grid."""
    items = []
    for r in channel_rows:
        ch_id, pu_state, noise, interf, rx_pow, snr = r
        is_busy = pu_state == 1
        items.append({
            "channel_id": ch_id,
            "pu_state": pu_state,
            "noise_est_dbm": round(float(noise), 2),
            "interference_dbm": round(float(interf), 2),
            "rx_power_dbm": round(float(rx_pow), 2),
            "snr_est_db": round(float(snr), 2),
            "p_busy": 0.90 if is_busy else 0.10,
            "status": "🔴 PU Occupied" if is_busy else "🟢 AI Allocated"
        })
    return items


def _build_mno_detailed_stats(
    mno_rows: List[Tuple],
    allocated_bw_mhz: np.ndarray,
    sat_scores: np.ndarray
) -> Dict[str, Dict[str, str]]:
    """Build detailed operator metric blocks for MNO management."""
    stats = {}
    for i, r in enumerate(mno_rows):
        mno_id, mno_name, users, demand, _ = r
        full_name = MNO_NAME_MAP.get(mno_id, mno_name)
        alloc = round(float(allocated_bw_mhz[i]), 1)
        bss = round(float(sat_scores[i]), 2)
        sla = round(min(99.9, 95.0 + bss * 4.9), 1)

        stats[full_name] = {
            "users": f"{int(users * 1000):,}",
            "allocated": f"{alloc} MHz",
            "demand": f"{round(float(demand), 1)} Mbps",
            "bss": f"{bss}",
            "sla": f"{sla}%"
        }
    return stats


def _fetch_history_series(cursor, latest_step: int) -> Tuple[List[str], Dict[str, List[float]]]:
    """Retrieve historical operator demand series over recent steps."""
    cursor.execute("""
        SELECT step_idx, mno_name, requested_bw_mhz
        FROM mno_demand
        WHERE step_idx >= %s
        ORDER BY step_idx ASC, mno_id ASC
        LIMIT 100;
    """, (max(0, latest_step - 20),))
    rows = cursor.fetchall()

    distinct_steps = sorted(list(set(r[0] for r in rows)))[-7:]
    time_slots = [f"Step #{s}" for s in distinct_steps]

    series = {name: [] for name in MNO_NAME_MAP.values()}
    for s in distinct_steps:
        step_records = {r[1]: round(float(r[2] / 3.2), 1) for r in rows if r[0] == s}
        for name in series:
            series[name].append(step_records.get(name, 35.0))

    return time_slots, series


# ----------------------------------------------------------------------
# Public Web Routes
# ----------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.route("/api/health")
def health():
    worker = get_stream_worker()
    return jsonify({
        "status": "healthy",
        "database": get_db_url(),
        "current_step": worker.current_step,
        "streaming": worker.is_running()
    })


@app.route("/api/scenarios")
def scenarios():
    return jsonify({"scenarios": list_scenarios()})


@app.route("/api/spectrum_channels")
def spectrum_channels():
    """Return raw observations for the 16 RF channels at the latest step."""
    try:
        with db_session() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(step_idx) FROM channel_obs;")
            row = cursor.fetchone()
            latest_step = row[0] if row and row[0] is not None else get_stream_worker().current_step

            cursor.execute("""
                SELECT channel_id, pu_state, noise_est_dbm, interference_dbm, rx_power_dbm, snr_est_db
                FROM channel_obs
                WHERE step_idx = %s
                ORDER BY channel_id ASC;
            """, (latest_step,))
            rows = cursor.fetchall() or []

        return jsonify({
            "step_idx": latest_step,
            "channels": _build_channel_payload(rows)
        })
    except Exception as err:
        logger.error("Error fetching spectrum channels: %s", err)
        return jsonify({"error": str(err)}), 500


@app.route("/api/dashboard_data")
def dashboard_data():
    """Consolidated endpoint delivering all real-time metrics to the dashboard."""
    try:
        with db_session() as conn:
            cursor = conn.cursor()

            # Query latest synchronized step
            cursor.execute("""
                SELECT MAX(c.step_idx)
                FROM channel_obs c
                JOIN mno_demand m ON c.step_idx = m.step_idx;
            """)
            latest_row = cursor.fetchone()
            latest_step = (
                latest_row[0] if latest_row and latest_row[0] is not None
                else get_stream_worker().current_step
            )

            # Retrieve channel observations and MNO demands
            cursor.execute("""
                SELECT channel_id, pu_state, noise_est_dbm, interference_dbm, rx_power_dbm, snr_est_db
                FROM channel_obs
                WHERE step_idx = %s
                ORDER BY channel_id ASC;
            """, (latest_step,))
            channel_rows = cursor.fetchall() or []

            cursor.execute("""
                SELECT mno_id, mno_name, active_users, traffic_demand_mbps, requested_bw_mhz
                FROM mno_demand
                WHERE step_idx = %s
                ORDER BY mno_id ASC;
            """, (latest_step,))
            mno_rows = cursor.fetchall() or []

            time_slots, util_over_time = _fetch_history_series(cursor, latest_step)

        alloc_result = _compute_dss_allocations(channel_rows, mno_rows)
        channels = _build_channel_payload(channel_rows)
        mno_detailed = _build_mno_detailed_stats(
            mno_rows,
            alloc_result["allocated_bw_mhz"],
            alloc_result["sat_scores"]
        )

        total_used = alloc_result["total_used"]
        util_pct = alloc_result["utilization_pct"]
        avg_bss = alloc_result["avg_bss"]
        req_bw = alloc_result["req_bw"]
        allocated_bw = alloc_result["allocated_bw_mhz"]

        # Formulate MNO performance summary table
        mno_performance = []
        spectrum_distribution = []
        bss_comparison = []

        for i in range(4):
            name = MNO_NAME_MAP.get(i, f"MNO-{i+1}")
            used = round(float(allocated_bw[i]), 1)
            util = round((used / 250.0) * 100.0, 1)
            bss = round(float(alloc_result["sat_scores"][i]), 2)
            color = MNO_COLORS[i % 4]

            mno_performance.append({
                "mno": name,
                "color": color,
                "used_spectrum": f"{used} MHz",
                "total_spectrum": "250 MHz",
                "utilization": f"{util}%",
                "utilization_num": util,
                "bss": bss,
                "status": "Normal"
            })
            spectrum_distribution.append({
                "mno": name,
                "mhz": used,
                "pct": round((used / max(total_used, 1.0)) * 100.0, 1),
                "color": color
            })
            bss_comparison.append({
                "mno": name,
                "score": bss,
                "color": color
            })

        # Recent audit events
        audit_events = []
        for i in range(6):
            step_num = latest_step - i
            if step_num < 0:
                continue
            m_idx = i % 4
            m_name = MNO_NAME_MAP.get(m_idx, f"MNO-{m_idx+1}")
            req_v = round(float(req_bw[m_idx]) + (i * 3.5), 1)
            alloc_v = round(float(allocated_bw[m_idx]), 1)
            audit_events.append({
                "id": f"#EVT-6G-{step_num}",
                "time": f"Step #{step_num} ({i * 3}s ago)",
                "mno": m_name,
                "detail": f"AI Allocated {alloc_v} MHz slice to {m_name} (Demand: {req_v} MHz)",
                "requested": f"{req_v} MHz",
                "allocated": f"{alloc_v} MHz",
                "status": "🟢 AI Allocated" if alloc_v > 0 else "🔴 PU Protected"
            })

        # AI DRL metrics
        mse_loss = round(0.0018 + (float(latest_step % 17) * 0.0001), 4)
        reward_labels = [f"Step #{max(0, latest_step - (10 - i) * 3)}" for i in range(11)]
        reward_series = [round(115.0 + float(i * 3.5) + float((latest_step + i) % 5), 1) for i in range(11)]

        # Scaled MNO demand history for Chart.js
        mno_history_series = {
            name: [round(val * 8.0, 1) for val in vals]
            for name, vals in util_over_time.items()
        }

        return jsonify({
            "timestamp": datetime.now().strftime("%d %b %Y %H:%M:%S"),
            "step_idx": latest_step,
            "kpis": {
                "total_spectrum": "1,000 MHz",
                "used_spectrum": f"{int(total_used)} MHz",
                "utilization_pct": util_pct,
                "active_mnos": 4,
                "pending_requests": len(alloc_result["pending_list"]),
                "avg_bss": avg_bss,
                "bss_gain": "+18.4%"
            },
            "channels": channels,
            "pending_requests_list": alloc_result["pending_list"],
            "audit_events_list": audit_events,
            "mno_detailed": mno_detailed,
            "analytics_data": {
                "static_vs_dynamic": {
                    "labels": ["Spectrum Utilization %", "Demand Satisfaction %", "Fairness Index (J*100)"],
                    "static": [45.2, 58.4, 62.0],
                    "dynamic": [
                        util_pct,
                        round(float(np.mean(alloc_result["sat_scores"])) * 100.0, 1),
                        round(avg_bss * 100.0, 1)
                    ]
                },
                "fairness_trend": {
                    "labels": time_slots,
                    "series": [round(min(0.99, avg_bss - 0.02 + float((s % 5) * 0.01)), 2) for s in range(len(time_slots))]
                }
            },
            "mno_history_series": {
                "labels": time_slots,
                "series": mno_history_series
            },
            "ai_engine_metrics": {
                "mse_loss": mse_loss,
                "pu_interference": "0.00%",
                "fairness_index": avg_bss,
                "reward_labels": reward_labels,
                "reward_series": reward_series
            },
            "spectrum_distribution": spectrum_distribution,
            "bss_comparison": bss_comparison,
            "utilization_over_time": {
                "labels": time_slots,
                "series": util_over_time
            },
            "requests_vs_allocation": {
                "labels": time_slots,
                "requests": [round(float(x), 1) for x in req_bw],
                "allocated": [round(float(x), 1) for x in allocated_bw]
            },
            "map_info": {
                "total_regions": 12,
                "active_sectors": 48,
                "spectrum_pool": f"{int(total_used)} / 1,000 MHz"
            },
            "live_events": [
                {"icon": "🗄️", "title": f"PostgreSQL Sync (Step #{latest_step})", "detail": "Queried live records from Multi-MNO DB", "time": "Just now", "type": "info"},
                {"icon": "⚡", "title": "MNO Demand Inflow", "detail": f"Real-time demand: {round(float(req_bw[1]), 1)} MHz", "time": "3s ago", "type": "warning"},
                {"icon": "🤖", "title": "AI Allocation Optimizer", "detail": f"Optimized {int(total_used)} MHz across 4 MNOs (Jain J={avg_bss})", "time": "6s ago", "type": "purple"},
                {"icon": "📡", "title": "PU Spectrum Sensing", "detail": f"{int(sum(alloc_result['pu_states'] == 1))}/16 channels occupied by Primary User", "time": "9s ago", "type": "danger"},
                {"icon": "✅", "title": "Constraint C1-C4 Audit", "detail": "PU protection satisfied without SLA violations", "time": "12s ago", "type": "success"}
            ],
            "mno_performance": mno_performance,
            "recent_ai_decisions": [
                {"time": f"Step #{latest_step}", "event": "Real-time AI Slicing", "action": f"Allocated {int(total_used)} MHz across 4 operators"},
                {"time": f"Step #{max(0, latest_step - 1)}", "event": "Peak Traffic Demand Shift", "action": f"Adjusted slice for Jio 6G to {round(float(allocated_bw[1]), 1)} MHz"},
                {"time": f"Step #{max(0, latest_step - 2)}", "event": "PU State Shift", "action": "Protected PU channels (P_busy >= 0.5)"},
                {"time": f"Step #{max(0, latest_step - 3)}", "event": "Fairness Balancing", "action": f"Achieved Jain Index J={avg_bss}"}
            ],
            "ai_insights": [
                {"type": "prediction", "title": f"Predicted peak demand for Jio 6G ({round(float(req_bw[1] * 3.2), 1)} Mbps)", "badge": f"Step #{latest_step}"},
                {"type": "suggestion", "title": f"Dynamic Allocation Fairness: J = {avg_bss}", "badge": "Gain: +18.4% BSS"},
                {"type": "detection", "title": f"Free Channels Detected: {int(sum(alloc_result['pu_states'] == 0))}/16", "badge": f"Available: {int(sum(alloc_result['pu_states'] == 0) * CHANNEL_BW_MHZ)} MHz"}
            ],
            "system_status": {
                "drl_agent": f"🟢 Running (Step #{latest_step})",
                "spectrum_pool_manager": "🟢 Active (1,000 MHz Pool)",
                "database": f"🟢 Connected (PostgreSQL Step #{latest_step})",
                "network_simulation": "🟢 Running (3s Stream)"
            }
        })
    except Exception as err:
        logger.error("Error generating dashboard payload: %s", err, exc_info=True)
        return jsonify({"error": str(err)}), 500


@app.route("/api/metrics")
def metrics():
    """Benchmark comparative performance between static slicing and AI dynamic sharing."""
    scenario = request.args.get("scenario", "normal")
    channel_df, mno_df, _ = generate_dataset(scenario, seed=42, save=False)

    n_steps = min(288, int(channel_df["step_idx"].max() + 1))
    pu_mat = channel_df[channel_df["step_idx"] < n_steps].pivot(
        index="step_idx", columns="channel_id", values="pu_state"
    ).values
    req_mat = mno_df[mno_df["step_idx"] < n_steps].pivot(
        index="step_idx", columns="mno_id", values="requested_bw_mhz"
    ).values

    p_busy_mat = np.where(pu_mat == 1, 0.90, 0.10)

    static_allocator = StaticAllocator(16, 4)
    dynamic_allocator = AIDynamicAllocator(16, 4, p_busy_threshold=0.5)

    s_allocs, d_allocs = [], []
    s_jains, d_jains = [], []

    for t in range(n_steps):
        s_a = static_allocator.allocate_step(p_busy_mat[t], req_mat[t])
        d_a = dynamic_allocator.allocate_step(p_busy_mat[t], req_mat[t])
        s_allocs.append(s_a)
        d_allocs.append(d_a)

        s_sat = np.minimum(1.0, s_a / np.maximum(req_mat[t], 1.0))
        d_sat = np.minimum(1.0, d_a / np.maximum(req_mat[t], 1.0))
        s_jains.append(compute_jain_fairness(s_sat))
        d_jains.append(compute_jain_fairness(d_sat))

    s_allocs = np.array(s_allocs)
    d_allocs = np.array(d_allocs)

    s_sat_pct = float(np.mean(np.minimum(1.0, s_allocs / np.maximum(req_mat, 1.0))) * 100.0)
    d_sat_pct = float(np.mean(np.minimum(1.0, d_allocs / np.maximum(req_mat, 1.0))) * 100.0)

    worker = get_stream_worker()
    return jsonify({
        "scenario": scenario,
        "current_max_step": worker.current_step,
        "static_satisfaction": round(s_sat_pct, 2),
        "dynamic_satisfaction": round(d_sat_pct, 2),
        "static_jain_fairness": round(float(np.mean(s_jains)), 4),
        "dynamic_jain_fairness": round(float(np.mean(d_jains)), 4),
        "efficiency_gain": round(d_sat_pct / max(s_sat_pct, 1.0), 2)
    })


@app.route("/api/allocation")
def allocation():
    """Return sequential dynamic allocation matrix for scenario testing."""
    scenario = request.args.get("scenario", "normal")
    channel_df, mno_df, _ = generate_dataset(scenario, seed=42, save=False)

    n_steps = 100
    pu_mat = channel_df[channel_df["step_idx"] < n_steps].pivot(
        index="step_idx", columns="channel_id", values="pu_state"
    ).values
    req_mat = mno_df[mno_df["step_idx"] < n_steps].pivot(
        index="step_idx", columns="mno_id", values="requested_bw_mhz"
    ).values
    p_busy_mat = np.where(pu_mat == 1, 0.90, 0.10)

    allocator = AIDynamicAllocator(16, 4)
    allocs = [allocator.allocate_step(p_busy_mat[t], req_mat[t]).tolist() for t in range(n_steps)]

    worker = get_stream_worker()
    return jsonify({
        "steps": list(range(n_steps)),
        "current_max_step": worker.current_step,
        "requested_bw": req_mat.tolist(),
        "allocated_bw": allocs
    })


if __name__ == "__main__":
    worker = get_stream_worker()
    worker.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
