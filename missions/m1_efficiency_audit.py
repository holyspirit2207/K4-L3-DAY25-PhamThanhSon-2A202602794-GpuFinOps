"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Extension 2: Right-sizing theo MBU & $/GB-VRAM ---
    rightsizing_recs = []
    for lie in lies:
        cur_type = lie["gpu_type"]
        cur_cat = cat[cur_type]
        cur_od = num(cur_cat["on_demand_hr"])
        cur_bw = num(cur_cat["peak_bw_tbs"])
        cur_vram = num(cur_cat["hbm_gb"])
        achieved_bw = lie["mbu"] * cur_bw

        # Tìm GPU trong catalog có đủ VRAM và đủ bandwidth nhưng giá/giờ thấp hơn
        candidates = []
        for cand_type, cand_cat in cat.items():
            cand_od = num(cand_cat["on_demand_hr"])
            cand_bw = num(cand_cat["peak_bw_tbs"])
            cand_vram = num(cand_cat["hbm_gb"])
            if cand_od < cur_od and cand_vram >= cur_vram * 0.9 and cand_bw >= achieved_bw * 1.2:
                cost_gb = cand_od / cand_vram if cand_vram else 0
                candidates.append({
                    "gpu_type": cand_type,
                    "on_demand_hr": cand_od,
                    "peak_bw_tbs": cand_bw,
                    "hbm_gb": cand_vram,
                    "cost_per_gb_vram": round(cost_gb, 4),
                    "expected_mbu": round(achieved_bw / cand_bw, 3) if cand_bw else 0,
                    "savings_hr": cur_od - cand_od,
                    "savings_monthly": round((cur_od - cand_od) * 24 * 30, 2),
                })

        candidates.sort(key=lambda x: x["on_demand_hr"])
        best = candidates[0] if candidates else None
        if best:
            rightsizing_recs.append({
                "gpu_id": lie["gpu_id"],
                "current_type": cur_type,
                "current_od": cur_od,
                "current_mbu": lie["mbu"],
                "achieved_bw_tbs": round(achieved_bw, 3),
                "proposed_type": best["gpu_type"],
                "proposed_od": best["on_demand_hr"],
                "expected_mbu": best["expected_mbu"],
                "monthly_savings": best["savings_monthly"],
                "reason": f"Sufficient {best['hbm_gb']}GB VRAM, peak BW {best['peak_bw_tbs']} TB/s satisfies {achieved_bw:.2f} TB/s; expected MBU increases to {best['expected_mbu']:.1%}",
            })

    total_rightsize_savings = sum(r["monthly_savings"] for r in rightsizing_recs)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n-- [Extension 2] MBU Right-sizing Recommendations --")
        print(f"{'GPU':12}{'Current':9}{'Proposed':10}{'Achieved BW':>13}{'Expected MBU':>14}{'Monthly Savings':>17}")
        for r in rightsizing_recs:
            print(f"{r['gpu_id']:12}{r['current_type']:9}{r['proposed_type']:10}{r['achieved_bw_tbs']:>11.2f} TB/s{r['expected_mbu']:>13.1%}${r['monthly_savings']:>16,.2f}")
        print(f"Total potential right-sizing savings: ${total_rightsize_savings:,.2f}/month")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "mbu_rightsizing": rightsizing_recs,
        "rightsize_monthly_savings": round(total_rightsize_savings, 2),
    }


if __name__ == "__main__":
    run()
