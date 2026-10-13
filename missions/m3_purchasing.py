"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # --- Extension 1: recommend_tier_matrix (interruption rates & duration policy) ---
    matrix_recs = []
    for j in jobs:
        m = pricing.recommend_tier_matrix(
            hours_per_day=num(j["hours_per_day"]),
            interruptible=bool(int(num(j["interruptible"]))),
            gpu_type=j["gpu_type"],
            job_days=int(num(j["days"])),
        )
        matrix_recs.append({"job_id": j["job_id"], "gpu_type": j["gpu_type"], **m})

    # --- Extension 5: Carbon-aware Scheduling cho interruptible workloads ---
    from finops import sustainability
    tot_kwh = 0.0
    for j in jobs:
        if int(num(j["interruptible"])) == 1:
            hpd = num(j["hours_per_day"])
            days = num(j["days"])
            ngpu = int(num(j["num_gpus"]))
            w = num(cat[j["gpu_type"]]["watts"])
            tot_kwh += (hpd * days * ngpu * w) / 1000.0

    carbon_by_region = {}
    for reg, g_kwh in sustainability.REGION_CARBON.items():
        price_kwh = sustainability.REGION_PRICE_KWH[reg]
        emissions_kg = (tot_kwh * g_kwh) / 1000.0
        p_cost = tot_kwh * price_kwh
        carbon_by_region[reg] = {
            "g_per_kwh": g_kwh,
            "price_per_kwh": price_kwh,
            "emissions_kg_co2": round(emissions_kg, 1),
            "electricity_cost_usd": round(p_cost, 2),
        }

    cleanest_region = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    cheapest_power_region = min(sustainability.REGION_PRICE_KWH, key=sustainability.REGION_PRICE_KWH.get)
    baseline_carbon_kg = carbon_by_region["us-east-1"]["emissions_kg_co2"]
    cleanest_carbon_kg = carbon_by_region[cleanest_region]["emissions_kg_co2"]
    carbon_saved_kg = baseline_carbon_kg - cleanest_carbon_kg
    carbon_saved_pct = (carbon_saved_kg / baseline_carbon_kg) * 100 if baseline_carbon_kg else 0.0

    carbon_scheduling = {
        "interruptible_kwh": round(tot_kwh, 1),
        "cleanest_region": cleanest_region,
        "cheapest_region": cheapest_power_region,
        "carbon_saved_kg": round(carbon_saved_kg, 1),
        "carbon_saved_pct": round(carbon_saved_pct, 1),
        "by_region": carbon_by_region,
    }

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- [Extension 1] Matrix Tier Recommendations (GPU Interruption & Horizon) --")
        print(f"{'job':18}{'gpu':6}{'interr_rate':>13}{'duty%':>8}{'recommended':>15}  {'rationale'}")
        for mr in matrix_recs:
            print(f"{mr['job_id']:18}{mr['gpu_type']:6}{mr['interruption_rate']:>12.1%}{mr['duty_cycle']:>8.0%}{mr['recommended_tier']:>15}  {mr['reason']}")

        print("\n-- [Extension 5] Carbon-Aware Scheduling (Batch & Training Workloads) --")
        print(f"Total energy for interruptible workloads: {tot_kwh:,.1f} kWh")
        print(f"{'Region':17}{'Carbon (g/kWh)':>16}{'Price ($/kWh)':>15}{'CO2e (kg)':>12}{'Power Cost':>12}")
        for reg, d in carbon_by_region.items():
            print(f"{reg:17}{d['g_per_kwh']:>16}{d['price_per_kwh']:>15.3f}{d['emissions_kg_co2']:>12.1f}${d['electricity_cost_usd']:>11.2f}")
        print(f"\nOptimization impact if scheduled in {cleanest_region} (vs us-east-1):")
        print(f"-> Reduces emissions by {carbon_saved_kg:.1f} kg CO2e ({carbon_saved_pct:.1f}% carbon reduction)")
        print(f"-> Cheapest power is in {cheapest_power_region} (${carbon_by_region[cheapest_power_region]['electricity_cost_usd']:.2f})")

    return {
        "recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
        "policy_matrix": matrix_recs,
        "carbon_scheduling": carbon_scheduling,
    }


if __name__ == "__main__":
    run()
