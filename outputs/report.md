# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

## Root Cause Analysis: The GPU-Util Lie

- **Technical Reality:** The `gpu_util_pct` metric from `nvidia-smi` measures time-active clock presence (fraction of time a kernel is on the SM), **not actual computational efficiency** (MFU/MBU).
- **Empirical Observation:** `gpu-h100-4` records `util = 98.2%` while `MFU = 19.4%` and `MBU = 20.7%`. The GPU is stalled waiting for tensor weights from HBM (memory-bound decode) or kernel launch overhead, yet billed at the full on-demand rate ($2.50/hr) with over 80% wasted compute capacity.

## Prioritized Action Plan (Ranked by ROI)

1. **Priority 1 — Purchasing Optimization (Highest ROI, Low Risk):**
   - Monthly Savings: **$10,040/month** (~80.3% of total savings).
   - Action: Migrate all 5 interruptible workloads (`interruptible=1`) to Spot instances with automated checkpointing; secure 3-year Reserved commitments for 24/7 inference services (`duty cycle >= 55%`).
2. **Priority 2 — Inference Architecture (Cascade + Cache + Batch):**
   - Monthly Savings: **$1,212/month** ($40.39/day, reducing inference bill by 82.6%).
   - Action: Route simple prompts to the Small Model ($0.20/1M vs $3.00/1M), activate Prompt Caching for repeated prefixes, and submit non-realtime evaluation queries via Batch API (-50%).
3. **Priority 3 — Right-sizing 'Util-Lie' GPUs:**
   - Monthly Savings: **$655/month**.
   - Action: Downgrade `gpu-h100-4` to `A100-80GB` (saving $0.71/hr) and `gpu-a10g-1` to `L4` (saving $0.20/hr) as memory bandwidth and capacity of target GPUs fully satisfy achieved workload.
4. **Priority 4 — Reclaim Idle GPU Waste:**
   - Monthly Savings: **$600/month** ($20.00/day).
   - Action: Configure auto-stop termination for instance `gpu-h100-5` when idle (<10% util) exceeds 1 hour.

## Sustainability & Carbon-Aware Scheduling

- Standard queries consume **0.24 Wh (~0.091 gCO2e)** in `us-east-1`.
- Reasoning queries (`is_reasoning=1`) consume **~80x more energy** (19.2 Wh/query). Despite accounting for only 8.4% of traffic, reasoning drives **94.0% of total inference energy consumption**.
- Scheduling batch training jobs to the cleanest region (**`europe-north1`** - 30 gCO2/kWh) reduces emissions by **92.1%** (saving 626.1 kg CO2e/month); scheduling to **`us-east-wa`** ($0.055/kWh) reduces power costs by **54.2%**.

_Figures are June-2026 as-of snapshots; re-baseline before acting._