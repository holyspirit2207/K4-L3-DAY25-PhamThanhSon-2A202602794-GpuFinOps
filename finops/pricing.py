"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# Interruption rates by GPU type (snapshot from major cloud providers 2026)
GPU_SPOT_INTERRUPT_RATES: dict[str, float] = {
    "H100": 0.08,   # High demand / contention, ~8% per hour
    "H200": 0.12,   # Very scarce, ~12% per hour
    "A100": 0.05,   # Moderate demand, ~5% per hour
    "A10G": 0.03,   # Commodity GPU, ~3% per hour
    "L4": 0.02,     # Inference commodity, ~2% per hour
    "B200": 0.15,   # Cutting-edge, high preemption
    "MI300X": 0.04,
}


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: int | None = None,
    interruption_rate: float | None = None,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    Enhanced policy (Extension 1):
      - Considers interruption rate by GPU type and checkpoint overhead.
      - Supports job duration (job_days) to distinguish reserved commitments.
      - Maintains 100% backward compatibility when optional parameters are omitted.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)

    if gpu_type is None and job_days is None and interruption_rate is None:
        if interruptible and hours_per_day < 24:
            return "spot"
        if duty >= be:
            return "reserved"
        return "on_demand"

    # Enhanced logic:
    rate = interruption_rate if interruption_rate is not None else GPU_SPOT_INTERRUPT_RATES.get(gpu_type or "", 0.05)
    if interruptible and hours_per_day < 24:
        # If preemption rate is excessive (>20%/hr), checkpoint rework erodes spot savings
        if rate < 0.20:
            return "spot"

    if duty >= be:
        return "reserved"
    return "on_demand"


def recommend_tier_matrix(
    hours_per_day: float,
    interruptible: bool,
    gpu_type: str,
    job_days: int,
    reserved_1yr_discount: float = 0.20,
    reserved_3yr_discount: float = 0.45,
) -> dict:
    """Matrix recommendation considering 1yr vs 3yr commitment duration and interruption rate."""
    duty = max(0.0, hours_per_day) / 24.0
    rate = GPU_SPOT_INTERRUPT_RATES.get(gpu_type, 0.05)
    be_3yr = break_even_utilization(reserved_3yr_discount)
    be_1yr = break_even_utilization(reserved_1yr_discount)

    if interruptible and hours_per_day < 24 and rate < 0.20:
        tier = "spot"
        reason = f"Workload interruptible & spot preemption rate ({rate:.1%}) within acceptable threshold"
    elif duty >= be_3yr and job_days >= 365:
        tier = "reserved_3yr"
        reason = f"High duty cycle ({duty:.0%} >= {be_3yr:.0%}) & long horizon ({job_days}d >= 365d) qualifies for 3yr commitment"
    elif duty >= be_1yr and job_days >= 90:
        tier = "reserved_1yr"
        reason = f"Steady duty cycle ({duty:.0%} >= {be_1yr:.0%}) & medium horizon ({job_days}d) qualifies for 1yr commitment"
    else:
        tier = "on_demand"
        reason = f"Duty cycle ({duty:.0%}) below break-even or short transient duration ({job_days}d)"

    return {
        "recommended_tier": tier,
        "interruption_rate": rate,
        "duty_cycle": round(duty, 3),
        "reason": reason,
    }


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
    read_price_per_m: float = 3.00,
) -> bool:
    """Cache pays off only when total savings from subsequent reads > write/storage cost.

    Breakeven: avg_cache_reads * (read_price_per_m * (1 - read_discount)) >= write_cost_per_m
    Example: With 90% read discount (Anthropic/Gemini), savings per read is 0.9 * read_price.
    If write_cost is 1.25x read_price ($3.75 vs $3.00), break-even requires >= 1.39 reads.
    """
    savings_per_read = read_price_per_m * (1.0 - read_discount)
    if savings_per_read <= 0:
        return False
    return (avg_cache_reads * savings_per_read) >= write_cost_per_m


def cache_breakeven_reads(
    write_cost_per_m: float,
    read_discount: float = 0.10,
    read_price_per_m: float = 3.00,
) -> float:
    """Number of cache reads required to break even against initial write cost."""
    savings_per_read = read_price_per_m * (1.0 - read_discount)
    if savings_per_read <= 0:
        return float("inf")
    return round(write_cost_per_m / savings_per_read, 2)


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
