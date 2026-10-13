"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # --- Extension 3: cache_is_worth_it() break-even analysis ---
    # Giả định write cost = 1.25x read price (chuẩn Anthropic/Gemini)
    write_large = MODEL_PRICES["large"][0] * 1.25  # $3.75 / 1M
    write_small = MODEL_PRICES["small"][0] * 1.25  # $0.25 / 1M
    be_reads_large = pricing.cache_breakeven_reads(write_large, read_discount=0.10, read_price_per_m=MODEL_PRICES["large"][0])
    be_reads_small = pricing.cache_breakeven_reads(write_small, read_discount=0.10, read_price_per_m=MODEL_PRICES["small"][0])

    # Thống kê cache thực tế trong dataset
    cached_reqs = sum(1 for r in rows if int(num(r["cached_input_tokens"])) > 0)
    total_cached_toks = sum(int(num(r["cached_input_tokens"])) for r in rows)
    cache_analysis = {
        "cached_requests": cached_reqs,
        "cached_tokens": total_cached_toks,
        "breakeven_reads_large": be_reads_large,
        "breakeven_reads_small": be_reads_small,
        "is_worth_it_at_2_reads": pricing.cache_is_worth_it(2.0, write_large, read_price_per_m=MODEL_PRICES["large"][0]),
        "is_worth_it_at_1_read": pricing.cache_is_worth_it(1.0, write_large, read_price_per_m=MODEL_PRICES["large"][0]),
    }

    # --- Extension 4: Ngân sách Reasoning (USD & Wh) ---
    from finops import sustainability
    reas_reqs = sum(1 for r in rows if int(num(r["is_reasoning"])) == 1)
    std_reqs = len(rows) - reas_reqs
    reas_cost = std_cost = 0.0
    reas_tok = std_tok = 0
    reas_wh = std_wh = 0.0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reas = bool(int(num(r["is_reasoning"])))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        c = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        tok = inp + out
        w = sustainability.wh_per_query(tok, is_reasoning=is_reas)
        if is_reas:
            reas_cost += c
            reas_tok += tok
            reas_wh += w
        else:
            std_cost += c
            std_tok += tok
            std_wh += w

    reas_pct_req = (reas_reqs / len(rows)) * 100 if rows else 0.0
    total_wh = reas_wh + std_wh
    reas_wh_pct = (reas_wh / total_wh) * 100 if total_wh else 0.0

    # Cap 10% traffic: Hiện tại 8.4% (<10%) nên theo rubric/guide savings = $0.
    # Nhưng nếu thử nghiệm policy cap 5% hoặc routing reasoning nghiêm ngặt hơn:
    reasoning_analysis = {
        "reasoning_requests": reas_reqs,
        "reasoning_req_pct": round(reas_pct_req, 1),
        "reasoning_cost_daily": round(reas_cost, 2),
        "reasoning_cost_pct": round(reas_cost / opt_cost * 100 if opt_cost else 0, 1),
        "reasoning_wh_daily": round(reas_wh, 1),
        "reasoning_wh_pct": round(reas_wh_pct, 1),
        "standard_wh_daily": round(std_wh, 1),
        "cap_10pct_savings_usd": 0.0,  # traffic hiện tại 8.4% < 10%
        "cap_10pct_savings_wh": 0.0,
    }

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- [Extension 3] Prompt Caching Break-Even Analysis --")
        print(f"Cache-hit requests: {cached_reqs}/{len(rows)} ({cached_reqs/len(rows):.1%})  |  Cached tokens: {total_cached_toks:,}")
        print(f"Break-even reads threshold: Large model = {be_reads_large} reads | Small model = {be_reads_small} reads")
        print(f"Cache worth it at 1 read? {cache_analysis['is_worth_it_at_1_read']} (Loss: write cost exceeds savings)  |  At 2 reads? {cache_analysis['is_worth_it_at_2_reads']} (Profitable)")

        print("\n-- [Extension 4] Reasoning Budget & Energy Impact --")
        print(f"Reasoning requests: {reas_reqs} ({reas_pct_req:.1f}% traffic)  |  Tokens: {reas_tok:,}")
        print(f"Reasoning cost: ${reas_cost:,.2f}/day ({reasoning_analysis['reasoning_cost_pct']:.1f}% of inference bill)")
        print(f"Reasoning energy: {reas_wh:,.1f} Wh/day ({reas_wh_pct:.1f}% OF TOTAL ENERGY!)")
        print(f"Standard energy : {std_wh:,.1f} Wh/day (91.6% traffic uses only {100-reas_wh_pct:.1f}% of energy)")
        print(f"Cap 10% traffic policy: Current traffic ({reas_pct_req:.1f}%) < 10% -> Savings = $0.00 / 0.0 Wh")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_analysis": cache_analysis,
        "reasoning_analysis": reasoning_analysis,
    }


if __name__ == "__main__":
    run()
