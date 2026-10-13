import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finops import pricing, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing


def test_cache_is_worth_it_and_breakeven():
    # Break-even threshold test
    # Write cost = $3.75, Read price = $3.00, discount = 10% (savings = $2.70/read)
    # Threshold = 3.75 / 2.70 = 1.3888 -> 1.39 reads
    be = pricing.cache_breakeven_reads(3.75, read_discount=0.10, read_price_per_m=3.00)
    assert abs(be - 1.39) < 0.02
    assert not pricing.cache_is_worth_it(1.0, 3.75, read_discount=0.10, read_price_per_m=3.00)
    assert pricing.cache_is_worth_it(2.0, 3.75, read_discount=0.10, read_price_per_m=3.00)


def test_recommend_tier_matrix_and_extensions():
    # 24h duty, 365 days -> reserved_3yr
    m_res = pricing.recommend_tier_matrix(24, False, "A100", 365)
    assert m_res["recommended_tier"] == "reserved_3yr"

    # Interruptible, 20h/d -> spot
    m_spot = pricing.recommend_tier_matrix(20, True, "H100", 14)
    assert m_spot["recommended_tier"] == "spot"
    assert m_spot["interruption_rate"] == 0.08


def test_mbu_rightsizing_extension():
    res = m1_efficiency_audit.run(verbose=False)
    assert "mbu_rightsizing" in res
    assert len(res["mbu_rightsizing"]) >= 1
    # Verify gpu-h100-4 is suggested for right-sizing to A100 or cheaper
    rec_ids = [r["gpu_id"] for r in res["mbu_rightsizing"]]
    assert "gpu-h100-4" in rec_ids


def test_carbon_scheduling_extension():
    res = m3_purchasing.run(verbose=False)
    assert "carbon_scheduling" in res
    cs = res["carbon_scheduling"]
    assert cs["cleanest_region"] == "europe-north1"
    assert cs["carbon_saved_pct"] > 70.0  # >70% carbon saved
