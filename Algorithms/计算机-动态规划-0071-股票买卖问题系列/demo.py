"""Stock trading dynamic-programming series MVP.

Covered variants:
1) At most one transaction
2) Unlimited transactions
3) Unlimited transactions with transaction fee
4) Unlimited transactions with 1-day cooldown
5) At most k transactions
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def validate_prices(prices: np.ndarray | list[float]) -> np.ndarray:
    """Return a finite 1D float array of prices."""
    arr = np.asarray(prices, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"prices must be a 1D array, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("prices contains non-finite values")
    return arr


def max_profit_one_transaction(prices: np.ndarray | list[float]) -> float:
    """Best profit with at most one buy-sell pair."""
    arr = validate_prices(prices)
    if arr.size < 2:
        return 0.0

    min_price = float(arr[0])
    best = 0.0

    for price in arr[1:]:
        p = float(price)
        min_price = min(min_price, p)
        best = max(best, p - min_price)

    return best


def max_profit_unlimited_transactions(prices: np.ndarray | list[float]) -> float:
    """Best profit with unlimited transactions (single share held at a time)."""
    arr = validate_prices(prices)
    if arr.size < 2:
        return 0.0

    cash = 0.0
    hold = -float(arr[0])

    for price in arr[1:]:
        p = float(price)
        prev_cash, prev_hold = cash, hold
        cash = max(prev_cash, prev_hold + p)
        hold = max(prev_hold, prev_cash - p)

    return cash


def max_profit_with_fee(prices: np.ndarray | list[float], fee: float) -> float:
    """Best profit with unlimited transactions and sell-side transaction fee."""
    if fee < 0:
        raise ValueError("fee must be >= 0")

    arr = validate_prices(prices)
    if arr.size < 2:
        return 0.0

    cash = 0.0
    hold = -float(arr[0])

    for price in arr[1:]:
        p = float(price)
        prev_cash, prev_hold = cash, hold
        cash = max(prev_cash, prev_hold + p - fee)
        hold = max(prev_hold, prev_cash - p)

    return cash


def max_profit_with_cooldown(prices: np.ndarray | list[float]) -> float:
    """Best profit with unlimited transactions and 1-day cooldown after selling."""
    arr = validate_prices(prices)
    if arr.size < 2:
        return 0.0

    hold = -float(arr[0])
    sold = -np.inf
    rest = 0.0

    for price in arr[1:]:
        p = float(price)
        prev_hold, prev_sold, prev_rest = hold, sold, rest
        hold = max(prev_hold, prev_rest - p)
        sold = prev_hold + p
        rest = max(prev_rest, prev_sold)

    return float(max(sold, rest))


def max_profit_k_transactions(prices: np.ndarray | list[float], k: int) -> float:
    """Best profit with at most k transactions using O(nk) DP."""
    if k < 0:
        raise ValueError("k must be >= 0")

    arr = validate_prices(prices)
    n = arr.size
    if n < 2 or k == 0:
        return 0.0

    # Degenerates to the unlimited-transactions case.
    if k >= n // 2:
        return max_profit_unlimited_transactions(arr)

    buy = np.full(k + 1, -np.inf, dtype=float)
    sell = np.zeros(k + 1, dtype=float)

    for price in arr:
        p = float(price)
        prev_buy = buy.copy()
        prev_sell = sell.copy()
        for t in range(1, k + 1):
            buy[t] = max(prev_buy[t], prev_sell[t - 1] - p)
            sell[t] = max(prev_sell[t], prev_buy[t] + p)

    return float(np.max(sell))


def run_all_variants(prices: np.ndarray | list[float], fee: float, k: int) -> Dict[str, float]:
    """Run all variants and return their profits."""
    arr = validate_prices(prices)
    return {
        "one_transaction": max_profit_one_transaction(arr),
        "unlimited_transactions": max_profit_unlimited_transactions(arr),
        "with_fee": max_profit_with_fee(arr, fee=fee),
        "with_cooldown": max_profit_with_cooldown(arr),
        "k_transactions": max_profit_k_transactions(arr, k=k),
    }


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    prices = np.array([3.0, 2.0, 6.0, 5.0, 0.0, 3.0, 1.0, 4.0], dtype=float)
    fee = 1.0
    k = 2

    print("=== Input ===")
    print(f"prices = {prices}")
    print(f"fee    = {fee}")
    print(f"k      = {k}")

    profits = run_all_variants(prices, fee=fee, k=k)

    print("\n=== Profits by Variant ===")
    for name, value in profits.items():
        print(f"{name:24s}: {value:.4f}")

    k_large = len(prices) // 2
    k_large_profit = max_profit_k_transactions(prices, k_large)
    unlimited_profit = max_profit_unlimited_transactions(prices)
    consistency_ok = bool(np.isclose(k_large_profit, unlimited_profit))

    print("\n=== Consistency Check ===")
    print(f"k_large ({k_large}) profit   : {k_large_profit:.4f}")
    print(f"unlimited profit           : {unlimited_profit:.4f}")
    print(f"consistency_check_passed   : {consistency_ok}")

    cooldown_case = [1.0, 2.0, 3.0, 0.0, 2.0]
    fee_case = [1.0, 3.0, 2.0, 8.0, 4.0, 9.0]
    print("\n=== Known Cases ===")
    print(
        "cooldown [1,2,3,0,2] ->",
        f"{max_profit_with_cooldown(cooldown_case):.4f} (expected 3.0000)",
    )
    print(
        "fee [1,3,2,8,4,9], fee=2 ->",
        f"{max_profit_with_fee(fee_case, fee=2.0):.4f} (expected 8.0000)",
    )

    try:
        _ = max_profit_k_transactions(prices, k=-1)
    except ValueError as exc:
        print("\nExpected failure on invalid k:")
        print(exc)


if __name__ == "__main__":
    main()
