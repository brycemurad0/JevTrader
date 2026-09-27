"""Fee models. Every fill in backtest, paper and live accounting goes through one of these.

Defaults are deliberately CONSERVATIVE (they over-estimate cost rather than under-estimate).
Regulatory rates change; verify current values at https://alpaca.markets/disclosures and
https://docs.alpaca.markets/docs/crypto-fees and override them in config/fees.yaml.

Spread and slippage are NOT fees: they are modelled by the fill simulator. A strategy's real
cost per round trip = 2 x fee + spread crossed + slippage + adverse selection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from jevtrader.core.types import AssetClass, Instrument, Liquidity, Side


class FeeModel(ABC):
    @abstractmethod
    def fee(self, instrument: Instrument, side: Side, qty: float, price: float, liquidity: Liquidity) -> float:
        """Fee in USD (>= 0) for a single fill."""

    def round_trip_bps(self, instrument: Instrument, price: float, qty: float, maker: bool = False) -> float:
        """Fee cost of buy+sell of `qty` at `price`, in basis points of notional."""
        liq = Liquidity.MAKER if maker else Liquidity.TAKER
        notional = price * qty
        if notional <= 0:
            return 0.0
        total = self.fee(instrument, Side.BUY, qty, price, liq) + self.fee(instrument, Side.SELL, qty, price, liq)
        return 1e4 * total / notional


class ZeroFees(FeeModel):
    def fee(self, instrument, side, qty, price, liquidity) -> float:
        return 0.0


@dataclass
class BpsFees(FeeModel):
    maker_bps: float = 0.0
    taker_bps: float = 0.0

    def fee(self, instrument, side, qty, price, liquidity) -> float:
        bps = self.maker_bps if liquidity is Liquidity.MAKER else self.taker_bps
        return abs(qty * price) * bps / 1e4


@dataclass
class AlpacaEquityFees(FeeModel):
    """Alpaca US equities: $0 commission, but regulatory fees on SELLS.

    - SEC Section 31 fee: `sec_fee_per_million` USD per $1M of sell notional.
    - FINRA TAF: `taf_per_share` per share sold, capped at `taf_max` per trade.
    Both are passed through by Alpaca. Defaults err high.
    """

    commission_per_share: float = 0.0
    sec_fee_per_million: float = 27.80
    taf_per_share: float = 0.000195
    taf_max: float = 9.79

    def fee(self, instrument, side, qty, price, liquidity) -> float:
        f = self.commission_per_share * qty
        if side is Side.SELL:
            f += abs(qty * price) * self.sec_fee_per_million / 1e6
            f += min(self.taf_per_share * qty, self.taf_max)
        return round(f, 6)


@dataclass
class AlpacaCryptoFees(FeeModel):
    """Alpaca crypto: tiered maker/taker by 30-day volume. Tier 1 (< $100k) is 15/25 bps.

    `tiers` = list of (min_30d_volume_usd, maker_bps, taker_bps), ascending.
    """

    thirty_day_volume: float = 0.0
    tiers: list[tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0, 15.0, 25.0),
            (100_000, 12.0, 22.0),
            (500_000, 10.0, 20.0),
            (1_000_000, 8.0, 18.0),
            (10_000_000, 5.0, 15.0),
            (25_000_000, 2.0, 13.0),
            (50_000_000, 2.0, 12.0),
            (100_000_000, 0.0, 10.0),
        ]
    )

    def current_tier(self) -> tuple[float, float, float]:
        tier = self.tiers[0]
        for t in self.tiers:
            if self.thirty_day_volume >= t[0]:
                tier = t
        return tier

    def fee(self, instrument, side, qty, price, liquidity) -> float:
        _, maker, taker = self.current_tier()
        bps = maker if liquidity is Liquidity.MAKER else taker
        return abs(qty * price) * bps / 1e4


@dataclass
class CompositeFees(FeeModel):
    """Dispatch by asset class (the usual choice for a mixed stock + crypto book)."""

    equity: FeeModel = field(default_factory=AlpacaEquityFees)
    crypto: FeeModel = field(default_factory=AlpacaCryptoFees)

    def fee(self, instrument, side, qty, price, liquidity) -> float:
        model = self.crypto if instrument.asset_class is AssetClass.CRYPTO else self.equity
        return model.fee(instrument, side, qty, price, liquidity)


def default_fees() -> FeeModel:
    return CompositeFees()
