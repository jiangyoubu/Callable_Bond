"""
Yield Curve Construction Module

This module provides classes and methods for constructing and managing
zero coupon (spot) yield curves using the bootstrap method.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import newton, brentq


@dataclass
class Instrument:
    """Represents a fixed income instrument used for curve construction."""
    
    maturity: float  # Time to maturity in years
    coupon_rate: float  # Annual coupon rate (0 for zero-coupon)
    price: float  # Clean price (as % of par, e.g., 99.52)
    par_value: float = 100.0  # Par/face value (default 100)
    coupon_frequency: int = 2  # Annual coupon frequency (2 = semi-annual)
    day_count_convention: str = "Act/365"  # Day count convention
    settlement_days: int = 0  # Days until settlement
    
    def __post_init__(self):
        """Validate instrument data."""
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")
        if not 0 <= self.price <= 150:
            raise ValueError("Price should be between 0 and 150")
        if not 0 <= self.coupon_rate <= 1:
            raise ValueError("Coupon rate should be between 0 and 1")
    
    def get_payment_schedule(self) -> List[Tuple[float, float]]:
        """
        Get the payment schedule for this instrument.
        
        Returns:
            List of (time_to_payment, payment_amount) tuples
        """
        payments = []
        coupon_amount = (self.coupon_rate * self.par_value) / self.coupon_frequency
        
        # Add coupon payments
        time_between_coupons = 1.0 / self.coupon_frequency
        current_time = time_between_coupons
        
        while current_time < self.maturity - 1e-6:
            # Regular coupon payment (not at maturity)
            payments.append((current_time, coupon_amount))
            current_time += time_between_coupons
        
        # Add final payment: coupon + principal at maturity
        payments.append((self.maturity, coupon_amount + self.par_value))
        
        return payments


class YieldCurve:
    """
    Zero coupon yield curve constructed via bootstrap method.
    
    Provides methods to:
    - Build curves from market instruments
    - Calculate discount factors, spot rates, and forward rates
    - Interpolate rates at arbitrary maturities
    """
    
    def __init__(self):
        """Initialize an empty yield curve."""
        self.maturities: List[float] = []
        self.discount_factors: List[float] = []
        self.spot_rates: List[float] = []
        self.interpolation_method: str = "cubic_spline"
        self._spline = None
    
    def bootstrap(self, instruments: List[Instrument], day_count: str = "Act/365") -> None:
        """
        Build yield curve using bootstrap method.
        
        Args:
            instruments: List of Instrument objects, sorted by maturity (ascending)
            day_count: Day count convention ('Act/365', 'Act/360', '30/360', 'Act/Act')
        
        Raises:
            ValueError: If instruments are not sorted by maturity
        """
        if not instruments:
            raise ValueError("At least one instrument is required")
        
        # Verify instruments are sorted by maturity
        for i in range(len(instruments) - 1):
            if instruments[i].maturity >= instruments[i + 1].maturity:
                raise ValueError("Instruments must be sorted by maturity (ascending)")
        
        self.maturities = []
        self.discount_factors = []
        self.spot_rates = []
        
        # Bootstrap each instrument sequentially
        for idx, instrument in enumerate(instruments):
            if len(self.maturities) == 0:
                # First instrument: assume it's a zero coupon bond or simple rate
                df = self._bootstrap_first_instrument(instrument)
            else:
                # Subsequent instruments: use previously calculated DFs
                df = self._bootstrap_instrument(instrument)
            
            self.maturities.append(instrument.maturity)
            self.discount_factors.append(df)
            
            # Calculate spot rate from discount factor
            spot_rate = self._df_to_spot_rate(df, instrument.maturity, 
                                             instrument.coupon_frequency)
            self.spot_rates.append(spot_rate)
        
        # Build interpolation function
        self._build_interpolation()
    
    def _bootstrap_first_instrument(self, instrument: Instrument) -> float:
        """
        Bootstrap the first (typically shortest maturity) instrument.
        
        Args:
            instrument: The first Instrument object
        
        Returns:
            Discount factor at instrument's maturity
        """
        T = instrument.maturity
        
        # For zero-coupon bond (coupon_rate = 0)
        if instrument.coupon_rate == 0:
            # Price = Par * DF(T)
            df = instrument.price / instrument.par_value
        else:
            # For coupon-bearing bond, solve: Price = sum of PV of cash flows
            # For first instrument, approximate as zero-coupon
            # Or use Newton-Raphson to solve
            df = self._solve_for_df_first(instrument)
        
        return df
    
    def _bootstrap_instrument(self, instrument: Instrument) -> float:
        """
        Bootstrap a subsequent instrument using previously calculated discount factors.
        
        Args:
            instrument: The Instrument object to bootstrap
        
        Returns:
            Discount factor at instrument's maturity
        """
        T = instrument.maturity
        payments = instrument.get_payment_schedule()
        
        # Calculate PV of all payments except the final one
        pv_known_payments = 0.0
        final_payment_time = None
        final_payment = 0.0
        
        for time, amount in payments:
            if abs(time - T) < 1e-6:  # This is the final payment
                final_payment_time = time
                final_payment = amount
            else:
                # Interpolate discount factor for this payment date
                df = self.get_discount_factor(time)
                pv_known_payments += amount * df
        
        if final_payment_time is None:
            raise ValueError("No final payment found for instrument")
        
        # Solve for DF(T):
        # Price = pv_known_payments + final_payment * DF(T)
        # DF(T) = (Price - pv_known_payments) / final_payment
        
        df_T = (instrument.price - pv_known_payments) / final_payment
        
        if not (0 < df_T <= 1):
            raise ValueError(f"Invalid discount factor: {df_T} for maturity {T}")
        
        return df_T
    
    def _solve_for_df_first(self, instrument: Instrument) -> float:
        """
        Solve for DF of first instrument using Newton-Raphson method.
        
        Args:
            instrument: The Instrument object
        
        Returns:
            Discount factor
        """
        def price_error(df):
            """Calculate difference between theoretical and market price."""
            theoretical_price = self._calculate_bond_price(instrument, df)
            return theoretical_price - instrument.price
        
        def price_derivative(df):
            """Derivative of price with respect to discount factor."""
            h = 1e-8
            return (price_error(df + h) - price_error(df - h)) / (2 * h)
        
        # Initial guess: use coupon rate as starting point
        df_initial = 1.0 / (1 + instrument.coupon_rate)
        
        try:
            df = newton(price_error, df_initial, fprime=price_derivative, maxiter=50)
        except:
            # Fallback to brent method if Newton-Raphson fails
            df = brentq(price_error, 0.01, 1.0)
        
        return df
    
    def _calculate_bond_price(self, instrument: Instrument, df: float) -> float:
        """
        Calculate theoretical bond price given a single discount factor.
        
        This is a simplified calculation assuming constant DF for all periods.
        Used only for the first instrument bootstrapping.
        """
        coupon_amount = (instrument.coupon_rate * instrument.par_value) / instrument.coupon_frequency
        time_between_coupons = 1.0 / instrument.coupon_frequency
        
        price = 0.0
        current_time = time_between_coupons
        
        while current_time <= instrument.maturity + 1e-6:
            price += coupon_amount * (df ** current_time)
            current_time += time_between_coupons
        
        # Final principal payment
        price += instrument.par_value * (df ** instrument.maturity)
        
        return price
    
    def _df_to_spot_rate(self, df: float, maturity: float, frequency: int) -> float:
        """
        Convert discount factor to spot rate.
        
        Args:
            df: Discount factor
            maturity: Time to maturity in years
            frequency: Coupon frequency (used for compounding)
        
        Returns:
            Spot rate (annualized)
        """
        # DF = 1 / (1 + Z)^T for annual compounding
        # Z = (1/DF)^(1/T) - 1
        
        if df <= 0:
            raise ValueError("Discount factor must be positive")
        
        spot_rate = (1.0 / df) ** (1.0 / maturity) - 1.0
        return spot_rate
    
    def _build_interpolation(self) -> None:
        """Build interpolation function for arbitrary maturities."""
        if len(self.maturities) < 2:
            # Only one point - cannot build spline yet, skip
            return
        
        if self.interpolation_method == "cubic_spline":
            # Cubic spline interpolation on log(DF) for smoothness
            self._spline = CubicSpline(self.maturities, np.log(self.discount_factors))
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")
    
    def get_discount_factor(self, maturity: float, allow_extrapolation: bool = True) -> float:
        """
        Get discount factor at arbitrary maturity via interpolation.
        
        Args:
            maturity: Time to maturity in years
            allow_extrapolation: If True, allows extrapolation beyond curve bounds
                                 using flat extrapolation. Used during bootstrap.
        
        Returns:
            Discount factor at this maturity
        """
        # Handle exact match
        if len(self.maturities) > 0:
            for i, mat in enumerate(self.maturities):
                if abs(mat - maturity) < 1e-10:
                    return self.discount_factors[i]
        
        if len(self.maturities) == 1:
            # Only one point available - use flat extrapolation
            if allow_extrapolation:
                return self.discount_factors[0]
            else:
                raise ValueError(f"Maturity {maturity} outside curve range "
                               f"[{self.maturities[0]}, {self.maturities[0]}]")
        
        if len(self.maturities) == 0:
            raise ValueError("Yield curve is empty")
        
        # Build spline if not already built
        if self._spline is None:
            self._build_interpolation()
        
        # If still no spline (only 1 point), use linear interpolation with the one point
        if self._spline is None:
            if allow_extrapolation:
                # Use flat extrapolation with the single point
                return self.discount_factors[0]
            else:
                raise ValueError(f"Maturity {maturity} outside curve range "
                               f"[{min(self.maturities)}, {max(self.maturities)}]")
        
        # Handle out-of-bounds maturity with spline available
        if maturity < min(self.maturities):
            if not allow_extrapolation:
                raise ValueError(f"Maturity {maturity} outside curve range "
                               f"[{min(self.maturities)}, {max(self.maturities)}]")
            # Flat extrapolation: use first point
            return self.discount_factors[0]
        
        if maturity > max(self.maturities):
            if not allow_extrapolation:
                raise ValueError(f"Maturity {maturity} outside curve range "
                               f"[{min(self.maturities)}, {max(self.maturities)}]")
            # Flat extrapolation: use last point's rate extended
            # This assumes a flat forward rate beyond the last point
            last_maturity = self.maturities[-1]
            last_df = self.discount_factors[-1]
            last_spot_rate = self.spot_rates[-1]
            
            # Extend using last spot rate: DF(t) = DF(T) * [1/(1+r)]^(t-T)
            periods_beyond = maturity - last_maturity
            df = last_df * ((1.0 + last_spot_rate) ** (-periods_beyond))
            return df
        
        # Normal interpolation within bounds
        log_df = self._spline(maturity)
        return np.exp(log_df)
    
    def get_spot_rate(self, maturity: float, frequency: int = 2) -> float:
        """
        Get spot rate at arbitrary maturity.
        
        Args:
            maturity: Time to maturity in years
            frequency: Coupon frequency for rate calculation
        
        Returns:
            Spot rate (annualized, semi-annual compounding by default)
        """
        df = self.get_discount_factor(maturity)
        spot_rate = self._df_to_spot_rate(df, maturity, frequency)
        return spot_rate
    
    def get_forward_rate(self, time_start: float, time_end: float) -> float:
        """
        Calculate forward rate between two future dates.
        
        f(t1, t2) = [(1 + Z(t2))^t2 / (1 + Z(t1))^t1]^(1/(t2-t1)) - 1
        
        Or equivalently:
        f(t1, t2) = [DF(t1) / DF(t2)]^(1/(t2-t1)) - 1
        
        Args:
            time_start: Start time of forward period (years)
            time_end: End time of forward period (years)
        
        Returns:
            Forward rate for period (time_start, time_end)
        """
        if time_start >= time_end:
            raise ValueError("time_start must be less than time_end")
        
        df_start = self.get_discount_factor(time_start)
        df_end = self.get_discount_factor(time_end)
        
        period = time_end - time_start
        forward_rate = (df_start / df_end) ** (1.0 / period) - 1.0
        
        return forward_rate
    
    def get_annuity_factor(self, start_time: float, end_time: float, 
                          frequency: int = 2) -> float:
        """
        Calculate annuity factor (sum of discount factors × year fractions).
        
        Used in swap pricing and bond valuation.
        
        Args:
            start_time: Start time of annuity period
            end_time: End time of annuity period
            frequency: Payment frequency per year
        
        Returns:
            Annuity factor
        """
        time_between_payments = 1.0 / frequency
        annuity_factor = 0.0
        current_time = start_time + time_between_payments
        
        while current_time <= end_time + 1e-6:
            df = self.get_discount_factor(min(current_time, end_time))
            year_fraction = time_between_payments
            annuity_factor += df * year_fraction
            current_time += time_between_payments
        
        return annuity_factor
    
    def print_curve(self) -> str:
        """Return a formatted string representation of the curve."""
        output = "Yield Curve (Bootstrap)\n"
        output += "=" * 70 + "\n"
        output += f"{'Maturity (Y)':>12} | {'Discount Factor':>18} | {'Spot Rate (%)':>15}\n"
        output += "-" * 70 + "\n"
        
        for mat, df, zr in zip(self.maturities, self.discount_factors, self.spot_rates):
            output += f"{mat:>12.4f} | {df:>18.8f} | {zr*100:>15.6f}\n"
        
        output += "=" * 70 + "\n"
        return output
    
    def __repr__(self) -> str:
        return f"YieldCurve(maturities={len(self.maturities)} points)"


# Example usage and testing
if __name__ == "__main__":
    # Create sample instruments
    instruments = [
        Instrument(maturity=0.25, coupon_rate=0.0, price=98.75, coupon_frequency=4),
        Instrument(maturity=0.5, coupon_rate=0.0, price=97.48, coupon_frequency=4),
        Instrument(maturity=1.0, coupon_rate=0.05, price=99.52, coupon_frequency=2),
        Instrument(maturity=2.0, coupon_rate=0.055, price=100.00, coupon_frequency=2),
        Instrument(maturity=5.0, coupon_rate=0.06, price=100.00, coupon_frequency=2),
    ]
    
    # Build curve
    curve = YieldCurve()
    curve.bootstrap(instruments)
    
    print(curve.print_curve())
    
    # Test interpolation
    print("\nInterpolation Test:")
    test_maturities = [0.75, 1.5, 3.0, 4.5]
    for mat in test_maturities:
        df = curve.get_discount_factor(mat)
        zr = curve.get_spot_rate(mat)
        print(f"Maturity {mat:>4.2f}Y: DF={df:.8f}, Spot Rate={zr*100:.6f}%")
    
    # Test forward rates
    print("\nForward Rate Test:")
    forward = curve.get_forward_rate(1.0, 2.0)
    print(f"1Y-2Y Forward Rate: {forward*100:.6f}%")
    
    forward = curve.get_forward_rate(2.0, 5.0)
    print(f"2Y-5Y Forward Rate: {forward*100:.6f}%")
