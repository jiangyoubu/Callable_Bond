"""
Bond Basis Model Module

This module provides tools for calculating and analyzing bond basis,
including forward pricing, basis decomposition, and carry analysis.

Bond Basis = Cash Bond Price - Implied Forward Price
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from yield_curve import YieldCurve


@dataclass
class Bond:
    """Represents a fixed income bond with pricing and valuation methods."""
    
    name: str
    maturity: float  # Time to maturity in years
    coupon_rate: float  # Annual coupon rate (decimal, e.g., 0.05 for 5%)
    price: float  # Current market price (as % of par)
    par_value: float = 100.0  # Par/face value
    coupon_frequency: int = 2  # Annual coupon frequency (2 = semi-annual)
    issue_date: Optional[float] = None  # Years since issuance (for accrued interest)
    
    def __post_init__(self):
        """Validate bond data."""
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")
        if not 0 <= self.price <= 150:
            raise ValueError("Price should be between 0 and 150")
        if not 0 <= self.coupon_rate <= 1:
            raise ValueError("Coupon rate should be between 0 and 1")
    
    def get_ytm(self, max_iterations: int = 100, tolerance: float = 1e-8) -> float:
        """
        Calculate Yield to Maturity (YTM) using Newton-Raphson method.
        
        Args:
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
        
        Returns:
            YTM as decimal (e.g., 0.035 for 3.5%)
        """
        def price_error(ytm):
            """Calculate price error for given YTM."""
            theoretical_price = self._calculate_price_from_ytm(ytm)
            return theoretical_price - self.price
        
        def price_derivative(ytm):
            """Calculate derivative of price with respect to YTM."""
            h = 1e-8
            return (price_error(ytm + h) - price_error(ytm - h)) / (2 * h)
        
        # Initial guess: coupon rate
        ytm = self.coupon_rate
        
        for i in range(max_iterations):
            error = price_error(ytm)
            if abs(error) < tolerance:
                return ytm
            
            derivative = price_derivative(ytm)
            if abs(derivative) < 1e-12:
                break
            
            ytm = ytm - error / derivative
            
            # Sanity check
            if ytm < -0.5 or ytm > 1.0:
                return self.coupon_rate  # Return initial guess if diverging
        
        return ytm
    
    def _calculate_price_from_ytm(self, ytm: float) -> float:
        """Calculate bond price given a YTM."""
        coupon_payment = (self.coupon_rate * self.par_value) / self.coupon_frequency
        periods_per_year = self.coupon_frequency
        total_periods = int(self.maturity * periods_per_year)
        periodic_ytm = ytm / periods_per_year
        
        price = 0.0
        for t in range(1, total_periods + 1):
            price += coupon_payment / ((1 + periodic_ytm) ** t)
        
        price += self.par_value / ((1 + periodic_ytm) ** total_periods)
        
        return price
    
    def get_accrued_interest(self, time_since_last_coupon: float) -> float:
        """
        Calculate accrued interest.
        
        Args:
            time_since_last_coupon: Time elapsed since last coupon payment (in coupon periods)
        
        Returns:
            Accrued interest amount
        """
        coupon_payment = (self.coupon_rate * self.par_value) / self.coupon_frequency
        return coupon_payment * time_since_last_coupon
    
    def __repr__(self) -> str:
        return f"Bond(name={self.name}, maturity={self.maturity}Y, coupon={self.coupon_rate*100:.2f}%, price={self.price:.2f})"


@dataclass
class BondBasisAnalysis:
    """Analyzes bond basis: Cash vs Synthetic (Forward) pricing."""
    
    bond: Bond
    yield_curve: YieldCurve
    forward_date: float  # Settlement date for forward contract (in years)
    repo_rate: float  # Repo rate for financing (annual, decimal)
    
    # Cached results
    _forward_price: Optional[float] = field(default=None, init=False, repr=False)
    _basis: Optional[float] = field(default=None, init=False, repr=False)
    _carry: Optional[float] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Validate inputs."""
        if self.forward_date <= 0:
            raise ValueError("Forward date must be positive")
        if self.forward_date >= self.bond.maturity:
            raise ValueError("Forward date must be before bond maturity")
        if not -0.1 <= self.repo_rate <= 0.5:
            raise ValueError("Repo rate should be reasonable")
    
    def calculate_forward_price(self, method: str = "forward_rates") -> float:
        """
        Calculate the implied forward price of the bond.
        
        This represents what the bond "should" trade for on the forward date,
        derived from current yield curve.
        
        Args:
            method: "forward_rates" (default) or "financing"
        
        Returns:
            Forward price as % of par
        """
        if self._forward_price is not None:
            return self._forward_price
        
        if method == "forward_rates":
            self._forward_price = self._forward_price_from_rates()
        elif method == "financing":
            self._forward_price = self._forward_price_from_financing()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self._forward_price
    
    def _forward_price_from_rates(self) -> float:
        """
        Calculate forward price using forward rates extracted from yield curve.
        
        For each cash flow, discount it to forward date using forward rates.
        """
        time_to_forward = self.forward_date
        remaining_maturity = self.bond.maturity - self.forward_date
        
        forward_price = 0.0
        coupon_payment = (self.bond.coupon_rate * self.bond.par_value) / self.bond.coupon_frequency
        time_between_coupons = 1.0 / self.bond.coupon_frequency
        
        # Add all coupon payments from forward date to maturity
        current_time = self.forward_date + time_between_coupons
        
        while current_time < self.bond.maturity + 1e-6:
            # Forward discount factor from forward_date to current_time
            forward_df = self._get_forward_discount_factor(
                self.forward_date, min(current_time, self.bond.maturity)
            )
            forward_price += coupon_payment * forward_df
            current_time += time_between_coupons
        
        # Add principal at maturity
        forward_df_final = self._get_forward_discount_factor(
            self.forward_date, self.bond.maturity
        )
        forward_price += self.bond.par_value * forward_df_final
        
        return forward_price
    
    def _forward_price_from_financing(self) -> float:
        """
        Calculate forward price using spot price + financing costs.
        
        Forward Price ≈ Spot Price × exp(repo_rate × time) - Coupons Received
        """
        spot_price = self.bond.price
        time_to_forward = self.forward_date
        
        # Finance the bond from now to forward date
        financing_cost = spot_price * (np.exp(self.repo_rate * time_to_forward) - 1)
        
        # Add back coupons received (which reduce financing need)
        coupons_received = 0.0
        coupon_payment = (self.bond.coupon_rate * self.bond.par_value) / self.bond.coupon_frequency
        time_between_coupons = 1.0 / self.bond.coupon_frequency
        
        current_coupon_time = time_between_coupons
        while current_coupon_time <= time_to_forward + 1e-6:
            # Re-invest coupons at repo rate
            time_to_invest = time_to_forward - current_coupon_time
            reinvested_coupon = coupon_payment * np.exp(self.repo_rate * time_to_invest)
            coupons_received += reinvested_coupon
            current_coupon_time += time_between_coupons
        
        forward_price = spot_price + financing_cost - coupons_received
        
        return forward_price
    
    def _get_forward_discount_factor(self, start_time: float, end_time: float) -> float:
        """
        Get forward discount factor from start_time to end_time.
        
        Forward DF = DF(end_time) / DF(start_time)
        """
        df_start = self.yield_curve.get_discount_factor(start_time)
        df_end = self.yield_curve.get_discount_factor(end_time)
        
        return df_end / df_start
    
    def calculate_basis(self) -> float:
        """
        Calculate the bond basis (cash vs synthetic).
        
        Basis = Cash Price - Forward Price
        
        Positive basis = Cash bond is cheap relative to forward (buy cash, short forward)
        Negative basis = Cash bond is expensive (short cash, buy forward)
        
        Returns:
            Basis in price terms (% of par)
        """
        if self._basis is not None:
            return self._basis
        
        forward_price = self.calculate_forward_price()
        self._basis = self.bond.price - forward_price
        
        return self._basis
    
    def calculate_basis_bps(self) -> float:
        """
        Calculate basis in basis points.
        
        Returns:
            Basis in basis points (1 bp = 0.01% = 0.0001)
        """
        basis = self.calculate_basis()
        return basis * 100  # Convert from % to bps (approximately)
    
    def calculate_carry(self) -> Dict[str, float]:
        """
        Decompose basis into carry components.
        
        Carry = Coupon Accrual - Financing Cost - Roll-Down Profit + Curve Change
        
        Returns:
            Dictionary with breakdown:
            - coupon_accrual: Coupon earned over forward period
            - financing_cost: Cost to finance the bond
            - roll_down: Profit from yield curve roll (if curve is steep)
            - total_carry: Net carry
        """
        if self._carry is not None:
            return {"total_carry": self._carry}
        
        time_to_forward = self.forward_date
        
        # 1. Coupon Accrual
        coupon_payment = (self.bond.coupon_rate * self.bond.par_value) / self.bond.coupon_frequency
        coupon_accrual = coupon_payment  # Simplified: assume one coupon payment
        
        # 2. Financing Cost
        financing_cost = self.bond.price * self.repo_rate * time_to_forward
        
        # 3. Roll-Down (change in price from curve flattening)
        current_spot_rate = self.yield_curve.get_spot_rate(self.bond.maturity)
        forward_maturity = self.bond.maturity - self.forward_date
        forward_spot_rate = self.yield_curve.get_spot_rate(forward_maturity)
        
        # Simplified roll-down: if curve is steep, rolling down is profitable
        rate_change = current_spot_rate - forward_spot_rate
        roll_down = (self.bond.maturity * rate_change * self.bond.price) / 100  # DV01 × rate change
        
        # Total Carry
        total_carry = coupon_accrual - financing_cost + roll_down
        
        self._carry = total_carry
        
        return {
            "coupon_accrual": coupon_accrual,
            "financing_cost": financing_cost,
            "roll_down": roll_down,
            "total_carry": total_carry,
        }
    
    def print_analysis(self) -> str:
        """Return formatted analysis results."""
        basis = self.calculate_basis()
        basis_bps = self.calculate_basis_bps()
        carry = self.calculate_carry()
        forward_price = self.calculate_forward_price()
        
        output = "\n" + "=" * 80 + "\n"
        output += "BOND BASIS ANALYSIS\n"
        output += "=" * 80 + "\n\n"
        
        output += f"Bond: {self.bond.name}\n"
        output += f"Maturity: {self.bond.maturity:.2f}Y | Coupon: {self.bond.coupon_rate*100:.2f}% | Par: {self.bond.par_value}\n"
        output += f"Current Price: {self.bond.price:.4f}\n\n"
        
        output += f"Forward Date: {self.forward_date:.2f}Y\n"
        output += f"Repo Rate: {self.repo_rate*100:.2f}%\n\n"
        
        output += "-" * 80 + "\n"
        output += "PRICING ANALYSIS\n"
        output += "-" * 80 + "\n"
        output += f"Cash Price (Spot):        {self.bond.price:>10.4f}\n"
        output += f"Forward Price (Implied):  {forward_price:>10.4f}\n"
        output += f"Basis:                    {basis:>10.4f}  ({basis_bps:>8.2f} bps)\n\n"
        
        output += "-" * 80 + "\n"
        output += "CARRY DECOMPOSITION\n"
        output += "-" * 80 + "\n"
        output += f"Coupon Accrual:           {carry['coupon_accrual']:>10.4f}\n"
        output += f"Financing Cost:           {carry['financing_cost']:>10.4f}\n"
        output += f"Roll-Down Profit:         {carry['roll_down']:>10.4f}\n"
        output += f"Total Carry:              {carry['total_carry']:>10.4f}\n\n"
        
        output += "=" * 80 + "\n"
        
        return output


def analyze_bond_basis(bond: Bond, yield_curve: YieldCurve, 
                       forward_date: float, repo_rate: float) -> BondBasisAnalysis:
    """
    Convenience function to analyze bond basis.
    
    Args:
        bond: Bond to analyze
        yield_curve: YieldCurve object
        forward_date: Forward settlement date (years)
        repo_rate: Financing repo rate (annual, decimal)
    
    Returns:
        BondBasisAnalysis object with results
    """
    return BondBasisAnalysis(bond, yield_curve, forward_date, repo_rate)


# Example usage
if __name__ == "__main__":
    print("Bond Basis Model Module")
    print("Use: from bond_basis import Bond, BondBasisAnalysis, analyze_bond_basis")
