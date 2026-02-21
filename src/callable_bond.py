"""
Callable Bond Pricing Model

This module provides classes and methods for pricing callable bonds using:
- Interest rate SDE (Vasicek model) for bond valuation
- Hazard rate SDE for credit risk modeling
- Monte Carlo simulation for option pricing
- Tree-based methods for early exercise decisions

Callable Bond Price = Straight Bond Price - Embedded Call Option Price
"""

from typing import List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import warnings


class RateModelType(Enum):
    """Enumeration for interest rate models."""
    VASICEK = "vasicek"
    CIR = "cir"


class CallableType(Enum):
    """Enumeration for callable bond types."""
    AMERICAN = "american"
    EUROPEAN = "european"
    BERMUDA = "bermuda"


@dataclass
class VasicekParams:
    """Parameters for Vasicek interest rate model."""
    mean_reversion: float  # κ (kappa) - mean reversion speed
    long_term_mean: float  # θ (theta) - long-term mean rate
    volatility: float  # σ (sigma) - volatility of short rate
    initial_rate: float  # r0 - initial short rate
    
    def __post_init__(self):
        """Validate parameters."""
        if self.mean_reversion <= 0:
            raise ValueError("Mean reversion speed must be positive")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive")
        if self.initial_rate < 0:
            raise ValueError("Initial rate must be non-negative")


@dataclass
class HazardRateParams:
    """Parameters for hazard rate (credit spread) SDE."""
    mean_reversion: float  # κ_h - mean reversion speed
    long_term_mean: float  # λ_0 - long-term hazard rate
    volatility: float  # σ_h - volatility of hazard rate
    initial_hazard: float  # h0 - initial hazard rate
    recovery_rate: float = 0.4  # Recovery rate in default
    
    def __post_init__(self):
        """Validate parameters."""
        if self.mean_reversion <= 0:
            raise ValueError("Mean reversion speed must be positive")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive")
        if self.initial_hazard < 0:
            raise ValueError("Initial hazard rate must be non-negative")
        if not 0 <= self.recovery_rate <= 1:
            raise ValueError("Recovery rate must be between 0 and 1")


@dataclass
class CallableBondParameters:
    """Parameters for callable bond specification."""
    par_value: float  # Face value
    coupon_rate: float  # Annual coupon rate
    maturity: float  # Time to maturity in years
    coupon_frequency: int  # Coupon payments per year (2 = semi-annual)
    call_price: float  # Strike price for call option
    first_call_date: float  # Earliest call date in years
    last_call_date: Optional[float] = None  # Latest call date (None = maturity)
    callable_type: CallableType = CallableType.AMERICAN
    
    def __post_init__(self):
        """Validate parameters."""
        if self.par_value <= 0:
            raise ValueError("Par value must be positive")
        if not 0 <= self.coupon_rate <= 1:
            raise ValueError("Coupon rate must be between 0 and 1")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")
        if self.coupon_frequency <= 0:
            raise ValueError("Coupon frequency must be positive")
        if self.call_price <= 0:
            raise ValueError("Call price must be positive")
        if self.first_call_date < 0 or self.first_call_date > self.maturity:
            raise ValueError("First call date must be between 0 and maturity")
        if self.last_call_date is None:
            self.last_call_date = self.maturity
        elif self.last_call_date > self.maturity:
            raise ValueError("Last call date cannot exceed maturity")


class InterestRateModel(ABC):
    """Abstract base class for interest rate models."""
    
    @abstractmethod
    def simulate_path(self, t_steps: int, num_paths: int, 
                     seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate interest rate paths.
        
        Args:
            t_steps: Number of time steps
            num_paths: Number of simulation paths
            seed: Random seed for reproducibility
        
        Returns:
            Array of shape (t_steps + 1, num_paths) with interest rate paths
        """
        pass
    
    @abstractmethod
    def get_discount_factor(self, rate_path: np.ndarray, dt: float) -> float:
        """
        Calculate discount factor from a rate path.
        
        Args:
            rate_path: Array of interest rates over time
            dt: Time step
        
        Returns:
            Discount factor from t=0 to t=final
        """
        pass


class VasicekModel(InterestRateModel):
    """
    Vasicek interest rate model.
    
    dr_t = κ(θ - r_t)dt + σ dW_t
    """
    
    def __init__(self, params: VasicekParams):
        """
        Initialize Vasicek model.
        
        Args:
            params: VasicekParams dataclass with model parameters
        """
        self.params = params
    
    def simulate_path(self, t_steps: int, num_paths: int, 
                     seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate Vasicek interest rate paths using Euler scheme.
        
        Args:
            t_steps: Number of time steps
            num_paths: Number of simulation paths
            seed: Random seed for reproducibility
        
        Returns:
            Array of shape (t_steps + 1, num_paths) with interest rate paths
        """
        if seed is not None:
            np.random.seed(seed)
        
        rates = np.zeros((t_steps + 1, num_paths))
        rates[0, :] = self.params.initial_rate
        
        dt = 1.0 / t_steps  # Assuming unit time interval
        
        kappa = self.params.mean_reversion
        theta = self.params.long_term_mean
        sigma = self.params.volatility
        
        for i in range(t_steps):
            dW = np.random.normal(0, np.sqrt(dt), num_paths)
            rates[i + 1, :] = (rates[i, :] + 
                             kappa * (theta - rates[i, :]) * dt + 
                             sigma * dW)
            # Floor rates at 0 to avoid negative rates (optional)
            rates[i + 1, :] = np.maximum(rates[i + 1, :], 0)
        
        return rates
    
    def get_discount_factor(self, rate_path: np.ndarray, dt: float) -> float:
        """
        Calculate discount factor using numerical integration.
        
        Discount Factor = exp(-∫_0^T r_t dt)
        
        Args:
            rate_path: Array of interest rates over time
            dt: Time step
        
        Returns:
            Discount factor
        """
        integral = np.sum(rate_path[:-1]) * dt  # Trapezoidal approximation
        return np.exp(-integral)


class CIRModel(InterestRateModel):
    """
    Cox-Ingersoll-Ross (CIR) interest rate model.
    
    dr_t = κ(θ - r_t)dt + σ√r_t dW_t
    """
    
    def __init__(self, params: VasicekParams):
        """
        Initialize CIR model.
        
        Args:
            params: VasicekParams dataclass (same parameters as Vasicek)
        """
        self.params = params
        # CIR has a positivity constraint
        if 2 * params.mean_reversion * params.long_term_mean < params.volatility ** 2:
            warnings.warn("CIR model parameters violate Feller condition for "
                         "positivity. Rates may become negative.")
    
    def simulate_path(self, t_steps: int, num_paths: int, 
                     seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate CIR interest rate paths using Milstein scheme.
        
        Args:
            t_steps: Number of time steps
            num_paths: Number of simulation paths
            seed: Random seed for reproducibility
        
        Returns:
            Array of shape (t_steps + 1, num_paths) with interest rate paths
        """
        if seed is not None:
            np.random.seed(seed)
        
        rates = np.zeros((t_steps + 1, num_paths))
        rates[0, :] = self.params.initial_rate
        
        dt = 1.0 / t_steps
        kappa = self.params.mean_reversion
        theta = self.params.long_term_mean
        sigma = self.params.volatility
        
        for i in range(t_steps):
            dW = np.random.normal(0, np.sqrt(dt), num_paths)
            sqrt_r = np.sqrt(np.maximum(rates[i, :], 0))
            
            # Milstein scheme for CIR
            rates[i + 1, :] = (rates[i, :] + 
                             kappa * (theta - rates[i, :]) * dt + 
                             sigma * sqrt_r * dW +
                             0.25 * sigma ** 2 * (dW ** 2 - dt))
            
            # Ensure non-negativity
            rates[i + 1, :] = np.maximum(rates[i + 1, :], 0)
        
        return rates
    
    def get_discount_factor(self, rate_path: np.ndarray, dt: float) -> float:
        """
        Calculate discount factor using numerical integration.
        
        Args:
            rate_path: Array of interest rates over time
            dt: Time step
        
        Returns:
            Discount factor
        """
        integral = np.sum(rate_path[:-1]) * dt
        return np.exp(-integral)


class HazardRateModel:
    """
    Hazard rate (credit spread) SDE model.
    
    dh_t = κ_h(λ_0 - h_t)dt + σ_h dW_t^h
    """
    
    def __init__(self, params: HazardRateParams):
        """
        Initialize hazard rate model.
        
        Args:
            params: HazardRateParams dataclass with model parameters
        """
        self.params = params
    
    def simulate_path(self, t_steps: int, num_paths: int, 
                     seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate hazard rate paths using Euler scheme.
        
        Args:
            t_steps: Number of time steps
            num_paths: Number of simulation paths
            seed: Random seed for reproducibility
        
        Returns:
            Array of shape (t_steps + 1, num_paths) with hazard rate paths
        """
        if seed is not None:
            np.random.seed(seed)
        
        hazard_rates = np.zeros((t_steps + 1, num_paths))
        hazard_rates[0, :] = self.params.initial_hazard
        
        dt = 1.0 / t_steps
        kappa_h = self.params.mean_reversion
        lambda_0 = self.params.long_term_mean
        sigma_h = self.params.volatility
        
        for i in range(t_steps):
            dW = np.random.normal(0, np.sqrt(dt), num_paths)
            hazard_rates[i + 1, :] = (hazard_rates[i, :] + 
                                     kappa_h * (lambda_0 - hazard_rates[i, :]) * dt + 
                                     sigma_h * dW)
            # Floor hazard rates at 0
            hazard_rates[i + 1, :] = np.maximum(hazard_rates[i + 1, :], 0)
        
        return hazard_rates
    
    def survival_probability(self, hazard_path: np.ndarray, dt: float) -> float:
        """
        Calculate survival probability from hazard rate path.
        
        Survival Prob = exp(-∫_0^T h_t dt)
        
        Args:
            hazard_path: Array of hazard rates over time
            dt: Time step
        
        Returns:
            Survival probability from t=0 to t=final
        """
        integral = np.sum(hazard_path[:-1]) * dt
        return np.exp(-integral)


class StraightBondPricer:
    """Pricer for straight (non-callable) bonds."""
    
    def __init__(self, bond_params: CallableBondParameters,
                 ir_model: InterestRateModel):
        """
        Initialize straight bond pricer.
        
        Args:
            bond_params: Bond specifications
            ir_model: Interest rate model
        """
        self.bond_params = bond_params
        self.ir_model = ir_model
    
    def calculate_coupon_payment(self) -> float:
        """Calculate semi-annual coupon payment."""
        coupon_payment = (self.bond_params.coupon_rate * 
                         self.bond_params.par_value / 
                         self.bond_params.coupon_frequency)
        return coupon_payment
    
    def get_coupon_dates(self) -> np.ndarray:
        """Get all coupon payment dates."""
        time_between_coupons = 1.0 / self.bond_params.coupon_frequency
        coupon_dates = np.arange(time_between_coupons, 
                               self.bond_params.maturity + 1e-6, 
                               time_between_coupons)
        # Ensure maturity is included
        coupon_dates = np.append(coupon_dates, self.bond_params.maturity)
        coupon_dates = np.unique(coupon_dates)
        return coupon_dates
    
    def price_bond(self, rate_path: np.ndarray, hazard_path: Optional[np.ndarray] = None,
                  t_steps: int = 252) -> float:
        """
        Price straight bond from a rate path.
        
        Bond Price = Σ [Coupon * DF(t_i)] + Par * DF(T)
                    × Survival Probability (if hazard rate provided)
        
        Args:
            rate_path: Interest rate path from simulation
            hazard_path: Hazard rate path (optional for credit risk)
            t_steps: Number of time steps
        
        Returns:
            Bond price
        """
        dt = self.bond_params.maturity / t_steps
        coupon_payment = self.calculate_coupon_payment()
        coupon_dates = self.get_coupon_dates()
        
        bond_price = 0.0
        
        # Price each cash flow
        for coupon_date in coupon_dates:
            # Get discount factor for this date
            coupon_idx = int(np.round(coupon_date / dt))
            coupon_idx = min(coupon_idx, len(rate_path) - 1)
            
            # Discount the coupon (or coupon + principal at maturity)
            rate_integral = np.sum(rate_path[:coupon_idx + 1]) * dt
            df = np.exp(-rate_integral)
            
            if abs(coupon_date - self.bond_params.maturity) < 1e-6:
                # Final payment: coupon + principal
                bond_price += (coupon_payment + self.bond_params.par_value) * df
            else:
                # Regular coupon payment
                bond_price += coupon_payment * df
        
        # Apply credit risk (survival probability) if hazard rates provided
        if hazard_path is not None:
            hazard_integral = np.sum(hazard_path[:-1]) * dt
            survival_prob = np.exp(-hazard_integral)
            bond_price *= survival_prob
        
        return bond_price
    
    def monte_carlo_price(self, num_paths: int = 1000, t_steps: int = 252,
                         include_credit_risk: bool = False,
                         hazard_params: Optional[HazardRateParams] = None,
                         seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Price bond using Monte Carlo simulation.
        
        Args:
            num_paths: Number of simulation paths
            t_steps: Number of time steps per path
            include_credit_risk: Whether to include hazard rate model
            hazard_params: Hazard rate parameters (required if include_credit_risk=True)
            seed: Random seed
        
        Returns:
            Tuple of (mean_price, std_price)
        """
        # Simulate interest rate paths
        rate_paths = self.ir_model.simulate_path(t_steps, num_paths, seed)
        
        # Simulate hazard rate paths if needed
        hazard_paths = None
        if include_credit_risk:
            if hazard_params is None:
                raise ValueError("Hazard rate parameters required for credit risk pricing")
            hazard_model = HazardRateModel(hazard_params)
            hazard_paths = hazard_model.simulate_path(t_steps, num_paths, seed)
        
        # Price bond on each path
        prices = np.zeros(num_paths)
        dt = self.bond_params.maturity / t_steps
        
        for i in range(num_paths):
            hazard_path = hazard_paths[:, i] if hazard_paths is not None else None
            prices[i] = self.price_bond(rate_paths[:, i], hazard_path, t_steps)
        
        return np.mean(prices), np.std(prices)


class CallableOptionPricer:
    """Pricer for the embedded call option in callable bonds."""
    
    def __init__(self, bond_params: CallableBondParameters,
                 ir_model: InterestRateModel):
        """
        Initialize call option pricer.
        
        Args:
            bond_params: Bond specifications
            ir_model: Interest rate model
        """
        self.bond_params = bond_params
        self.ir_model = ir_model
        self.straight_bond_pricer = StraightBondPricer(bond_params, ir_model)
    
    def is_callable_at_time(self, t: float) -> bool:
        """Check if bond is callable at time t."""
        return (self.bond_params.first_call_date <= t <= 
                self.bond_params.last_call_date + 1e-6)
    
    def value_call_option_path(self, rate_path: np.ndarray, t_steps: int = 252) -> float:
        """
        Value the call option using backward induction on a single rate path.
        
        Uses tree-based dynamic programming for optimal call decisions.
        
        Args:
            rate_path: Interest rate path from simulation
            t_steps: Number of time steps
        
        Returns:
            Call option value (to be subtracted from bond price)
        """
        dt = self.bond_params.maturity / t_steps
        coupon_payment = self.straight_bond_pricer.calculate_coupon_payment()
        
        # Start from maturity and work backward
        # Bond value at maturity
        bond_value = self.bond_params.par_value
        call_value = 0.0
        
        # Backward induction through time steps
        current_time = self.bond_params.maturity
        
        for step in range(t_steps, 0, -1):
            current_time = step * dt
            
            # Discount one period
            discount_rate = rate_path[step]
            discount_factor = np.exp(-discount_rate * dt)
            
            # Bond value includes coupon (if any)
            time_between_coupons = 1.0 / self.bond_params.coupon_frequency
            if abs((self.bond_params.maturity - current_time) % time_between_coupons) < 1e-6:
                bond_value = (bond_value + coupon_payment) * discount_factor
            else:
                bond_value = bond_value * discount_factor
            
            # Check if bond is callable at this time
            if self.is_callable_at_time(current_time):
                # Issuer exercises if bond > call price (bond trading above strike)
                if bond_value > self.bond_params.call_price:
                    bond_value = self.bond_params.call_price
                    call_value = bond_value  # Option is exercised
        
        return max(bond_value - self.bond_params.call_price, 0)
    
    def monte_carlo_price(self, num_paths: int = 1000, t_steps: int = 252,
                         seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Price call option using Monte Carlo simulation.
        
        Args:
            num_paths: Number of simulation paths
            t_steps: Number of time steps per path
            seed: Random seed
        
        Returns:
            Tuple of (mean_option_value, std_option_value)
        """
        # Simulate interest rate paths
        rate_paths = self.ir_model.simulate_path(t_steps, num_paths, seed)
        
        # Value option on each path
        option_values = np.zeros(num_paths)
        
        for i in range(num_paths):
            option_values[i] = self.value_call_option_path(rate_paths[:, i], t_steps)
        
        return np.mean(option_values), np.std(option_values)


class CallableBond:
    """Main class for callable bond pricing and analysis."""
    
    def __init__(self, bond_params: CallableBondParameters,
                 ir_params: VasicekParams,
                 hazard_params: Optional[HazardRateParams] = None,
                 ir_model_type: RateModelType = RateModelType.VASICEK):
        """
        Initialize callable bond.
        
        Args:
            bond_params: Bond specifications
            ir_params: Interest rate model parameters
            hazard_params: Hazard rate parameters (optional)
            ir_model_type: Type of interest rate model to use
        """
        self.bond_params = bond_params
        self.ir_params = ir_params
        self.hazard_params = hazard_params
        
        # Initialize interest rate model
        if ir_model_type == RateModelType.VASICEK:
            self.ir_model = VasicekModel(ir_params)
        elif ir_model_type == RateModelType.CIR:
            self.ir_model = CIRModel(ir_params)
        else:
            raise ValueError(f"Unknown interest rate model: {ir_model_type}")
        
        # Initialize pricers
        self.straight_bond_pricer = StraightBondPricer(bond_params, self.ir_model)
        self.call_option_pricer = CallableOptionPricer(bond_params, self.ir_model)
    
    def price_callable_bond(self, num_paths: int = 1000, t_steps: int = 252,
                           include_credit_risk: bool = False,
                           seed: Optional[int] = None) -> dict:
        """
        Price callable bond: CB Price = Bond Price - Call Option Price
        
        Args:
            num_paths: Number of Monte Carlo paths
            t_steps: Number of time steps per path
            include_credit_risk: Whether to include credit spread
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary with pricing results
        """
        # Price straight bond
        bond_price, bond_std = self.straight_bond_pricer.monte_carlo_price(
            num_paths=num_paths,
            t_steps=t_steps,
            include_credit_risk=include_credit_risk,
            hazard_params=self.hazard_params,
            seed=seed
        )
        
        # Price embedded call option
        option_price, option_std = self.call_option_pricer.monte_carlo_price(
            num_paths=num_paths,
            t_steps=t_steps,
            seed=seed
        )
        
        # Callable bond price
        callable_bond_price = bond_price - option_price
        
        # Approximate standard error (combining independent errors)
        callable_bond_std = np.sqrt(bond_std ** 2 + option_std ** 2)
        
        return {
            'callable_bond_price': callable_bond_price,
            'callable_bond_std': callable_bond_std,
            'straight_bond_price': bond_price,
            'straight_bond_std': bond_std,
            'embedded_call_value': option_price,
            'embedded_call_std': option_std,
            'option_adjusted_spread': self._calculate_oas(callable_bond_price),
        }
    
    def _calculate_oas(self, price: float) -> float:
        """
        Estimate Option-Adjusted Spread (simplified).
        
        This is a placeholder for more sophisticated OAS calculation.
        """
        ytm_spread = np.log(self.bond_params.par_value / price) / self.bond_params.maturity
        return ytm_spread
    
    def calculate_duration(self, num_paths: int = 1000, t_steps: int = 252,
                          spread_basis_points: float = 1.0,
                          seed: Optional[int] = None) -> float:
        """
        Calculate effective duration of callable bond.
        
        Duration ≈ (Price_down - Price_up) / (2 × Price × Δy)
        
        Args:
            num_paths: Number of Monte Carlo paths
            t_steps: Number of time steps per path
            spread_basis_points: Yield spread to shock (in basis points)
            seed: Random seed
        
        Returns:
            Effective duration
        """
        base_price = self.price_callable_bond(num_paths, t_steps, seed=seed)['callable_bond_price']
        
        # Shift rates up
        original_rate = self.ir_params.initial_rate
        self.ir_params.initial_rate = original_rate + spread_basis_points / 10000
        price_up = self.price_callable_bond(num_paths, t_steps, seed=seed)['callable_bond_price']
        
        # Shift rates down
        self.ir_params.initial_rate = original_rate - spread_basis_points / 10000
        price_down = self.price_callable_bond(num_paths, t_steps, seed=seed)['callable_bond_price']
        
        # Restore original rate
        self.ir_params.initial_rate = original_rate
        
        # Calculate duration
        delta_yield = 2 * spread_basis_points / 10000
        duration = (price_down - price_up) / (base_price * delta_yield)
        
        return duration
    
    def calculate_convexity(self, num_paths: int = 1000, t_steps: int = 252,
                           spread_basis_points: float = 1.0,
                           seed: Optional[int] = None) -> float:
        """
        Calculate effective convexity of callable bond.
        
        Convexity = (Price_down + Price_up - 2 × Price) / (Price × (Δy)²)
        
        Args:
            num_paths: Number of Monte Carlo paths
            t_steps: Number of time steps per path
            spread_basis_points: Yield spread to shock (in basis points)
            seed: Random seed
        
        Returns:
            Effective convexity
        """
        base_price = self.price_callable_bond(num_paths, t_steps, seed=seed)['callable_bond_price']
        
        # Shift rates up
        original_rate = self.ir_params.initial_rate
        self.ir_params.initial_rate = original_rate + spread_basis_points / 10000
        price_up = self.price_callable_bond(num_paths, t_steps, seed=seed)['callable_bond_price']
        
        # Shift rates down
        self.ir_params.initial_rate = original_rate - spread_basis_points / 10000
        price_down = self.price_callable_bond(num_paths, t_steps, seed=seed)['callable_bond_price']
        
        # Restore original rate
        self.ir_params.initial_rate = original_rate
        
        # Calculate convexity
        delta_yield = spread_basis_points / 10000
        convexity = (price_down + price_up - 2 * base_price) / (base_price * (delta_yield ** 2))
        
        return convexity
    
    def print_summary(self, pricing_results: dict) -> str:
        """Generate a summary of callable bond valuation."""
        output = "\n" + "=" * 70 + "\n"
        output += "CALLABLE BOND PRICING SUMMARY\n"
        output += "=" * 70 + "\n\n"
        
        output += "Bond Specifications:\n"
        output += f"  Par Value:           ${self.bond_params.par_value:,.2f}\n"
        output += f"  Coupon Rate:         {self.bond_params.coupon_rate*100:.3f}%\n"
        output += f"  Maturity:            {self.bond_params.maturity:.2f} years\n"
        output += f"  Call Price:          ${self.bond_params.call_price:,.2f}\n"
        output += f"  First Call Date:     {self.bond_params.first_call_date:.2f} years\n"
        output += f"  Call Type:           {self.bond_params.callable_type.value}\n\n"
        
        output += "Interest Rate Model:\n"
        output += f"  Model Type:          {type(self.ir_model).__name__}\n"
        output += f"  Mean Reversion:      {self.ir_params.mean_reversion:.4f}\n"
        output += f"  Long-term Mean:      {self.ir_params.long_term_mean*100:.3f}%\n"
        output += f"  Volatility:          {self.ir_params.volatility*100:.3f}%\n"
        output += f"  Initial Rate:        {self.ir_params.initial_rate*100:.3f}%\n\n"
        
        output += "Valuation Results:\n"
        output += f"  Straight Bond Price: ${pricing_results['straight_bond_price']:.4f} "
        output += f"(±${pricing_results['straight_bond_std']:.4f})\n"
        output += f"  Embedded Call Value: ${pricing_results['embedded_call_value']:.4f} "
        output += f"(±${pricing_results['embedded_call_std']:.4f})\n"
        output += f"  Callable Bond Price: ${pricing_results['callable_bond_price']:.4f} "
        output += f"(±${pricing_results['callable_bond_std']:.4f})\n"
        output += f"  Option Adj. Spread:  {pricing_results['option_adjusted_spread']*100:.3f}%\n\n"
        
        output += "=" * 70 + "\n"
        
        return output


# Example usage and testing
if __name__ == "__main__":
    # Define bond specifications
    bond_params = CallableBondParameters(
        par_value=100.0,
        coupon_rate=0.06,
        maturity=5.0,
        coupon_frequency=2,
        call_price=102.0,
        first_call_date=2.0,
        last_call_date=5.0,
        callable_type=CallableType.AMERICAN
    )
    
    # Define interest rate model parameters (Vasicek)
    ir_params = VasicekParams(
        mean_reversion=0.15,
        long_term_mean=0.05,
        volatility=0.015,
        initial_rate=0.04
    )
    
    # Define hazard rate parameters (optional)
    hazard_params = HazardRateParams(
        mean_reversion=0.10,
        long_term_mean=0.02,
        volatility=0.01,
        initial_hazard=0.015,
        recovery_rate=0.4
    )
    
    # Create callable bond
    callable_bond = CallableBond(
        bond_params=bond_params,
        ir_params=ir_params,
        hazard_params=hazard_params,
        ir_model_type=RateModelType.VASICEK
    )
    
    # Price the callable bond
    pricing_results = callable_bond.price_callable_bond(
        num_paths=5000,
        t_steps=252,
        include_credit_risk=False,
        seed=42
    )
    
    # Print summary
    print(callable_bond.print_summary(pricing_results))
    
    # Calculate risk metrics
    print("\nCalculating effective duration and convexity...")
    duration = callable_bond.calculate_duration(num_paths=1000, t_steps=252, seed=42)
    convexity = callable_bond.calculate_convexity(num_paths=1000, t_steps=252, seed=42)
    
    print(f"Effective Duration:  {duration:.4f} years")
    print(f"Effective Convexity: {convexity:.4f}\n")
