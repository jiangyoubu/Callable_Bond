# Callable Bond Pricing Model - Documentation

## Overview
This module provides an object-oriented framework for pricing callable bonds using Monte Carlo simulation with stochastic interest rate and hazard rate models.

## Pricing Methodology

### Core Formula
```
Callable Bond Price = Straight Bond Price - Embedded Call Option Price
```

## Key Classes

### 1. **Interest Rate Models**

#### VasicekModel
- Implements the Vasicek interest rate model: `dr_t = κ(θ - r_t)dt + σ dW_t`
- Suitable for positive and negative interest rates
- Parameters:
  - `mean_reversion` (κ): Speed of mean reversion
  - `long_term_mean` (θ): Long-term equilibrium rate
  - `volatility` (σ): Rate volatility
  - `initial_rate` (r0): Starting rate

#### CIRModel
- Implements the Cox-Ingersoll-Ross model: `dr_t = κ(θ - r_t)dt + σ√r_t dW_t`
- Ensures non-negative interest rates (subject to Feller condition)
- Uses Milstein scheme for better numerical accuracy

### 2. **Hazard Rate Model**

#### HazardRateModel
- Models credit spreads: `dh_t = κ_h(λ_0 - h_t)dt + σ_h dW_t^h`
- Calculates survival probability to account for credit risk
- Parameters:
  - `mean_reversion`: Mean reversion speed of spreads
  - `long_term_mean`: Long-term hazard rate
  - `volatility`: Hazard rate volatility
  - `recovery_rate`: Recovery in case of default

### 3. **Bond Pricing Classes**

#### StraightBondPricer
- Prices non-callable bonds using Monte Carlo
- Handles coupon payments and principal repayment
- Optional credit risk adjustment via survival probability

#### CallableOptionPricer
- Values the embedded call option on callable bonds
- Uses backward induction on interest rate paths
- Supports American, European, and Bermuda callable options

### 4. **Main Class: CallableBond**

The central class that orchestrates:
- Bond and option valuation
- Risk metrics calculation (duration, convexity)
- Summary reporting

## Usage Example

```python
from callable_bond import (
    CallableBond, CallableBondParameters, VasicekParams, 
    HazardRateParams, RateModelType
)

# Define bond specifications
bond_params = CallableBondParameters(
    par_value=100.0,
    coupon_rate=0.06,           # 6% annual coupon
    maturity=5.0,               # 5-year bond
    coupon_frequency=2,         # Semi-annual coupons
    call_price=102.0,           # Issuer can call at 102
    first_call_date=2.0,        # Callable starting in 2 years
    callable_type=CallableType.AMERICAN
)

# Define interest rate model parameters
ir_params = VasicekParams(
    mean_reversion=0.15,
    long_term_mean=0.05,
    volatility=0.015,
    initial_rate=0.04
)

# Optional: Define credit spread parameters
hazard_params = HazardRateParams(
    mean_reversion=0.10,
    long_term_mean=0.02,
    volatility=0.01,
    initial_hazard=0.015
)

# Create and price the callable bond
callable_bond = CallableBond(
    bond_params=bond_params,
    ir_params=ir_params,
    hazard_params=hazard_params,
    ir_model_type=RateModelType.VASICEK
)

# Price using Monte Carlo
results = callable_bond.price_callable_bond(
    num_paths=5000,
    t_steps=252,
    include_credit_risk=False
)

print(callable_bond.print_summary(results))

# Calculate effective duration and convexity
duration = callable_bond.calculate_duration(num_paths=1000)
convexity = callable_bond.calculate_convexity(num_paths=1000)
```

## Data Classes

### CallableBondParameters
Bond specification including:
- Par value and coupon
- Maturity and call terms
- Call type (American/European/Bermuda)

### VasicekParams / HazardRateParams
Model parameter containers with validation

## Monte Carlo Simulation Details

1. **Interest Rate Paths**: Simulates multiple scenarios for interest rates using Euler/Milstein discretization
2. **Bond Valuation**: Calculates present value of all cash flows on each path
3. **Option Valuation**: Uses backward induction to determine optimal call exercise
4. **Aggregation**: Averages results across paths to obtain expected prices and standard errors

## Risk Metrics

### Effective Duration
- Measures sensitivity to parallel rate shifts
- Calculated via finite differences in bond prices

### Effective Convexity
- Measures non-linearity of price-yield relationship
- Negative for callable bonds due to embedded short call option

## Parameters (Currently Constant)

The following parameters can be calibrated to market data:
- Interest rate model parameters (κ, θ, σ, r0)
- Hazard rate model parameters
- Initial conditions and volatilities

## Future Extensions

- Stochastic volatility models
- Two-factor interest rate models
- Real-world vs. risk-neutral probability measures
- Calibration routines for parameters
- Binomial/trinomial tree methods
- Parallel computing for large-scale simulations
