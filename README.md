# RMWM — World Models for Airline Revenue Management

## A Framework for Causal, Simulatable Revenue Management

RMWM develops world models for airline revenue management — a unified environmental representation that connects pricing decisions, latent demand, competitive dynamics, and observed booking outcomes over the booking horizon.

### The Core Thesis

Current airline RM systems are **forecasting-optimization pipelines** that break the critical feedback loop between pricing and demand. There is no coherent **world model** in industry RM that:

- Represents latent demand as a proper state with an observation model
- Explicitly models censoring (bookings = min(demand, capacity))
- Enables causal counterfactual queries about pricing interventions
- Supports long-horizon policy optimization beyond myopic bid prices
- Captures competitive and strategic customer dynamics

### Approach: Dual Path — Hybrid vs. Learned

Two parallel research tracks:

1. **Hybrid (Structural + Learned)** — Structural causal priors from econometric theory + data-driven components for flexibility. Follows the Pearl/Turing tradition.
2. **Pure Learned** — Latent state models learned entirely from data, following the Ha/Schmidhuber/Dreamer tradition.

Both approaches are developed formally and compared across RM-specific dimensions.

### Scope

- **Unit level**: Origin-Destination (OD) level — where pricing decisions and willingness-to-pay are fundamentally defined
- **Time scale**: Full booking window (time-to-departure as state variable)
- **Market**: Two-class (business/leisure) → Multi-class generalization

### Goals

1. Publish a rigorous definition of "world model" for RM that the industry can adopt
2. Produce implementation-ready architectures for both hybrid and pure-learned approaches
3. Provide a high-fidelity synthetic data generator for experimental validation
4. Target venues: **AGIFORS** (industry impact) + **NeurIPS/KDD/AAAI** (AI community)

---

## Repository Structure

```
rmwm/
├── README.md                    # This file
├── docs/
│   ├── phase1/                  # World model definitions & literature
│   ├── phase2/                  # POMDP formalization
│   ├── phase3/                  # Architecture specifications
│   ├── phase4/                  # Data generation & simulation
│   └── phase5/                  # Final paper & experiments
├── design/
│   ├── hybrid/                  # Hybrid framework design
│   └── learned/                 # Pure-learned framework design
├── data/                        # Synthetic data generators
├── simulations/                 # Experiment code & results
├── paper/                       # Research paper drafts
└── assets/                      # Figures & supplementary
```

---

## Key Decisions & Constraints

### 1. Od-Level Focus
Pricing and WTP naturally occur at the Origin-Destination (OD) level. While itinerary-level cross-elasticity exists, the OD level is the fundamental unit of pricing decisions.

### 2. Dual Approach Investigation
- **Structural/CA**: Follow Pearl's causal graphs + structural econometrics (BLP models) + SCM
- **ML/Learned**: Follow world models/RL (Dreamer, MuZero, PETS)
- Both approaches investigated in parallel for comparison

### 3. Synthetic Data First
High-fidelity synthetic data generated via:
- Probability distribution generators modeled on airline demand patterns
- Reverse-engineering real airline pricing data from online sources
- Structured demand scenarios with known ground truth for validation

### 4. Target Venues
- **AGIFORS**: Industry standards body — focus on practical applicability
- **AI Community**: NeurIPS/KDD/AAAI — focus on novel ML methods applied to economics

---

## Status: Phase 1 Complete

- ✅ World model definition (3 competing definitions → unified 4-component spec)
- ✅ Literature map across 5 domains (80+ references)
- ✅ 6 critical gaps in current RM practice identified
- ✅ WM = (T, O, C, Π) formal definition established
- ⏳ Phase 2: POMDP formalization → OD-level state space
- ⏳ Phase 3: Architecture specs for both approaches
- ⏳ Phase 4: Synthetic data generator design
- ⏳ Phase 5: Final paper draft

---

## References

### World Models in ML/RL
- Ha & Schmidhuber (2018) — World Models
- Ha & Hafner (2018) — Dreamer
- Hafner et al. (2021) — DreamerV2
- Hafner et al. (2023) — DreamerV3
- Schrittwieser et al. (2020) — MuZero

### Airline Revenue Management
- Talluri & van Ryzin (2004) — RM textbook
- Smith & Leimkuhler (1980) — Yield management optimization
- Littlewood (1972) — Two-class booking limit

### Causal Inference in Pricing
- Pearl (2009) — Causality
- Imbens & Rubin (2015) — Causal Inference
- Athey & Imbens (2017) — Statistical economics of causal inference
- Kallus & Zhou (2018) — Causal inference in pricing

### Structural Econometrics
- Berry, Levinsohn, Pakes (1995) — BLP model
- Dubé et al. (2012) — Competitive pricing dynamics

---

## License: MIT

Developed by [edtaylor]
