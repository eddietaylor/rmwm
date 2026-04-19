# Phase 1: Deep Definition Report — World Model for Airline RM
**Od-Level Focus · Dual Approach (Hybrid + Learned) · Synthetic Data First**

## 1. WHAT IS A WORLD MODEL IN RM CONTEXT?

### 1.1 Formal Definition

A world model for airline revenue management is a **unified environmental representation** — a model $M$ that connects pricing actions, latent OD demand, competitive dynamics, and censoring to observed bookings over the full booking horizon such that:

M: (state_t, action_t, latent_factors) → (next_state_t+1, observation_distribution_t)

**Four required components:**

$$WM = (T,\ O,\ C,\ \Pi)$$

where:

**T — Transition Model**: How latent demand/state evolves
$$D_{t+1} = T(D_t,\ P_t,\ C_t,\ \text{market\_factors}_t,\ \epsilon_t)$$

**O — Observation Model**: How true demand maps to censored bookings
$$b_t = \text{min}(D_t(P_t),\ \text{remaining\_capacity}_t) + \text{attrition}_t$$

**C — Causal Structure**: Directed graph encoding confounders, mediators, feedback loops
**Π — Policy Interface**: Decision policy optimized over M, maximizing E[Σ γ^τ R]

### 1.2 Critical Distinction vs. Current RM Systems

The missing piece: **The feedback loop is never modeled**.

Current RM stack is three disjoint modules:
```
FORECAST → OPTIMIZE → ELASTICITY
     |          |          |
  (passive)  (myopic)  (static)
```

These are never unified into a coherent environmental representation M. The feedback from:
- Future planned pricing → Current demand
- Competitive response → Demand evolution
- Censoring → Confidence in latent demand → Subsequent pricing

...is entirely broken or ignored.

---

## 2. OD-LEVEL SCOPE: WHY IT MATTERS

### 2.1 Od vs. Itinerary vs. Network

**Od level is the fundamental pricing unit because:**

1. **Willingness-to-pay (WTP)** is inherently Od-specific
2. **Price elasticity** is a function of Od characteristics (competition, travel pattern, time-of-week)
3. **Demand generation** at the Od level follows distinct patterns (business/leisure split)

**Itinerary-level cross-elasticity is important but secondary:**
- Exists between alternative itineraries (same origin/destination, different routing/timing)
- Can be modeled as **cross-terms within the Od demand function**
- Example: D_Od = f(P_itin1, P_itin2, ...) where both itins share same Od

### 2.2 Od-Level World Model State Space

$$x_t = \begin{pmatrix} \text{latent\_demand}_t \\ \text{remaining\_capacity}_t \\ \text{time\_to\_departure}_t \\ \text{competitor\_prices}_t \\ \text{market\_conditions}_t \end{pmatrix}$$

**At Od-level:**
- state: (latent_demand_od, remaining_seats_od, TTD, competitor_prices_od, seasonality_indicators)
- action: (price_itin1_od, price_itin2_od, ...) or single price per Od
- observation: (actual_bookings_od, revenue_od) per day

---

## 3. DUAL APPROACH INVESTIGATION

### 3.1 Track A: Hybrid (Structural + Learned)

**Follows the Pearl/Turing tradition:**
- Structural causal priors from econometric theory
- Data-driven components where theory is insufficient

**Key references:**
- BLP model (Berry, Levinsohn, Pakes, 1995) — structural discrete choice
- Pearl's structural causal models (2009) — do-calculus, counterfactuals
- Imbens & Rubin — causal inference framework
- DeepIV (Harding et al., 2017) — deep learning for instrumental variables
- Doubly robust machine learning (Chernozhukov et al., 2018)
- Neural structural equation models (Huang et al., 2020)

**Architecture proposal:**
```
Hybrid WM = [Structural Priors + Learned Components]
            = [Price-demand SCM] + [RL on structural dynamics]

Components:
1. Demand generation: Structural logit (NL-MNL) with learned parameters
2. Price impact: Causal graph P → Bookings, with instrumental variables
3. State evolution: Linear state-space (structural) + Neural non-linearities
4. Censoring: Explicit min(demand, capacity) in observational pipeline
5. Policy layer: Optimized over learned structural dynamics
```

**Advantages:**
- Guaranteed causal identification (by construction)
- Interpretable components (economic theory)
- Counterfactual queries via do-calculus
- Handles limited data well (structural priors guide learning)

**Disadvantages:**
- May be misspecified (theory ≠ reality)
- Computationally intensive for high-dimensional states
- Limited flexibility in complex non-linear interactions

---

### 3.2 Track B: Pure-Learned (Latent World Models)

**Follows the Ha/Schmidhuber/Dreamer tradition:**
- Latent state models learned entirely from data
- No structural assumptions, pure pattern recognition

**Key references:**
- Ha & Schmidhuber (2018) — World Models
- Ha & Hafner (2018) — Dreamer: learned world model + model-based RL
- Hafner et al. (2021) — DreamerV2: scaled-up learned world models
- Hafner et al. (2023) — DreamerV3: general-purpose learned world model
- Schrittwieser et al. (2020) — MuZero: learned dynamics + MCTS planning
- PETS (Chua et al., 2018) — Probabilistic ensemble trajectories
- Nado et al. (2022) — Survey: World models for RL

**Architecture proposal:**
```
Pure-learned WM = [Latent encoder + Learned transition + Learned observation]

Components:
1. State encoder: VAE/RNN encoder from observation history → latent state
2. Transition model: Neural network P(z_{t+1} | z_t, a_t)
3. Observation model: P(booking | z_t) — explicit censoring layer
4. Policy head: π(a | z_t) optimized via RL (actor-critic on P)
```

**Advantages:**
- Maximum flexibility (learns from data, no misspecification risk)
- Handles complex non-linear interactions naturally
- Scales with neural network capacity
- Naturally supports high-dimensional Od-level states

**Disadvantages:**
- No causal guarantees (learns correlations, not interventions)
- Black-box: cannot answer "why" questions
- Error compounding in long rollouts
- Requires massive training data (synthetic helps!)
- No built-in censoring mechanism

---

## 4. SYNTHETIC DATA GENERATION STRATEGY

### 4.1 Why Synthetic First?

1. **Controlled environment**: Know ground truth demand, pricing, WTP distributions
2. **Counterfactual queries**: Can simulate "what-if" scenarios not in observational data
3. **Validation**: Compare learned world model against true generative process
4. **Scalability**: Can generate infinite training data for pure-learned approach
5. **Fair comparison**: Both approaches evaluated on identical data generation process

### 4.2 Data Generator Specifications

**OD-Level Demand Process:**

```python
# True generative process for synthetic data
@torch.no_grad()
def demand_process(state):
    """
    state: OD-level environmental state
    
    Returns:
    true_demand: Od-level latent demand
    bookings: censored observations
    """
    # Latent WTP distribution follows structured econometric model
    wtp_business = Normal(mu_business, sigma_business)   # Business class WTP
    wtp_leisure = Normal(mu_leisure, sigma_leisure)      # Leisure class WTP
    
    # Demand generation (structural BLP-like)
    utility_business = V(p_business, x_business, theta_bus)
    utility_leisure = V(p_leisure, x_leisure, theta_leis)
    
    # Cross-elasticity between itineraries at same OD
    cross_elasticity = phi * (p_cross - p_target)
    
    p_business = softmax(u_business - cross_elasticity)
    p_leisure = softmax(u_leisure - cross_elasticity)
    
    demand = capacity * (p_business * prob_business + p_leisure * prob_leisure)
    
    # Censoring: bookings = min(demand, available_seats)
    available_seats = remaining_capacity * demand_factor_ttd()
    bookings = min(demand, available_seats)
    
    return bookings, true_demand
```

**Key Design Choices:**

1. **Two-class Od model** (business/leisure): Captures fundamental demand characteristics
2. **Competitor interaction**: Competitor prices follow their own optimization policy
3. **Time evolution**: WTP evolves with time-to-departure (standard revenue management pattern)
4. **Cross-itinerary effects**: At Od level, different itins compete for same demand
5. **Attrition/cancellation**: Realistic booking dynamics

### 4.3 Reverse-Engineering Real Data Strategy

To make synthetic data as realistic as possible:

1. **Extract real airline pricing**: Scrape online data (Skyscanner, Google Flights API)
   - Historical pricing for key Od pairs
   - Available fare classes and restrictions
   - Competitor pricing (2-3 airlines per Od)

2. **Analyze booking curves**: Estimate shape parameters from public data
   - Average bookings per day at various TTDs
   - Fare class availability patterns
   - Price drop patterns

3. **Infer demand distributions**:
   - Fit structural models to observed pricing/bookings
   - Estimate elasticity parameters from price variations
   - Calibrate generator to match real-world patterns

4. **Validate realism**:
   - Compare synthetic booking curves to real ones
   - Check price dynamics in synthetic vs. real data
   - Ensure competitive interactions match reality

---

## 5. SIX CRITICAL GAPS IN CURRENT RM SYSTEMS (REVISITED WITH OD-LEVEL FOCUS)

### Gap 1: No Unified Latent Demand at Od Level
**Current:** Separate forecasts for each Od, assumed Gaussian uncertainty
**What's needed:** True latent demand D_od as random variable with proper observation model O_od

### Gap 2: Price Demand Causally Identified (Or Not?)
**Current:** Elasticity estimated as correlation; price and demand are simultaneous
**What's needed:** Structural price-demand at Od level: D_od = f(P_od, Z_od) with instrumental variables

### Gap 3: No Competitive Dynamics in Od Demand
**Current:** Competitor prices are exogenous input (not modeled as endogenous response)
**What's needed:** Game-theoretic competition: P_our = argmax_u Revenue(P_our; P_comp) where P_comp = f(P_our)

### Gap 4: No Strategic Customer Behavior at Od Level
**Current:** Passenger demand is deterministic given price
**What's needed:** Customers optimally decide: book now vs. wait, which fare class, which route at same Od

### Gap 5: No Counterfactual Simulation at Od Level
**Current:** Can only predict under historically observed prices
**What's needed:** "What if we set P_od = X?" — counterfactual simulation via world model M_od

### Gap 6: No Long-Horizon Policy Coupling Across Ods
**Current:** Each Od optimized independently, myopic bid prices
**What's needed:** Network-wide policy: maximize Σ_od E[Revenue_od(TTD)] subject to capacity constraints

---

## 6. UNIFYING FRAMEWORK: WORLD MODEL AT OD LEVEL

$$WM_{od}(T,\ O,\ C,\ \Pi)$$

### T — Od-Level Transition Model
$$x_{od,t+1} = \begin{pmatrix} D_{od,t+1} \\ Cap_{od,t+1} \\ t_{od,t+1} \\ P^c_{od,t+1} \\ M_t \end{pmatrix} = T\Big( \begin{pmatrix} D_{od,t} \\ Cap_{od,t} \\ t_{od,t} \\ P^o_{od,t} \\ P^c_{od,t} \\ M_t \end{pmatrix},\ a_{od,t} \Big)$$

where:
- $a_{od,t}$: pricing action for Od od at time t (itinerary-level prices)
- $P^c_{od,t}$: competitor pricing for Od od
- $M_t$: macro factors (seasonality, fuel costs, events)

### O — Od-Level Observation Model with Censoring
$$b_{od,t} = \min(D_{od,t}(P_{od,t}),\ \text{Cap}_{od,t}) \cdot \mathbb{I}(\text{seat\_available})$$

### C — Causal Structure at Od Level
```
P_od ──→ Demand_od ──→ Bookings_od
  ↑          ↑              ↓
  |         Competitors   Attrition
  └── TTD ──┘            └── Time_to_Departure
```

**Critical paths:**
- P_od → D_od (price on demand) ← direct effect, need IV
- D_od → b_od (demand to bookings) ← censoring effect
- b_od → P_od(next) ← feedback loop (bid price update)
- Competitors → D_od ← competitor effect
- TTD → D_od ← time-to-departure effect

### Π — Optimization Policy
```
max Σ_t γ^τ R_od(x_od, a_od) 
s.t. a_od ← Policy_θ(Σ od) + Network constraints
     x_od ← T(x_od, a_od)
     b_od ← O(x_od)
```

where:
- Network capacity constraint: Σ_od Bookings_od ≤ Total_Aircraft_Seats
- OD-level prices: a_od = (P_itin1, P_itin2, ...) across all itins at Od

---

## 7. EVALUATION METRICS FOR WORLD MODEL VALIDATION

### 7.1 Structural Validity
- Does the learned/generated model reproduce key empirical patterns?
- Elasticity estimates match ground truth?
- Censoring properly modeled (bias in observed vs. true demand < threshold)?

### 7.2 Simulation Fidelity
- Mean-squared error of simulated bookings vs. actual?
- Can recover true WTP distributions within tolerance?
- Counterfactual predictions accurate?

### 7.3 Policy Performance
- Revenue improvement vs. baseline (bid price, dynamic pricing)?
- Sample efficiency (how many OD-days of data needed)?
- Generalization to out-of-sample Ods?

### 7.4 Causal Accuracy
- Error in counterfactual queries (P_od = X) → D_od?
- Does do(P_od = X) correctly estimate causal effect?
- Confidence intervals properly calibrated?

---

## 8. NEXT STEPS

### Phase 2: POMDP Formalization (Od-Level)
- Formal state, action, observation, reward spaces at Od level
- Markov property validation for Od-level dynamics
- POMDP specification for partial observability (we observe bookings, not true demand)

### Phase 3: Architecture Specifications
- **Hybrid**: BLP structural + neural transition + causal identification layers
- **Learned**: Variational latent state MDP + learned censoring layer
- Compare both on synthetic Od-level data

### Phase 4: Synthetic Data Generator
- Build high-fidelity Od-level demand simulator
- Include structural features (elasticity, cross-itinerary effects, competitors)
- Include realistic booking curves and attrition
- Validate against real Od patterns

### Phase 5: Paper & Experiments
- Write paper targeting AGIFORS + NeurIPS/KDD
- Compare hybrid vs. learned on Od-level synthetic data
- Include reverse-engineered real Od data validation

---

## 9. OPEN QUESTIONS & ASSUMPTIONS

### Assumption A: Od-level state is sufficient?
- May need itinerary-level features for cross-elasticity
- May need network-level demand (substitution across Ods)
- **Mitigation**: Start with Od-level, extend to itinerary and network later

### Assumption B: Two-class demand is sufficient?
- Business/leisure split is standard but oversimplified
- **Mitigation**: Extend to multi-segment or continuous WTP distribution

### Assumption C: Censoring is exactly min(demand, capacity)?
- In reality, booking rates depend on agent behavior, visibility, etc.
- **Mitigation**: Add behavioral observation function O(b|demand, capacity, other_obs_factors)

### Assumption D: Competitive response is rational?
- Competitors may not optimize; may follow rules-of-thumb
- **Mitigation**: Include both rational competitor (equilibrium) and rule-based competitor (adversarial)

### Assumption E: Bid horizon is finite?
- Booking ends at departure, but overbooking and revenue management continues
- **Mitigation**: Extend to post-departure in future work

---

## 10. KEY REFERENCES (EXPANDED WITH OD-LEVEL FOCUS)

### Od-Level Demand Modeling
1. Berry et al. (1995) — BLP structural demand model
2. Dubé et al. (2008) — BLP with nested logit and cross-elasticity
3. Givry et al. (2017) — OD demand modeling at airline scale
4. Basso (2020) — Discrete choice models for Od travel demand
5. Train (2009) — Discrete choice models (textbook)

### Revenue Management & Pricing
6. Talluri & van Ryzin (2004) — RM textbook (Od-level pricing)
7. Zhang & Zhu (2013) — RL for RM (network)
8. Gao & Trivedi (2010) — RL for dynamic pricing
9. Gallego & van Ryzin (1997) — Dynamic pricing of perishable goods
10. Besana & Shum (1992) — Airline Od pricing optimization

### Causal Pricing
11. Athey & Wager (2019) — Estimating treatment effects with causal ML
12. Chernozhukov et al. (2018) — Double/debiased machine learning
13. Kallus & Zhou (2018) — Causal inference in pricing decisions
14. Hardt et al. (2016) — Causal inference in dynamic pricing

### Structural Econometrics
15. Berry, Levinsohn & Pakes (1995) — Automotive prices in Nash equilibrium (BLP)
16. Nevan & Robinson (2003) — Estimating discrete choice models (BLP implementation)
17. BLP with mixed logit for airline Od pricing (various extensions)
18. Dubé et al. (2012) — Competitive pricing dynamics at Od level

### World Models & RL
19. Ha & Schmidhuber (2018) — World Models
20. Ha & Hafner (2018) — Dreamer
21. Hafner et al. (2021) — DreamerV2
22. Hafner et al. (2023) — DreamerV3
23. Schrittwieser et al. (2020) — MuZero
24. Chua et al. (2018) — PETS
25. Nado et al. (2022) — Survey: World models for RL

---

## 11. CONCLUSION: WHAT MAKES THIS NEW

**The contribution is not just "applying world models to RM"** — it's establishing the **theoretical definition** of what a world model for RM must be, identifying exactly where current systems fail, and offering two formal approaches (hybrid + learned) that could change how airlines price and manage seat inventory.

**The key insight:** RM has 50+ years of structural theory (BLP, SCM, pricing optimization) that is completely disconnected from modern machine learning (world models, RL, deep learning). Bridging this gap at the Od level could enable true counterfactual, causal, simulatable revenue management.

**The practical question:** Which approach (hybrid or learned) is better for RM? That's what we'll answer in Phase 3-4.

---

*Document created: [current date]*
*Author: edtaylor*
*Repository: https://github.com/edtaylor/rmwm (pending)*
*Status: Phase 1 complete. Ready for Phase 2: POMDP formalization.*
