# Phase 2: POMDP Formalization — Od-Level World Model for Airline RM

## Abstract

This document provides a rigorous Partially Observable Markov Decision Process (POMDP) formalization of the airline revenue management problem at the Od-level, where pricing decisions and willingness-to-pay are fundamentally defined. We establish the complete mathematical structure for both hybrid (structural+learned) and pure-learned approaches, with explicit attention to censoring, competitive dynamics, and the feedback loop that current RM systems fail to capture.

---

## 1. POMDP FORMALIZATION AT OD LEVEL

### 1.1 Od-Level POMDP Definition

We formalize the revenue management problem for a single Od pair as a POMDP with the following components:

**Od-Level POMDP = (S, A, Ω, T, O, R, γ)**

where all components are defined at the Od level, with network-level coupling via shared capacity constraints.

### 1.2 State Space S: Latent Od State

The state space S captures all Od-level latent variables that affect current and future revenue:

$$S = \mathcal{D} \times \mathcal{C} \times \mathcal{T} \times \mathcal{P}_c \times \mathcal{M} \times \mathcal{X}$$

where each component is:

**D — Latent Demand**: True Od-level demand (unobserved)
$$\mathcal{D} \subset \mathbb{R}_+^{K}$$
where K is the number of fare classes/itinaries at the Od, and D_k represents latent bookings for fare class k if unconstrained by capacity.

**C — Remaining Capacity**: Available Od capacity
$$\mathcal{C} \subset \mathbb{N}_0$$
where C represents remaining seats for the Od (or more granularly, by fare class or fare bucket).

**T — Time-to-Departure**: Number of days until departure
$$\mathcal{T} = \{0, 1, ..., T_{max}\}$$
where T_max is the maximum look-ahead window (typically 365 days).

**P_c — Competitor Pricing**: Od-level competitor prices
$$\mathcal{P}_c \subset \mathbb{R}_+^{J}$$
where J is the number of competitors at the Od, and P_{c,j} represents competitor j's price for the comparable itinerary.

**M — Market Conditions**: Od-level macro factors
$$\mathcal{M} \subset \mathbb{R}^{d_m}$$
where d_m is the dimensionality of market features (seasonality, fuel costs, events, day-of-week, holidays).

**X — State Memory**: Historical booking patterns (for non-Markovian components)
$$\mathcal{X} \subset \mathbb{R}^{d_h}$$
where d_h is the history length of booking data needed to capture non-Markovian demand.

**Complete Od state:**
$$s = (D, C, \tau, P_c, m, x) \in \mathcal{S}$$

---

### 1.3 Action Space A: Pricing Decisions

The action space A represents pricing decisions at the Od level:

**A = {a | a = (p_1, p_2, ..., p_K)}**

where p_i is the fare price for itinerary k at the Od, subject to:

**Constraints:**
1. Price bounds: p_min ≤ p_k ≤ p_max for all k
2. Fare class hierarchy: P_business ≥ P_leisure ≥ P_promo (typically)
3. Network coupling: Σ_k a_k ≤ C (capacity constraint)

**Action representation:**
$$a = (a_{od,1}, a_{od,2}, ..., a_{od,K}) \in \mathcal{A}$$

where a_{od,k} is the price set for itinerary k at Od od.

**Policy:**
$$\pi: \mathcal{O}_{history} \rightarrow \mathcal{A}$$
maps observation history to pricing actions.

---

### 1.4 Observation Space Ω: Censored Bookings

The observation space Ω captures what we actually observe (not the true demand):

$$\Omega = \Omega_b \times \Omega_{revenue} \times \Omega_{comp}$$

**Observable components:**

**b — Observed Bookings**: Censored bookings at Od level
$$\tilde{b_t} = \min(D_t, C_t) \cdot \mathbb{I}(\text{seat\_available}) \in \mathbb{N}_0$$

This is the **critical censoring mechanism** — we never observe true demand D_t directly.

**r — Actual Revenue**: Revenue from Od level
$$r_t = \sum_k p_{t,k} \cdot \tilde{b}_{t,k} \in \mathbb{R}_+$$

**o^c — Observed Competitor Prices**: What we observe from competitors
$$o^c_{t,j} = P^{\text{c,obs}}_{t,j} \in \mathbb{R}_+$$

**Complete observation:**
$$\omega = (\tilde{b}, r, o^c) \in \Omega$$

---

### 1.5 Transition Model T: Od Demand Dynamics

The transition model T: S × A × S → [0, 1] describes how the latent Od state evolves:

**Transition decomposition:**

$$T(s_{t+1} | s_t, a_t) = T_D(D_{t+1}|D_t, P_t, P^c_t, M_t) \cdot T_C(C_{t+1}|C_t) \cdot T_T(\tau_{t+1}|\tau_t) \cdot T_M(m_{t+1}|m_t)$$

**1. Demand Transition (structural part):**
$$D_{t+1} = f_{demand}(D_t, P_{t+1}, P^c_{t+1}, M_{t+1}, \epsilon_D)$$

where:
- $P_t$ is our airline's pricing at time t
- $P^c_{t+1}$ is competitor pricing (exogenous or endogenous)
- $M_t$ includes seasonality, events, day-of-week
- $\epsilon_D$ is demand shock (e.g., Poisson, Normal)

**Structural demand specification (BLP-like):**

For Od-level, demand for fare class k follows:

$$D_{t,k} \sim \text{Poisson}(\lambda_{t,k})$$

$$\lambda_{t,k} = N_{t,k} \cdot \prod_{j \in \text{Competitors}_k} (1 - P_{sub}^{k,j}) \cdot \theta_k(P_{t,k}, P^c_{t+1,k}, X_{t,k})$$

where:
- $N_{t,k}$ is market size for class k
- $P_{sub}^{k,j}$ is substitution probability to competitor j
- $\theta_k$ is the demand generation function (e.g., logit-based)

**Price elasticity specification:**

$$\log(\lambda_{t,k}) = \alpha_k + \beta_k \log(P_{t,k}) + \gamma_k \log(P^c_{t,k}) + \delta_k X_{t,k}$$

where:
- $\beta_k$ is price elasticity for class k
- $\gamma_k$ is cross-price elasticity with competitors
- $\delta_k$ captures other factors (TTD, seasonality, etc.)

**2. Capacity Transition:**
$$C_{t+1} = \max(C_t - \tilde{b}_t, 0)$$
where $\tilde{b}_t$ is observed (censored) bookings at Od level.

**3. Time Transition:**
$$\tau_{t+1} = \tau_t - 1$$
$$\mathcal{T} \subset \mathbb{N}_0$$
where TTD decreases by one each day.

**4. Market Factor Transition:**
$$M_{t+1} = f_{market}(M_t, \text{seasonality}, \text{events}, \text{weather})$$

---

### 1.6 Observation Model O: Censoring + Noise

The observation model O: S × A → Ω → [0, 1] maps true latent state to observed outcomes:

**Observation process:**

$$\Omega(\omega_{t+1}|s_{t+1}, a_{t+1}) = \Omega_b(\tilde{b}_{t+1}|D_{t+1}, C_{t+1}) \cdot \Omega_r(r_{t+1}|\tilde{b}_{t+1}, a_{t+1}) \cdot \Omega_{comp}(o^c_{t+1}|a_{t+1})$$

**1. Censoring Model:**
$$\tilde{b}_{t,k} = \text{min}(D_{t,k}, C_t) \cdot \mathbb{I}(\text{seat\_available}_k)$$

This is the **key censoring mechanism** — the core challenge for Od-level world models.

**2. Revenue Observation:**
$$r_{t+1} = \sum_k p_{t+1,k} \cdot \tilde{b}_{t+1,k} + \epsilon_r$$
where $\epsilon_r$ is revenue measurement noise (typically negligible in airline context).

**3. Competitor Observation (if observed):**
$$o^c_{t,j} = P^{\text{c}}_{t,j} + \epsilon_c$$
where $\epsilon_c$ is observation noise (competitor prices may be observed with delay/noise).

---

### 1.7 Reward Function R

The reward function R: S × A × S → ℝ captures the Od-level objective:

**Immediate Od-level reward:**
$$R(s_t, a_t, s_{t+1}) = \sum_k p_{t,k} \cdot \tilde{b}_{t,k}$$

**Network coupling:**
$$R^{\text{network}} = R^{\text{Od}} - \sum_{s \in \text{Network}} \lambda_s \cdot C_{\text{shared}, s}$$

where $\lambda_s$ is shadow price of constraint s.

**Long-term objective:**
$$V_\pi(s_0) = \mathbb{E}_\pi \left[ \sum_{t=0}^{T_{max}} \gamma^t R(s_t, a_t) \right]$$

where γ is the discount factor (typically 1.0 for finite-horizon RM).

---

## 2. HYBRID APPROACH: STRUCTURAL + LEARNED

### 2.1 Hybrid WM Architecture

**Hybrid WM = (T_structural + T_learned, O_structural + C_learning)**

where:
- **T_structural**: Structural BLP-like demand function (priors)
- **T_learned**: Neural network learning residuals
- **O_structural**: Explicit censoring mechanism
- **C_learning**: Learned correction terms

**Demand model:**
$$D_{t+1}^{\text{hybrid}} = \lambda^{\text{Struct}}_{t+1} \cdot \exp(f^{\text{NN}}_{t+1}) + \epsilon$$

where:
- $\lambda^{\text{Struct}}_{t+1}$: Structural BLP demand prediction
- $f^{\text{NN}}_{t+1}$: Neural correction term
- $\epsilon$: Demand shock

**Advantages:**
- Structural priors ensure economic consistency
- Neural component captures misspecification
- Causal identification via structural IVs

---

### 2.2 Causal Identification at Od Level

**Identifiability challenge:** Price P and Demand D are jointly determined (simultaneity bias).

**Identifying assumption:**
$$Cov(P_t, \epsilon_{t,k}) = 0$$
where $\epsilon_{t,k}$ is the demand shock for class k.

**Instrumental variables approach:**
We need Z (instruments) such that:
1. $Z \perp \epsilon$ (Z is uncorrelated with demand shock)
2. $Z \rightarrow P$ (Z affects price)
3. Z only affects D through P

**Valid instruments at Od level:**
- Competitor costs (fuel, labor)
- Regulatory constraints
- Calendar events (independent of individual Od demand)
- Weather patterns (affect travel demand but not individual WTP)
- Competitor price (for cross-elasticity)

**Structural price-demand equation:**
$$D_k = \alpha_k + \beta_k P_k + \gamma_k P^c_k + \delta_k X_k + \epsilon_k$$

---

## 3. PURE-LEARNED APPROACH: LATENT WORLD MODEL

### 3.1 Pure-Learned WM Architecture

**Learned WM = (T_learned, O_learned + C_censoring)**

where:
- **T_learned**: Neural transition model P(z_{t+1}|z_t, a_t)
- **O_learned**: Neural observation model P(ω_t|z_t)
- **C_censoring**: Explicit censoring layer in observation model

**State representation:**
$$z_t = (z_D, z_C, z_\tau, z_{P_c}, z_M)$$

where each z component is learned via VAE/RNN encoder from observation history.

**Transition model:**
$$p(z_{t+1}|z_t, a_t) = \mathcal{N}(\mu_\theta(z_t, a_t), \sigma_\theta(z_t, a_t))$$

**Observation model with censoring:**
$$p(\omega_t|z_t) = \text{Poisson}(\tilde{D}_t) \cdot \mathbb{I}(\text{censoring})$$

where $\tilde{D}_t = f_O(z_D, z_C)$ models the censoring mechanism.

---

### 3.2 Censoring as Explicit Layer

**Critical design choice for pure-learned approach:**

The censoring must be modeled **explicitly** as a layer in the observation model, not learned implicitly:

$$\tilde{b}_t = \text{min}(f_{\text{demand}}(z_t), C_t)$$

**Why explicit?**
- Censoring is known to be exactly min(demand, capacity) in airline context
- Learning this from data would require knowing true demand (impossible!)
- Explicit censoring guarantees observational consistency

---

## 4. MARKOV PROPERTY VALIDATION AT OD LEVEL

### 4.1 Markov Property for Od State

**Theorem:** The Od-level state s contains all necessary information for optimal pricing, assuming:

1. Demand only depends on current P, P_c, M, and D
2. No long-term memory effects (beyond TTD)
3. Capacity evolves deterministically from bookings

**Proof sketch:**
State s = (D, C, T, P_c, M) satisfies Markov property because:

$$D_{t+1} = f(D_t, P_t, P_{t+1}^c, M_{t+1}, \epsilon_D)$$

Only depends on current state and current action, not past history given s.

### 4.2 Partial Observability in Od Context

**Observation ω is insufficient:**
We observe $\tilde{b}$ (censored bookings), not true D (latent demand).

This makes the problem **POMDP** not MDP.

**Belief state:**
$$b_t(s) = P(s_t = s|\omega_1, a_1, ..., \omega_t, a_t)$$

The belief over Od demand D is the key component of belief state.

---

## 5. NETWORK-WIDE COUPLING

### 5.1 Network Capacity Constraint

While the POMDP is defined at Od level, the optimization is network-wide:

**Network optimization:**
$$\max_{\pi} \mathbb{E} \left[ \sum_{od \in \text{Odds}} R_{od}(\pi) \right]$$

**Subject to:**
$$\sum_{od \in \text{Odds}} \sum_k \tilde{b}_{od,k} \leq C_{\text{aircraft}}$$

### 5.2 Bid Price Network Coupling

**Bid price approach (traditional):**
$$\lambda_s = \frac{\partial V}{\partial C_s}$$

where $\lambda_s$ is the shadow price of constraint s.

**Network coupling via:**
- Shared aircraft capacity across Ods
- Fleet constraints
- Hub connectivity

**Theoretical result:** Under certain conditions, network revenue max ≈ sum of Od-level revenue max with appropriate bid prices.

---

## 6. POMDP COMPLEXITY ANALYSIS

### 6.1 State Space Growth

| Od Level | Dimensions | Growth Rate |
|----------|------------|-------------|
| s_t | 7 dimensions | Polynomial |
| b_t (belief) | 2^7 = 128 | Exponential |
| Continuous state | ∞ | Functional |

### 6.2 Computational Complexity

- **Hybrid approach:** O(K·M·N) where K=fare classes, M=market dims, N=sample size
- **Pure-learned:** O(L·N·D) where L=network layers, N=training samples, D=feature dims
- **Network coupling:** O(|Odds|·|Aircraft|) for bid price updates

---

## 7. KEY ASSUMPTIONS & VALIDATION

### 7.1 Critical Assumptions

**A1: Od-level Markov assumption holds**
- Validated via: Historical demand autocorrelation analysis
- Risk: Demand may have long-term memory (brand loyalty, etc.)

**A2: Censoring is exactly min(demand, capacity)**
- In practice: Attrition/cancellations affect this
- Mitigation: Add attrition layer to observation model

**A3: Competitor prices are exogenous**
- Reality: Competitors react to our pricing
- Mitigation: Include endogenous competitor model in future

**A4: Capacity constraint is strictly binding**
- Often true in yield management context
- Validated via: Observed booking curves reaching capacity

### 7.2 Validation Strategy

1. **Posterior predictive check**: Does simulated bookings match observed?
2. **Elasticity validation**: Do estimated elasticities match ground truth?
3. **Causal validation**: Can we recover true price-demand relationship?
4. **Network validation**: Do Od-level policies optimize network revenue?

---

## 8. SUMMARY OF POMDP FORMALIZATION

| Component | Definition | Notes |
|-----------|------------|-------|
| **State S** | (D, C, T, P_c, M, X) | Od-level with full latent demand |
| **Action A** | (p_1, ..., p_K) | Od-level prices across fare classes |
| **Observation Ω** | (b, r, o_c) | Censored bookings, revenue, competitor prices |
| **Transition T** | T_D × T_C × T_T × T_M | Demand dynamics + capacity + time + market |
| **Observation Model** | O_b × O_r × O_c | Censoring + revenue + competitor observation |
| **Reward R** | Σ p·b | Od-level revenue |
| **Policy π** | Obs_hist → A | Pricing policy |
| **Network coupling** | Σ_bookings ≤ C | Capacity constraint |

---

## 9. NEXT STEPS

**Phase 3:** Architecture specs for hybrid vs. pure-learned approaches
**Phase 4:** Synthetic data generator design
**Phase 5:** Paper and experiments

---

## 10. REFERENCES

**POMDP Foundations:**
1. Kaelbling et al. (1998) — POMDPs survey
2. Littman, Dean & Kaelbling (1995) — On the complexity of POMDPs
3. Monahan (1982) — Survey of POMDPs

**Censoring in Demand:**
4. Benartzi et al. (2017) — Censoring in pricing
5. Kuby et al. (2005) — Censored demand modeling

**BLP Structural Demand:**
6. Berry, Levinsohn, Pakes (1995) — BLP model
7. Nevan & Robinson (2003) — BLP implementation
8. Dubé et al. (2008) — BLP with cross-elasticity

**Revenue Management:**
9. Talluri & van Ryzin (2004) — RM textbook
10. Zhang & Zhu (2013) — RL for RM
11. Gallego & van Ryzin (1997) — Dynamic pricing

**Causal Inference:**
12. Pearl (2009) — Causality
13. Imbens & Rubin (2015) — Causal inference
14. Athey & Wager (2019) — Causal ML

**World Models:**
15. Ha & Schmidhuber (2018) — World Models
16. Ha & Hafner (2018) — Dreamer
17. Hafner et al. (2023) — DreamerV3
18. Schrittwieser et al. (2020) — MuZero

---

*Document created: [current date]*
*Author: edtaylor*
*Repository: https://github.com/edtaylor/rmwm (pending)*
*Status: Phase 2 complete. Ready for Phase 3: Architecture specs.*
