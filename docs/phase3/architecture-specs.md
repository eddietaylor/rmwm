# Phase 3: Architecture Specifications — Od-Level World Models for RM

## Abstract

This document provides concrete architecture specifications for both the Hybrid (Structural + Learned) and Pure-Learned approaches to Od-level world models for airline revenue management. We compare architectures across implementation complexity, data requirements, computational cost, causal interpretability, and RM-specific performance metrics. Both architectures are designed to be implementation-ready for research and production deployment.

---

## 1. COMPARISION MATRIX

| Dimension | Hybrid (Structural + Learned) | Pure-Learned (Latent WM) |
|------|-------|-------|
| **Causal Guarantees** | ✅ Yes (structurally enforced) | ❌ No (learned correlations) |
| **Counterfactuals** | ✅ True counterfactual queries | ⚠️ Approximate via perturbation |
| **Data Requirements** | Small (structural priors guide) | Large (needs many episodes) |
| **Computational Cost** | Medium (BFGS + NN backprop) | High (RNN + attention) |
| **Interpretability** | High (economic theory) | Low (black-box) |
| **Generalization** | In-distribution (theory constrained) | Out-of-distribution (flexible) |
| **Censoring** | Explicit (by construction) | Explicit (enforced layer) |
| **RM-Specific Adaptation** | Direct (economic priors) | Indirect (learned from data) |
| **Network Coupling** | Via bid prices (analytical) | Via shared capacity (learned) |
| **Validation** | Structural econometrics metrics | RL evaluation metrics |
| **Theoretical Guarantees** | Convergence rates (MLE) | PAC bounds (if applicable) |

---

## 2. HYBRID APPROACH: DETAILED ARCHITECTURE

### 2.1 Architecture Overview

```
Hybrid Od-Level World Model = 
  [Structural Core] + [Neural Corridor]

Structural Core:
  D^struct = f_BLP(p, p^c, x, theta)  ← Blp demand function
  O^struct = min(D^struct, C)         ← Censoring (exact)

Neural Corridor:
  D^corr = NN(p, z, history, context) ← Learned residuals
  theta^adapt = NN(D_obs, p, ttd)     ← Adaptive elasticities

Final Model:
  D^final = D^struct * exp(D^corr)    ← Multiplicative correction
  C^censor = min(D^final, C)          ← Censoring

Policy:
  pi = O(Σ) + Bid_price_update        ← Network optimizer
```

### 2.2 Structural Demand Component

**OD-Level BLP Demand Model:**

For Od od with fare classes K={1, 2, ..., K}:

**Step 1: Utility computation**
$$U_{n,k} = \alpha_k p_{k} + \beta_k(p_{k}, p_{-k}) + x_{k} \gamma + \xi_{k} + \mu_{n,k}$$

where:
- $p_k$: price for class k at Od (action)
- $p_{-k}$: competitor prices at Od (from state)
- $x_k$: characteristics of class k (TTD, day-of-week, seasonality, etc.)
- $\xi_k$: unobserved class-level characteristics (random)
- $\mu_{n,k}$: individual-level random utility
- $\alpha_k$: base elasticity (structural prior)
- $\beta_k$: cross-price elasticity (structural prior)

**Step 2: Choice probability (Mixed Logit)**
$$P_{n,k} = \frac{\exp(U_{n,k})}{\sum_{j \in Od} \exp(U_{n,j})}$$

**Step 3: Aggregated demand**
$$\lambda_{k} = N \cdot \int_{\Theta} P_{n,k} \omega(\theta) d\theta$$

where $\omega(\theta)$ is the distribution of random coefficients, and N is market size.

**Step 4: Censoring**
$$b_k = \text{min}(\lambda_k, C_k)$$

This is the **structural censoring** — guaranteed to be physically consistent.

### 2.3 Neural Correction Component

**Residual Network Architecture:**

```python
class NeuralCorridor(nn.Module):
    """
    Learns residuals around structural BLP predictions.
    Captures misspecification in structural priors.
    """
    
    def __init__(self, n_features, hidden_dims=[64, 128, 64]):
        super().__init__()
        
        # Feature encoding
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0])
        )
        
        # Temporal processing (RNN)
        self.rnn = nn.GRU(hidden_dims[0], hidden_dims[1], batch_first=True)
        
        # Corridor computation
        self.corridor_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)  # Log-demand correction
        )
        
        # Adaptive elasticity head (optional)
        self.elasticity_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], K)  # Elasticity shift per class
        )
    
    def forward(self, features, history=None):
        """
        features: (batch, time, n_features)
        history: (batch, seq_len, embedding_dim)
        
        Returns:
        log_correction: (batch, K)
        elasticity_shift: (batch, K) (optional)
        """
        if history is not None:
            features = torch.cat([features, history], dim=-1)
        
        x = self.encoder(features)
        if history is not None:
            x, h = self.rnn(x)
            x = x[:, -1, :]  # Last time step
        
        log_correction = self.corridor_head(x)  # (batch, 1)
        
        return log_correction
```

**Feature vector:**

```
x = [
    log(price_od),           # Our Od price
    log(competitor_od),      # Competitor prices at Od
    ttd,                     # Time-to-departure
    day_of_week,             # One-hot
    seasonality,             # Fourier features for holidays
    competitor_cost,         # Competitor fleet cost index
    fuel_price,              # Current fuel price
    event_dummy,             # Event at Od location
    fare_class,              # Which fare class
    remaining_capacity_od,   # Available seats at Od
    booking_pattern,         # Historical booking curve
]
```

### 2.4 Training Strategy for Hybrid Approach

**Two-phase training:**

**Phase 1: Structural Estimation**

```python
# Estimate BLP parameters via simulated method of moments (SMM)
# or simulated maximum likelihood (SML)

def structural_loss(params, data):
    """
    params: BLP parameters (elasticities, costs, etc.)
    data: (p, D_observed, X, IVs)
    
    Returns: SMM moment conditions
    """
    predicted_lambda = blp_demand(params, data.p, data.X)
    predicted_observations = min(predicted_lambda, data.capacity)
    
    return predicted_observations  # Moments to match
```

**Phase 2: Neural Corridor Training**

```python
# Train neural corridor to match residuals after structural estimation

def corridor_loss(theta_corr, data):
    """
    theta_corr: Neural corridor parameters
    data: training data with observed bookings
    
    Returns: negative log-likelihood of residual
    """
    lambda_struct = blp_demand(structural_params, data.p)
    lambda_total = lambda_struct * exp(neural_correction(theta_corr))
    lambda_predicted = min(lambda_total, data.capacity)
    
    # Poisson likelihood of residual
    obs = data.bookings
    
    # Corrected distribution
    corrected_poisson = Poisson(lambda_predicted)
    return -corrected_poisson.log_prob(obs).mean()

# Training loop
for epoch in range(N_epochs):
    # Structural: update BLP params
    structural_params = minimize(structural_loss, initial_params)
    
    # Neural: update corridor params (constrained to be small)
    corridor_loss = corridor_loss(theta_corr, data)
    loss = corridor_loss + lambda_regularize * ||theta_corr||^2
    
    # Update θ with constrained steps (prevent drift from structural)
    update(theta_corr, -lr * gradient(corridor_loss))
```

**Regularization for structural integrity:**

```python
# Constrain neural corridor to be "small" relative to structural
L_total = L_corridor + L_lasso + L_spectral

# 1. Lasso on corridor coefficients (sparsity)
L_lasso = lambda_lasso * ||theta_corr||_1

# 2. Spectral normalization on corridor (prevent overfitting)
L_spectral = lambda_spectral * ||corridor_head[-1].weight||_spectral

# 3. Economic constraints (elasticities must be negative)
L_econ = lambda_econ * ReLU(elasticity)  # Penalize positive elasticities
```

### 2.5 Counterfactual Capability

**True counterfactual via do(·):**

```python
def counterfactual(p_new, structural_params, corridor_params):
    """
    Counterfactual query: "What if we had set price to p_new?"
    
    Returns: true demand D(p_new) with confidence interval
    """
    # Structural part (exact, via do-calculus)
    lambda_struct = do(blP_demand, {p: p_new}, params=structural_params)
    
    # Neural part (approximate via residual)
    lambda_corr = neural_correction(p_new, corridor_params)
    
    # Combined
    lambda_cf = lambda_struct * lambda_corr
    
    # Censoring (guaranteed correct)
    b_cf = min(lambda_cf, capacity)
    
    return b_cf
```

**Advantage:** The hybrid approach provides **guaranteed** causal identification at the Od level. The structural BLP part satisfies all causal assumptions (IV, exogeneity), and the neural corridor is only learning residuals (not the core causal mechanism).

---

## 3. PURE-LEARNED APPROACH: DETAILED ARCHITECTURE

### 3.1 Architecture Overview

```
Pure-Learned Od-Level World Model =
  [Latent State Encoder] → [Learned Transition] → [Learned Observation]
        ↓                       ↓                      ↓
    z_t (latent)          z_t+1 (predicted)     b_t (censored)
  
  Policy: pi(a_t | z_t) ← Actor-Critic on z_t
```

### 3.2 Architectural Components

**Component 1: Latent State Encoder**

```python
class OdStateEncoder(nn.Module):
    """
    Encodes observation history into latent state z.
    
    Input: history of (p, b, o^c) over timesteps
    Output: latent state z = (z_D, z_C, z_tau, z_Pc, z_M)
    """
    
    def __init__(self, n_obs_features, n_latent=128):
        super().__init__()
        
        # Feature embedding
        self.obs_embed = nn.Sequential(
            nn.Linear(n_obs_features, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Temporal encoder (Transformer/GRU)
        self.time_encoder = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Latent state projection
        self.state_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_latent),
            nn.LayerNorm(n_latent)
        )
    
    def forward(self, obs_history):
        """
        obs_history: (batch, seq_len, n_features)
        Returns: z_t (batch, n_latent)
        """
        x = self.obs_embed(obs_history)
        x, h = self.time_encoder(x)
        z = self.state_head(h[:, -1, :])  # Last hidden state
        return z
```

**Component 2: Learned Transition Model**

```python
class LearnableTransition(nn.Module):
    """
    Transition model T: z_t × a_t → z_t+1
    
    Learns dynamics purely from data (no structural priors).
    Uses residual learning for stability.
    """
    
    def __init__(self, n_z, n_actions):
        super().__init__()
        
        n = n_z + n_actions
        
        self.transition_net = nn.Sequential(
            nn.Linear(n, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 2 * n_z)  # Mean + log-variance
        )
        
        self.n_z = n_z
        self.n_actions = n_actions
    
    def forward(self, z_t, a_t):
        """
        z_t: (batch, n_z)
        a_t: (batch, n_actions)
        
        Returns: mu_t+1, sigma_t+1
        """
        x = torch.cat([z_t, a_t], dim=-1)
        out = self.transition_net(x)
        
        mu = out[:, :self.n_z]
        log_sigma = out[:, self.n_z:]
        
        return mu, log_sigma
    
    def sample(self, z_t, a_t):
        mu, log_sigma = self.forward(z_t, a_t)
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(log_sigma)
```

**Component 3: Censoring-Enabled Observation Model**

```python
class CensoredObservationModel(nn.Module):
    """
    Observation model with explicit censoring layer.
    
    This is the key innovation: censoring is built into the architecture,
    not just the loss function.
    """
    
    def __init__(self, capacity_dim, n_latent_demand):
        super().__init__()
        
        # Capacity projection (if not directly modeled)
        self.capacity_proj = nn.Linear(capacity_dim, 32)
        
        # Demand prediction
        self.demand_net = nn.Sequential(
            nn.Linear(n_latent_demand, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Predicted log-demand
            nn.Softplus()  # Ensure positive predictions
        )
        
        # Censoring layer (enforced min(demand, capacity))
        self.censor = CensoringLayer()
    
    def forward(self, z_t, capacity):
        """
        z_t: (batch, n_latent)
        capacity: (batch,)
        
        Returns: predicted bookings (censored)
        """
        # Predict true demand from latent state
        demand = self.demand_net(z_t)  # (batch, 1)
        
        # Apply censoring (enforced in architecture)
        bookings = self.censor(demand, capacity)  # min(demand, capacity)
        
        return bookings
```

```python
class CensoringLayer(nn.Module):
    """
    Enforced min(demand, capacity) layer.
    This ensures physical consistency of the world model.
    """
    
    def forward(self, demand: torch.Tensor, capacity: torch.Tensor) -> torch.Tensor:
        """
        demand: (batch, n_fare_classes)
        capacity: (batch,) or (batch, n_fare_classes)
        """
        return torch.min(demand, capacity)
```

**Component 4: Policy Network (Actor-Critic)**

```python
class OdPricePolicy(nn.Module):
    """
    Pricing policy pi: z_t → p_t (Od-level prices)
    
    Actor-Critic architecture for pricing optimization.
    """
    
    def __init__(self, n_z, n_outputs, hidden_dims=[256, 256]):
        super().__init__()
        
        # Actor (pricing policy)
        self.actor = nn.Sequential(
            nn.Linear(n_z, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], n_outputs),  # Pricing actions
            nn.Softplus()  # Ensure positive prices
        )
        
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(n_z, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)  # V(z_t)
        )
        
        # Price bounds
        self.p_min = 50.0  # $ minimum fare
        self.p_max = 2000.0  # $ maximum fare
    
    def forward(self, z_t):
        """
        z_t: (batch, n_latent)
        
        Returns: prices (batch, n_outputs)
        """
        p = self.actor(z_t)
        
        # Clip to valid price range
        p = torch.clamp(p, self.p_min, self.p_max)
        
        return p
    
    def value(self, z_t):
        """
        Critic value function V(z_t)
        """
        return self.critic(z_t)
```

### 3.3 Training Strategy for Pure-Learned Approach

**Dreamer-style learning (model-based RL):**

```python
class LearnedWorldModel(nn.Module):
    """
    Complete pure-learned world model (Dreamer-style).
    
    Training:
    1. Collect data from environment (real or synthetic)
    2. Train encoder and transition model
    3. Train observation model
    4. Optimize policy via RL on learned world model
    """
    
    def __init__(self, n_features, n_latent=128, n_actions=1):
        super().__init__()
        
        self.encoder = OdStateEncoder(n_features, n_latent)
        self.transition = LearnableTransition(n_latent, n_actions)
        self.observation = CensoredObservationModel(1, n_latent)
        self.policy = OdPricePolicy(n_latent, n_actions)
        
        self.n_latent = n_latent
    
    def forward(self, obs_history, prices):
        """
        Standard forward pass for training
        """
        z_t = self.encoder(obs_history)
        z_t1_mu, z_t1_logvar = self.transition(z_t, prices)
        predicted_bookings = self.observation(z_t1_mu, capacity)
        
        return z_t, z_t1_mu, z_t1_logvar, predicted_bookings
    
    def train_step(self, obs_history, true_bookings, capacity, prices):
        """
        Training step with all components
        """
        # Forward pass
        z_t, z_t1_mu, z_t1_logvar, pred_bookings = self(obs_history, prices)
        
        # Loss 1: Transition KLD (regularization toward prior)
        L_trans = KLD_gaussian(mu=z_t1_mu, sigma=torch.exp(z_t1_logvar), 
                               target_mu=z_t, target_sigma=torch.exp(z_t_logvar))
        
        # Loss 2: Observation loss (censored Poisson log-likelihood)
        L_obs = -observation_log_likelihood(true_bookings, pred_bookings)
        
        # Loss 3: Policy loss (actor-critic)
        z_t1_sample = z_t1_mu + torch.randn_like(z_t1_mu) * torch.exp(z_t1_logvar)
        v_next = self.policy.value(z_t1_sample)
        L_policy = -(prices * true_bookings - v_next)  # Bellman residual
        
        # Combined loss
        L_total = L_trans + L_obs + L_policy
        
        return L_total
```

### 3.4 Learning Dynamics (Dreamer-Style)

**The pure-learned approach follows the Dreamer paradigm:**

```
Learning Loop:
1. Collect episodes in environment (synthetic data)
2. Replay buffer stores (z_t, a_t, z_t1, r_t)
3. Train encoder: minimize reconstruction loss
4. Train transition: minimize KLD between pred and true
5. Train observation: maximize censored likelihood
6. Optimize policy: actor-critic on learned world model
7. Repeat until convergence
```

**Key differences from hybrid approach:**

| Aspect | Hybrid | Learned |
|-----|-----|---|
| Structural priors | Yes (BLP) | No (learned from data) |
| Censoring | Structural (guaranteed) | Learned + enforced layer |
| Elasticity estimation | Structural (identifiable) | Learned (correlated) |
| Counterfactuals | Exact (do-calculus) | Approximate |
| Training time | Shorter (priors help) | Longer (needs more data) |
| Generalization | In-distribution | Out-of-distribution |

---

## 4. CAPACITY AND BIDDING: NETWORK COUPLING

### 4.1 Bid Price Network Coupling

Both approaches need network coupling for multi-Od optimization:

**Bid Price Update (for both approaches):**

$$\Delta \lambda_s = \epsilon \cdot \sum_{od \in \text{ActiveOdds}} \text{shadow_price}_{od} \cdot \frac{\partial \text{bookings}_{od}}{\partial \lambda_s}$$

where $\lambda_s$ is the bid price for seat on route s.

**For Hybrid approach:**
- Bid prices are analytical (from BLP structural model)
- Updates via gradient of V(z_t)

**For Learned approach:**
- Bid prices are learned via policy gradient
- Updates via critic function

### 4.2 Network-Aware Architecture

```
Network-Coupled Architecture:

Hybrid:
  [BLP Structural Model] → [Bid Price Network] → [Price per Od]
  
Learned:
  [Latent Od States] → [Global Value Function] → [Global Prices]
```

Both architectures support network coupling, but the **pure-learned approach** may need additional mechanism to ensure capacity constraints are satisfied across the network.

---

## 5. TRAINING PHASES & DATA REQUIREMENTS

### 5.1 Hybrid Approach Training

**Phase 1: Structural Estimation**
```
Data needed: 10-50 OD-days of historical data
Method: Simulated Method of Moments (SMM)
Duration: 1-3 hours (depending on BLP model complexity)

Training:
- Estimate elasticity via IV (instrumental variable)
- Estimate BLP parameters via SML
- Validate with holdout Ods
```

**Phase 2: Neural Corridor**
```
Data needed: 100-1000 OD-days
Method: Gradient descent on corridor loss
Duration: 1-5 hours (depending on network size)

Training:
- Initialize with BLP structural parameters
- Train corridor to minimize residual
- Constrain corridor to be small (regularization)
- Validate on holdout Ods
```

**Data requirement per Od: 10-50 days** (shorter due to structural priors)

### 5.2 Pure-Learned Approach Training

**Phase 1: Data Collection**
```
Synthetic data needed: 10,000-100,000 OD-days
Real data: 100-1000 OD-days (for pre-training)
Method: GAN-based data augmentation (synthetic)

Data generation:
- Generate OD demand via BLP (for synthetic)
- Add realistic censoring
- Generate valid booking sequences
```

**Phase 2: Encoder/Transition Learning**
```
Data: synthetic data
Method: Variational inference (ELBO)
Duration: 2-6 hours (depending on network size)

Training:
- Train encoder to reconstruct history
- Train transition to predict next state
- Validate via transition error
```

**Phase 3: Policy Learning**
```
Data: learned world model trajectories
Method: Actor-critic RL
Duration: 2-4 hours (RL convergence)

Training:
- Rollout in learned world model
- Update policy via policy gradient
- Validate via revenue performance
```

**Data requirement: 10,000+ OD-days** (much longer due to no priors)

---

## 6. VALIDATION STRATEGIES

### 6.1 Structural Validation (Hybrid Approach)

**What to validate:**
1. Elasticity estimates match ground truth
2. Confidence intervals properly calibrated
3. Counterfactual predictions are accurate
4. Structural priors are honored

**Metrics:**
- Elasticity error: $| \hat{\beta} - \beta^{\text{true}} |$
- Counterfactual accuracy: $| D^{\text{cf}}_{\text{true}} - D^{\text{cf}}_{\text{pred}} |$
- CI coverage: $P(\beta^{\text{true}} \in [\hat{\beta} - 1.96\sigma, \hat{\beta} + 1.96\sigma])$

### 6.2 ML Validation (Pure-Learned Approach)

**What to validate:**
1. World model reproduces key patterns
2. Policy achieves near-optimal revenue
3. No structural violations (negative elasticity, etc.)
4. Generalization to out-of-sample Ods

**Metrics:**
- Booking MSE: $\text{MSE}(b_{\text{obs}}, b_{\text{pred}})$
- Revenue ratio: $\frac{\text{Revenue}_{\text{policy}}}{\text{Revenue}_{\text{oracle}}}$
- Generalization: $\frac{\text{Revenue}_{\text{new Od}}}{\text{Revenue}_{\text{train Od}}}$

---

## 7. COMPUTATIONAL COMPLEXITY

### 7.1 Hybrid Approach

| Component | Complexity | Memory |
|-----|---|-----|
| BLP estimation | O(N · M · K) | O(N · M) |
| Neural corridor | O(L · K) | O(K · L) |
| Counterfactual | O(L · K) | O(K) |
| Bid price update | O(|Odds| · |Routes|) | O(|Routes|) |

**L:** corridor network layers, **K:** fare classes, **N:** OD-days, **M:** features

### 7.2 Pure-Learned Approach

| Component | Complexity | Memory |
|-----|---|-----|
| Encoder | O(L · K) | O(K · L) |
| Transition | O(L · K · A) | O(K · L) |
| Observation | O(L · K) | O(K · L) |
| Policy | O(L · A) | O(K · A) |
| RL Rollout | O(T · N · L) | O(N · T) |

**T:** rollout length, **A:** action dim, **L:** network size

---

## 8. IMPLEMENTATION ROADMAP

### Phase 1: Baseline (Week 1-2)
- [ ] Implement OD-level BLP demand model
- [ ] Implement simple elastic-demand baseline
- [ ] Set up synthetic data generator

### Phase 2: Hybrid Architecture (Week 3-4)
- [ ] Implement BLP-structural core
- [ ] Implement neural corridor
- [ ] Train on synthetic Od data
- [ ] Validate structural priors

### Phase 3: Learned Architecture (Week 5-8)
- [ ] Implement latent encoder
- [ ] Implement learned transition
- [ ] Implement censored observation model
- [ ] Train on synthetic data (10k+ OD-days)
- [ ] Tune learning rates and architecture

### Phase 4: Policy Learning (Week 9-10)
- [ ] Implement Actor-Critic policy
- [ ] Train on learned Od world model
- [ ] Compare hybrid vs. learned policies
- [ ] Validate on synthetic data

### Phase 5: Network Coupling (Week 11-12)
- [ ] Implement bid price network coupling
- [ ] Extend both approaches to network level
- [ ] Validate network revenue optimization

### Phase 6: Paper Draft (Week 13-14)
- [ ] Compile results
- [ ] Write paper for AGIFORS + NeurIPS
- [ ] Submit to both venues

---

## 9. KEY DESIGN DECISIONS

Both approaches make critical design choices at the Od level:

### Decision A: State Representation

**Hybrid:** Explicit OD state (D, C, T, P_c, M, X)
- Pros: Structured, interpretable, causal
- Cons: May miss latent dynamics

**Learned:** Latent state z = (z_D, z_C, z_T, ...)
- Pros: Flexible, captures hidden dynamics
- Cons: Black-box, no causal guarantees

### Decision B: Elasticity Estimation

**Hybrid:** Structural elasticity (identifiable)
- Pros: Causal, interpretable
- Cons: Requires valid instruments

**Learned:** Learned elasticity (via gradient)
- Pros: Flexible, no IV needed
- Cons: May learn spurious correlations

### Decision C: Censoring Mechanism

**Hybrid:** Structural censoring (min(demand, capacity))
- Pros: Physically consistent by construction
- Cons: May not match reality perfectly

**Learned:** Enforced censored layer
- Pros: Flexible censoring function
- Cons: Requires careful training

### Decision D: Network Coupling

**Hybrid:** Global bid price optimization
- Pros: Analytical, theoretically sound
- Cons: May be suboptimal in complex networks

**Learned:** Learned network value function
- Pros: Learns complex interactions
- Cons: Hard to validate and explain

---

## 10. SUMMARY OF ARCHITECTURES

Both approaches are implemented-ready and target Od-level revenue management. The key differences are:

**When to use Hybrid:**
- ✅ Limited data (10-50 OD-days)
- ✅ Causal interpretation required
- ✅ Counterfactual queries needed
- ✅ Industry deployment (auditable)

**When to use Pure-Learned:**
- ✅ Abundant data (10k+ OD-days)
- ✅ Maximum performance (out-of-distribution)
- ✅ Highly non-linear interactions
- ✅ Complex Od-level dynamics

**Recommendation for RM:** Start with **hybrid** approach (smaller data requirement, auditability) and scale to **pure-learned** as data grows.

---

## 11. COMPARISON TO INDUSTRY PRACTICE

| Industry Approach | Hybrid WM | Learned WM |
|---|---|---|
| **Bid price** | Analytical (BLP) | Learned (neural) |
| **Forecasting** | Separate module | Integrated (encoder) |
| **Elasticity** | Estimated separately | Learned end-to-end |
| **Capacity booking** | Separate optimization | Integrated via censoring |
| **Network coupling** | Via bid prices | Via learned value |
| **Counterfactuals** | Exact (do-calculus) | Approximate |
| **Complexity** | 100-500 LOC | 500-1000 LOC |
| **Training time** | 1-3 hours | 10-30 hours |
| **Data needed** | 10-50 OD-days | 10k+ OD-days |
| **Industry adoption** | High (familiar) | Low (new paradigm) |

---

*Document created: [current date]*
*Author: edtaylor*
*Repository: https://github.com/edtaylor/rmwm (pending)*
*Status: Phase 3 complete. Ready for Phase 4: Synthetic data generator design.*
