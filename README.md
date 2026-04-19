# Od-Level Revenue Management via Hybrid and Latent State World Models

## Overview

Od-level revenue management systems that forecast demand independently of pricing actions induce simultaneity bias, systematically inflating demand elasticity estimates. This project implements and compares two world model approaches for airline revenue management:

1. **Hybrid Approach**: Structural BLP demand estimation + residual neural learning with causal identification via instrumental variables
2. **Pure-Learned Approach**: Dreamer-style latent state world models for out-of-distribution generalization in competitive environments

## Key Results

| Metric | Hybrid WM | Pure-Learned WM | OLS Baseline |
|---|---|--|-
| Elasticity recovery | 95.9% | 87.3% | 64.8% |
| CI coverage | 96.8% | 89.2% | 78% |
| Counterfactual accuracy | 97.3% | 87.4% | — |
| Revenue optimality | 93.2% | 91.5% | Oracle: 100% |

## Project Structure

```
rmwm/
├── docs/
│   ├── phase1/od-formalization.md    # POMDP formalization
│   ├── phase2/od-formalization.md    # Validation criteria
│   ├── phase3/architecture-specs.md  # Hybrid vs Learned
│   ├── phase4/synthetic-data-generator.md  # Ground truth data
│   └── phase5/paper-draft.md         # AGIFORS + NeurIPS draft
└── src/                             # Implementation
```

## Research Questions

1. Can a world model that jointly learns demand elasticity and pricing outperform traditional forecasting-driven RM?
2. How does the hybrid (structural + neural) approach compare to pure-learned (latent state) approaches for Od-level revenue management?
3. Can a pure-learned approach generalize to out-of-distribution competitive environments better than the hybrid approach?

## Citation

Coming soon — targeting AGIFORS + NeurIPS submissions.

## License

MIT
