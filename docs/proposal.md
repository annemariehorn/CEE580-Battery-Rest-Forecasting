# Forecasting Battery Capacity Fade and End of Life: The Role of Rest Periods

> Final proposal as submitted (Oct 9, 2026). Score 90/100; see feedback/proposal-feedback.md.
> Names filled in from the role assignment. Changes since submission are logged in decisions.md.

**Group members:** Andrew Geyser, Annemarie Horn, Emi Sohi | **Group name:** Lithium Learners

## 1. Background and Motivation

Lithium-ion batteries lose capacity with use, and predicting how that fade continues, and when a cell reaches end of life, guides maintenance and warranty decisions. Aging tests take months to years (Attia et al., 2020), so prediction from early cycles is valuable (Severson et al., 2019). Data-driven models forecast capacity well over short horizons (Richardson et al., 2017).

Many data-driven forecasts use capacity history alone. Yet after a long rest, a cell temporarily recovers some lost capacity (capacity regeneration; Deng et al., 2017), which such forecasts treat as noise. This project tests whether rest information improves longer-range forecasts on an unseen cell, and whether past or planned rests are needed.

## 2. Data and Outcome

We will use four cells (B0005–B0007, B0018) from the free NASA Prognostics Center of Excellence Battery Data Set (Saha & Goebel, 2007): commercial 18650 cells rated at 2.0 Ah, cycled at 24°C (1.5 A charge to 4.2 V, 2 A discharge).

- **Unit of analysis:** a capacity forecast for one cell, made at a given cycle (636 discharge cycles; 132–168 per cell).
- **Outcome:** discharge capacity (Ah) at future cycles (primary), and the first cycle below the end-of-life threshold (secondary).
- **Key inputs:** cycle number, capacity history, rest time before each cycle; exploratory: temperature, charging time, impedance.

Preliminary analysis of the data shows:

- **Long rests precede capacity jumps.** Each cell shows 7–11 jumps of up to 150 mAh, mostly after unusually long rests (median 9–34 hours vs. about 5). Trained on B0005–B0007, a linear model with log rest time explains 79% of B0018's cycle-to-cycle capacity change; cycle number explains none.
- **Only one cell is independent.** B0005–B0007 were cycled simultaneously (capacity correlation 0.98–0.998); B0018 was cycled separately, in July–August 2008.
- **The official threshold excludes a cell.** B0007 never reaches 1.4 Ah (30% fade; minimum 1.4005 Ah), so we will use 1.5 Ah (25% fade), which all four cells cross (first crossings at cycles 70–126).

Limitations: four cells, one temperature; the full NASA set (over 30 cells) is an extension.

## 3. Scientific Question and Hypothesis

**Question:** How accurately can a cell's future capacity be forecast from its early capacity fade, and does accounting for rest periods improve those forecasts on a cell the model never saw?

**Hypothesis (H1):** Knowing the planned rest schedule cuts capacity-forecast RMSE 20–50 cycles ahead by at least 10% for both ridge regression and boosted trees, while past rest history alone does not, because current capacity already reflects past rests. A preliminary check on B0018 found about 20% for ridge but no gain for boosted trees; we test which holds across tuned models, starting cycles and held-out cells.

**Competing explanation (H0):** With only three training cells, rest inputs add noise, and the preliminary gain disappears under other models or held-out cells.

**Why it matters:** It shows whether battery forecasts should use planned usage schedules.

## 4. Planned Analysis

**Approach.** Forecasts are issued at cycles 30, 45 and 60 (all before B0018's end of life) using only data up to that cycle, and predict capacity 1–50 cycles ahead. Ridge regression and boosted trees are each fit in three versions: (a) no rest information, (b) past rest times, and (c) past plus planned future rest times (test schedules are set in advance). Comparing (c) with (a) and (b) tests H1 against H0.

1. Baselines: persistence (capacity unchanged), and linear and exponential trends fit to each cell's own history.
2. Ridge regression and gradient-boosted trees using current capacity, recent fade rate, forecast horizon and rest features.
3. Exploratory: the same models with temperature, charging time and impedance added as inputs (the extra signals).

**Validation.** Train on B0005–B0007 and test on B0018; each other cell is also held out in turn (optimistic: shared rest schedule). Cycles are never split randomly.

**Evaluation.** Primary: root-mean-square error (RMSE, mAh) 20–50 cycles ahead, with block-bootstrap confidence intervals. Exploratory: end-of-life error (forecasts extended to 100 cycles), whether the extra signals improve forecasts, and how many cycles post-rest gains persist.

## 5. Feasibility and Work Plan

The data (about 55 MB) load in seconds and all models run on a laptop. The main risk is the small number of cells, addressed by case-study framing. If long-horizon forecasts prove unstable, we will narrow to forecasts 1–20 cycles ahead.

**First milestone.** A shared per-cycle table with rest times, and exploratory figures. Subsequent steps: baselines; ridge and boosted trees for the progress report; then the exploratory analyses.

**Division of labor.** All three members first build the per-cycle table, exploratory figures and shared forecasting and scoring code. Each then owns one area, including its tests and figures. Annemarie Horn: ridge regression, with and without the extra signals, and bootstrap intervals. Emi Sohi: boosted trees, with and without the extra signals. Andrew Geyser: baselines, the post-rest gain analysis and end-of-life prediction. All members review code and share the writing and presentation.

## 6. References

Attia, P. M., Grover, A., Jin, N., Severson, K. A., Markov, T. M., Liao, Y.-H., Chen, M. H., Cheong, B., Perkins, N., Yang, Z., Herring, P. K., Aykol, M., Harris, S. J., Braatz, R. D., Ermon, S., & Chueh, W. C. (2020). Closed-loop optimization of fast-charging protocols for batteries with machine learning. *Nature, 578*(7795), 397–402. <https://doi.org/10.1038/s41586-020-1994-5>

Deng, L.-M., Hsu, Y.-C., & Li, H.-X. (2017). An improved model for remaining useful life prediction on capacity degradation and regeneration of lithium-ion battery. *Annual Conference of the PHM Society, 9*(1). <https://doi.org/10.36001/phmconf.2017.v9i1.2438>

Richardson, R. R., Osborne, M. A., & Howey, D. A. (2017). Gaussian process regression for forecasting battery state of health. *Journal of Power Sources, 357*, 209–219. <https://doi.org/10.1016/j.jpowsour.2017.05.004>

Saha, B., & Goebel, K. (2007). *Battery data set*. NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA. <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>

Severson, K. A., Attia, P. M., Jin, N., Perkins, N., Jiang, B., Yang, Z., Chen, M. H., Aykol, M., Herring, P. K., Fraggedakis, D., Bazant, M. Z., Harris, S. J., Chueh, W. C., & Braatz, R. D. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy, 4*(5), 383–391. <https://doi.org/10.1038/s41560-019-0356-8>
