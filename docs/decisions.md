# Design decisions

Each entry: what we decided, why, and when. Add to this as the project develops;
it feeds directly into the "Preprocessing" and "Methods" sections of the reports.

| Date | Decision | Reason |
|---|---|---|
| Oct 9 | Use four NASA cells (B0005, B0006, B0007, B0018) | Free, documented, clean; rest gaps are visible in the timestamps |
| Oct 9 | B0018 is the main test battery | B0005–B0007 were cycled side by side with nearly identical schedules (capacity correlation 0.98–0.998); only B0018 is independent |
| Oct 9 | End of life = first measurement below 1.5 Ah (25% fade) | B0007 never reaches NASA's 1.4 Ah criterion (minimum 1.4005 Ah); capacity can bounce back after rests, so use the first crossing (B0006 recovers above 1.5 Ah until measurement 92) |
| Oct 9 | Forecast origins 30, 45, 60; horizons 1–50; score horizons 20–50 | All origins precede B0018's end of life (measurement 70); 20–50 tests longer-range forecasting, where rest effects might matter |
| Oct 9 | Dropped the Gaussian process | Unstable and overconfident in a pilot with four batteries; not needed for the hypothesis |
| Nov | Rest time = gap since previous discharge **ended**, minus recorded operation durations | The first approach (gap between discharge starts) counted charging time as rest. Impedance tests have no recorded duration, so the estimate is an upper bound |
| Nov | Future rest treated as an **oracle** input | Recorded rest times were not necessarily planned in advance; the oracle measures the most that rest information could help |
| Nov 14 | Models predict the capacity *change* from the origin, then add current capacity back | Batteries start at different capacities (1.86–2.04 Ah); the change transfers across batteries better than the level |
| Nov 14 | Trend baselines fit the last 20 measurements | Long enough to average out single jumps; short enough to follow recent fade. Sensitivity to this window still to be checked |
| Nov 26 | Ridge alpha = 3000 for all three settings | Lowest mean RMSE holding out each of B0005–B0007 in turn (57.8 mAh; alpha = 1000 gives 57.9, so the choice is not sensitive). One alpha for all settings so they differ only in rest inputs. B0018 not used |
| Nov 26 | Ridge uses log(1 + rest hours); alpha = 1000 | With raw hours, one 241 h rest dominates a linear model, and the proposal found log rest time explains 79% of B0018's cycle-to-cycle change. Chosen on B0005–B0007 only (mean validation RMSE across settings 55.1 vs 57.8 mAh for raw hours). B0018 had already been scored once with raw hours (alpha = 3000), so both B0018 results are reported (`ridge_raw_rest` and `ridge`) |
