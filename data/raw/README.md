# Raw data: NASA PCoE Battery Data Set (cells B0005, B0006, B0007, B0018)

| File | Battery | Discharge cutoff | Discharge measurements |
|---|---|---|---|
| `B0005.mat` | #5 | 2.7 V | 168 |
| `B0006.mat` | #6 | 2.5 V | 168 |
| `B0007.mat` | #7 | 2.2 V | 168 |
| `B0018.mat` | #18 | 2.5 V | 132 |

**What the experiment did.** Four commercial 18650 lithium-ion cells (rated 2.0 Ah)
were repeatedly charged (1.5 A constant current to 4.2 V, then constant voltage until
current fell to 20 mA), discharged at 2 A, and periodically given impedance tests,
at room temperature, until capacity faded by about 30% (2.0 Ah to 1.4 Ah).
See `NASA_README.txt` for NASA's own description and the file structure.

**Source.** NASA Ames Prognostics Center of Excellence (PCoE) Data Set Repository,
"5. Battery Data Set":
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
Direct download of the full set (includes many other cells we do not use):
https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

The copies here were taken from a public GitHub mirror of these four files
(https://github.com/XiuzeZhou/RUL, `datasets/NASA/`). `SHA256SUMS` lets you check that
your copy is identical: `sha256sum -c SHA256SUMS` (macOS: `shasum -a 256 -c SHA256SUMS`).

**Citation.** Saha, B., & Goebel, K. (2007). *Battery data set*. NASA Prognostics Data
Repository, NASA Ames Research Center, Moffett Field, CA.

Do not edit these files. All processing happens in `src/battery_forecast/data.py`.
