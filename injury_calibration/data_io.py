"""
Load the hand-digitised Jarzina (2022) key-event dose-response CSVs.

Schema: concentration_uM, mean, sd  (one file per key event x cell line).
All values are % of control. Missing `sd` (empty or NaN) means that arm has no
usable error bars and must be fitted UNWEIGHTED -- see `Arm.weighted`.

Deliberately NOT here: any auto-digitisation. Digitisation is a human step; this
module only consumes what `data/` already contains. A missing file is skipped
gracefully rather than raising, so a partial `data/` still produces a partial run.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Viability is a % -of-control readout bounded to [0, 100]; digitised values can stray
# slightly negative at the floor (Jarzina's -0.2), which we clamp rather than fit.
VIABILITY_MIN, VIABILITY_MAX = 0.0, 100.0


@dataclass
class Arm:
    """One (key event x cell line) dose-response arm."""
    key_event: str        # "ke3_viability" | "ke1_lamp"
    cell_line: str        # "RPTEC/TERT1" | "NRK-52E"
    conc_uM: np.ndarray
    mean: np.ndarray
    sd: np.ndarray        # may be all-NaN
    source_file: str

    @property
    def weighted(self):
        """True if this arm has usable SDs for variance weighting (weight ~ 1/sd^2)."""
        return bool(np.isfinite(self.sd).all() and (self.sd > 0).all())

    @property
    def n_points(self):
        return int(self.conc_uM.size)

    def __repr__(self):
        w = "weighted" if self.weighted else "UNWEIGHTED (no usable SDs)"
        return f"<Arm {self.key_event} {self.cell_line}: {self.n_points} pts, {w}>"


# (file stem, key event, cell line). Cathepsin D (Jarzina KE2, Fig 5C) is deliberately
# absent: Jarzina themselves flag it as responding out of AOP order and express "concern
# regarding the cathepsin assay as a reliable marker for KE2". Do not add it.
ARM_SPECS = [
    ("ke3_viability_rptec", "ke3_viability", "RPTEC/TERT1"),
    ("ke3_viability_nrk",   "ke3_viability", "NRK-52E"),
    ("ke1_lamp_rptec",      "ke1_lamp",      "RPTEC/TERT1"),
    ("ke1_lamp_nrk",        "ke1_lamp",      "NRK-52E"),
]


def load_arm(stem, key_event, cell_line, data_dir=DATA_DIR):
    """Load one arm's CSV. Returns None (with a note printed) if the file is absent."""
    path = os.path.join(data_dir, f"{stem}.csv")
    if not os.path.exists(path):
        print(f"  [skip] {stem}.csv not present -- arm skipped")
        return None

    df = pd.read_csv(path, comment="#", skipinitialspace=True)
    missing = {"concentration_uM", "mean", "sd"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")

    df = df.sort_values("concentration_uM").reset_index(drop=True)
    conc = df["concentration_uM"].to_numpy(dtype=float)
    mean = df["mean"].to_numpy(dtype=float)
    sd = df["sd"].to_numpy(dtype=float)

    if key_event == "ke3_viability":
        # Bounded readout: clamp digitisation noise at the floor/ceiling.
        mean = np.clip(mean, VIABILITY_MIN, VIABILITY_MAX)

    if conc.size and not np.all(np.diff(conc) > 0):
        raise ValueError(f"{path}: concentrations must be strictly increasing after sort")
    if np.any(conc <= 0):
        raise ValueError(f"{path}: concentrations must be positive (log-axis fitting)")

    return Arm(key_event=key_event, cell_line=cell_line, conc_uM=conc,
               mean=mean, sd=sd, source_file=os.path.basename(path))


def load_all(data_dir=DATA_DIR):
    """Load every available arm. Returns {(key_event, cell_line): Arm}."""
    print(f"Loading key-event data from {data_dir}/")
    arms = {}
    for stem, ke, cl in ARM_SPECS:
        arm = load_arm(stem, ke, cl, data_dir=data_dir)
        if arm is not None:
            arms[(ke, cl)] = arm
            print(f"  {arm}")
    return arms


def load_payasi(data_dir=DATA_DIR):
    """
    Optional Payasi (2024) cross-check arm (marketed-PMB, RPTEC/hTERT1). Returns None
    if absent, which is the expected case -- this is a qualitative overlay only and is
    never co-fitted with Jarzina. See SUMMARY.md for the caveats that must accompany it.
    """
    return load_arm("payasi_viability_rptec", "ke3_viability", "RPTEC/TERT1",
                    data_dir=data_dir)
