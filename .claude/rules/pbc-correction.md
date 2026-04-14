# PBC Correction

All RMSD checks in OpenMMRunner MUST use minimum image convention
(PBC-corrected distances).

Without it, periodic boundary artifacts produce RMSD values of ~100 A,
causing false binding-failure verdicts.
