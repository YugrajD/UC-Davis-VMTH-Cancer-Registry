"""Training: recipe (production hyperparameters + seeds), labels (targets from any
labels table), one trainer per head (backbone, case_presence, group, label_presence),
and oof (k-fold heads-only out-of-fold predictions). Calibration is a separate
package (WP5b), not this one.
"""
