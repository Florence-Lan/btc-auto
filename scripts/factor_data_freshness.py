"""Diagnostics mirror the frozen oil entry gate's unchanged seven-day limit.

Keep this outside multifactor.py: its whole-file fingerprint belongs to the
frozen forward strategy and must not change for collection diagnostics.
"""

OIL_MAX_AGE_MS = 7 * 86_400_000
