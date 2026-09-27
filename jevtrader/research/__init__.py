"""Research harness: loaders for the cached real-data CSVs, a vectorized fast-screen that shares
the SimBroker's cost assumptions, reproducible experiments, and record/replay tooling for real
quotes/books captured during paper trading.

Nothing in here is imported by the live runner; it exists to *validate* strategies before they
are promoted. See docs/STRATEGIES.md and research_reports/ for what the harness has found.
"""
