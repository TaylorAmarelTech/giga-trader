"""
GIGA TRADER - Data Manager with Caching (SHIM)
========================================
This module has been moved to:
  - src.phase_01_data_acquisition.data_manager

This file re-exports all public names for backward compatibility.
"""

from src.phase_01_data_acquisition.data_manager import (
    DataManager,
    DATA_CONFIG,
    get_data_manager,
    get_spy_data,
    _data_manager,
)

__all__ = [
    "DataManager",
    "DATA_CONFIG",
    "get_data_manager",
    "get_spy_data",
]

if __name__ == "__main__":
    from src.phase_01_data_acquisition.data_manager import __name__ as _  # noqa: F811
    # Delegate to the moved module's __main__ block
    import sys
    print("=" * 60)
    print("GIGA TRADER - Data Manager")
    print("=" * 60)

    dm = DataManager()

    # Check cache status
    print("\n[Cache Status]")
    info = dm.get_cache_info("SPY")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Check if update needed
    needs_update, reason = dm.needs_update("SPY")
    print(f"\n[Update Needed] {needs_update} - {reason}")

    # Get data (cached or download)
    df = dm.get_data("SPY", years=5)

    if df is not None and len(df) > 0:
        # Validate
        validation = dm.validate_data(df)
        print(f"\n[Validation]")
        print(f"  Valid: {validation['valid']}")
        if validation['issues']:
            for issue in validation['issues']:
                print(f"  Issue: {issue}")
        print(f"  Stats: {validation['stats']}")
