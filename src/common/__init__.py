"""
Common utilities shared across models.
"""

# Import only non-torch dependencies by default
# Other imports are available but not imported by default to avoid dependency issues

__all__ = ['TrafficDataset', 'get_transforms', 'load_config', 'save_model', 'load_model', 'calculate_metrics']