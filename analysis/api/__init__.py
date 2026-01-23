"""
REST API for Intervention Analysis System

Provides endpoints for:
- Individual analysis
- Intervention recommendations
- Counterfactual simulations
- Risk assessments
"""
from .main import app

__all__ = ['app']
