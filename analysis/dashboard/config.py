"""
Dashboard Configuration and Constants
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "empirical_study"

# Color Schemes (matching existing visualization.py)
ANIMAL_COLORS = {
    'Seeking': '#2ecc71',
    'Directing': '#e74c3c',
    'Conferring': '#3498db',
    'Revising': '#9b59b6'
}

PRIMARY_COLORS = {
    'COMPLEX': '#8e44ad',
    'FOCUSED': '#c0392b'
}

SUBTYPE_COLORS = {
    'Chameleon': '#9b59b6',
    'Multi-Modal': '#8e44ad',
    'Pure Predator': '#c0392b',
    'Strong Escalator': '#e74c3c',
    'Stalker-Striker': '#e67e22',
    'Fantasy-Actor': '#f39c12',
    'Standard': '#95a5a6'
}

ROLE_COLORS = {
    'HUB': '#e74c3c',
    'SOURCE': '#f39c12',
    'SINK': '#3498db',
    'GENERAL': '#95a5a6'
}

RISK_COLORS = {
    'CRITICAL': '#c0392b',
    'HIGH': '#e74c3c',
    'UNPREDICTABLE': '#8e44ad',
    'MODERATE': '#f39c12'
}

# Non-Technical Terminology Mapping
TERMINOLOGY = {
    'transfer_entropy': 'Pattern Similarity',
    'source': 'Pattern Originator',
    'sink': 'Pattern Inheritor',
    'hub': 'Central Connector',
    'lineage': 'Influence Chain',
    'outgoing_influence': 'Patterns Shared',
    'incoming_influence': 'Patterns Received'
}

# Risk Descriptions
RISK_DESCRIPTIONS = {
    'Pure Predator': 'CRITICAL - Highest danger. 76%+ probability of staying in exploitation mode.',
    'Fantasy-Actor': 'CRITICAL - Fast escalation. Short window for intervention.',
    'Strong Escalator': 'HIGH - Escalation is the key warning sign. Early detection critical.',
    'Stalker-Striker': 'HIGH - Surveillance phase offers detection opportunity.',
    'Chameleon': 'UNPREDICTABLE - No consistent pattern. Can switch modes rapidly.',
    'Multi-Modal': 'MODERATE-HIGH - Some pattern variability.',
    'Standard': 'HIGH - Typical danger profile.'
}

# Subtype Descriptions
SUBTYPE_DESCRIPTIONS = {
    'Pure Predator': 'Overwhelming dominance of exploitation behavior (75%+)',
    'Strong Escalator': 'Dramatic increase in Directing behavior over time',
    'Stalker-Striker': 'Methodical observation followed by calculated action',
    'Fantasy-Actor': 'Direct leap from internal fantasy to violent action',
    'Chameleon': 'Utilizes all four behavioral modes fluidly',
    'Multi-Modal': 'Multiple active states without full chameleon flexibility',
    'Standard': 'Typical Directing-dominant pattern'
}

# Primary Type Descriptions
PRIMARY_DESCRIPTIONS = {
    'COMPLEX': 'Multi-modal behavioral pattern with unpredictable state transitions',
    'FOCUSED': 'Directing-dominant pattern with clear escalation trajectory'
}
