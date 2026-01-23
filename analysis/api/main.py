"""
FastAPI REST API for Intervention Analysis

Run with:
    uvicorn api.main:app --reload --port 8080

Or:
    python -m api.main
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
from collections import Counter
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.data.loader import DashboardDataLoader, STATE_NAMES

# Import intervention module
try:
    from intervention import (
        BehavioralSCM,
        TrajectoryAnalyzer,
        InterventionOptimizer,
        ClinicalReportGenerator,
        INTERVENTION_PROTOCOLS,
        get_protocol,
        RiskLevel
    )
    INTERVENTION_AVAILABLE = True
except ImportError as e:
    INTERVENTION_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Initialize FastAPI app
app = FastAPI(
    title="Intervention Analysis API",
    description="""
    REST API for behavioral trajectory analysis and intervention planning.

    Features:
    - Individual trajectory analysis
    - Risk assessment
    - Intervention recommendations
    - Protocol catalog
    - Population statistics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "empirical_study"

# Global data loader (lazy initialized)
_data_loader = None


def get_data_loader() -> DashboardDataLoader:
    """Get or create data loader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DashboardDataLoader(DATA_DIR)
    return _data_loader


# ============= Pydantic Models =============

class StateDistribution(BaseModel):
    seeking: float = Field(..., ge=0, le=1)
    directing: float = Field(..., ge=0, le=1)
    conferring: float = Field(..., ge=0, le=1)
    revising: float = Field(..., ge=0, le=1)


class RiskAssessmentResponse(BaseModel):
    level: str
    score: float
    factors: Dict[str, float]
    trajectory_indicators: List[str]


class InterventionRecommendation(BaseModel):
    protocol_name: str
    display_name: str
    expected_benefit: float
    cost: float
    urgency: float
    rationale: str
    time_index: int


class IndividualSummary(BaseModel):
    name: str
    n_events: int
    classification: Dict[str, str]
    dominant_state: str
    state_distribution: Dict[str, float]


class IndividualAnalysis(BaseModel):
    name: str
    n_events: int
    trajectory: List[str]
    state_distribution: Dict[str, float]
    classification: Dict[str, str]
    risk_assessment: Optional[RiskAssessmentResponse]
    mfpt_to_directing: float
    n_critical_transitions: int
    n_intervention_windows: int
    recommendations: List[InterventionRecommendation]


class ProtocolInfo(BaseModel):
    name: str
    display_name: str
    category: str
    description: str
    cost_range: Dict[str, float]
    duration_range: Dict[str, str]
    effectiveness: Dict[str, float]
    applicable_states: List[str]


class PopulationStats(BaseModel):
    n_individuals: int
    risk_distribution: Dict[str, int]
    avg_state_distribution: Dict[str, float]
    avg_mfpt_to_directing: float
    classification_distribution: Dict[str, Dict[str, int]]


class SimulationRequest(BaseModel):
    individual_name: str
    intervention_time: int = Field(..., ge=1)
    intervention_strength: float = Field(0.3, ge=0.1, le=0.8)
    n_simulations: int = Field(30, ge=10, le=100)


class SimulationResult(BaseModel):
    original_directing_pct: float
    simulated_directing_pct: float
    simulated_directing_std: float
    potential_reduction_pct: float
    sample_trajectory: List[str]


# ============= API Endpoints =============

@app.get("/")
def root():
    """API root endpoint."""
    return {
        "name": "Intervention Analysis API",
        "version": "1.0.0",
        "status": "running",
        "intervention_available": INTERVENTION_AVAILABLE,
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    loader = get_data_loader()
    n_individuals = len(loader.get_individuals_with_trajectory_data())
    return {
        "status": "healthy",
        "intervention_module": INTERVENTION_AVAILABLE,
        "individuals_available": n_individuals,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/individuals", response_model=List[IndividualSummary])
def list_individuals():
    """List all individuals with trajectory data."""
    loader = get_data_loader()
    individuals = loader.get_individuals_with_trajectory_data()

    results = []
    for name in individuals:
        try:
            traj = loader.load_individual_trajectory(name)
            ind_data = loader.get_individual_data(name)
            classification = ind_data.get('classification', {})

            # Compute state distribution
            counts = Counter(traj)
            total = len(traj)
            state_dist = {s: counts.get(s, 0) / total for s in STATE_NAMES}

            results.append(IndividualSummary(
                name=name,
                n_events=len(traj),
                classification={
                    'primary_type': classification.get('primary_type', 'Unknown'),
                    'subtype': classification.get('subtype', 'Unknown')
                },
                dominant_state=max(state_dist.items(), key=lambda x: x[1])[0],
                state_distribution=state_dist
            ))
        except Exception:
            continue

    return results


@app.get("/individuals/{name}", response_model=IndividualAnalysis)
def get_individual_analysis(
    name: str,
    include_recommendations: bool = Query(True, description="Include intervention recommendations")
):
    """Get detailed analysis for an individual."""
    if not INTERVENTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Intervention module not available")

    loader = get_data_loader()
    individuals = loader.get_individuals_with_trajectory_data()

    if name not in individuals:
        raise HTTPException(status_code=404, detail=f"Individual '{name}' not found")

    try:
        trajectory = loader.load_individual_trajectory(name)
        transition_matrix = loader.load_individual_transition_matrix(name)
        ind_data = loader.get_individual_data(name)
        classification = ind_data.get('classification', {})

        # Compute state distribution
        counts = Counter(trajectory)
        total = len(trajectory)
        state_dist = {s: counts.get(s, 0) / total for s in STATE_NAMES}

        # Create analysis components
        scm = BehavioralSCM(transition_matrix, STATE_NAMES)
        analyzer = TrajectoryAnalyzer(transition_matrix=transition_matrix)

        # Run analysis
        analysis = analyzer.comprehensive_analysis(trajectory)

        # Generate report for risk assessment
        report_gen = ClinicalReportGenerator(transition_matrix, STATE_NAMES)
        archetype = classification.get('subtype', 'Unknown')
        report = report_gen.generate_individual_report(
            individual_id=name,
            trajectory=trajectory,
            archetype=archetype,
            include_retrospective=False  # Skip compute-intensive analysis
        )

        # Get recommendations if requested
        recommendations = []
        if include_recommendations:
            optimizer = InterventionOptimizer(scm)
            opt_result = optimizer.find_optimal_timing(
                trajectory=trajectory,
                budget_constraint=20000,
                max_interventions=5
            )

            if opt_result and opt_result.recommendations:
                for rec in opt_result.recommendations[:5]:
                    recommendations.append(InterventionRecommendation(
                        protocol_name=rec.protocol.name,
                        display_name=rec.protocol.display_name,
                        expected_benefit=rec.expected_benefit,
                        cost=rec.cost,
                        urgency=rec.urgency,
                        rationale=rec.rationale or "",
                        time_index=rec.time_index
                    ))

        # Build risk assessment response
        risk_response = None
        if report.risk_assessment:
            risk_response = RiskAssessmentResponse(
                level=report.risk_assessment.level.value,
                score=report.risk_assessment.score,
                factors=report.risk_assessment.factors,
                trajectory_indicators=report.risk_assessment.trajectory_indicators
            )

        return IndividualAnalysis(
            name=name,
            n_events=len(trajectory),
            trajectory=trajectory,
            state_distribution=state_dist,
            classification={
                'primary_type': classification.get('primary_type', 'Unknown'),
                'subtype': classification.get('subtype', 'Unknown')
            },
            risk_assessment=risk_response,
            mfpt_to_directing=report.mfpt_to_directing,
            n_critical_transitions=len(analysis.get('critical_transitions', [])),
            n_intervention_windows=len(analysis.get('intervention_windows', [])),
            recommendations=recommendations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/individuals/{name}/trajectory", response_model=List[str])
def get_trajectory(name: str):
    """Get just the trajectory for an individual."""
    loader = get_data_loader()
    individuals = loader.get_individuals_with_trajectory_data()

    if name not in individuals:
        raise HTTPException(status_code=404, detail=f"Individual '{name}' not found")

    return loader.load_individual_trajectory(name)


@app.get("/protocols", response_model=List[ProtocolInfo])
def list_protocols(
    category: Optional[str] = Query(None, description="Filter by category")
):
    """List all available intervention protocols."""
    if not INTERVENTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Intervention module not available")

    protocols = []
    for name, protocol in INTERVENTION_PROTOCOLS.items():
        info = ProtocolInfo(
            name=protocol.name,
            display_name=protocol.display_name,
            category=protocol.category,
            description=protocol.description or "",
            cost_range={
                'min': protocol.cost_range[0],
                'max': protocol.cost_range[1]
            },
            duration_range={
                'min': protocol.duration_range[0],
                'max': protocol.duration_range[1]
            },
            effectiveness={
                state: eff
                for state, eff in protocol.effectiveness_by_state.items()
            },
            applicable_states=protocol.applicable_states
        )

        if category is None or protocol.category == category:
            protocols.append(info)

    return protocols


@app.get("/protocols/{name}", response_model=ProtocolInfo)
def get_protocol_details(name: str):
    """Get details for a specific protocol."""
    if not INTERVENTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Intervention module not available")

    protocol = get_protocol(name)
    if protocol is None:
        raise HTTPException(status_code=404, detail=f"Protocol '{name}' not found")

    return ProtocolInfo(
        name=protocol.name,
        display_name=protocol.display_name,
        category=protocol.category,
        description=protocol.description or "",
        cost_range={
            'min': protocol.cost_range[0],
            'max': protocol.cost_range[1]
        },
        duration_range={
            'min': protocol.duration_range[0],
            'max': protocol.duration_range[1]
        },
        effectiveness={
            state: eff
            for state, eff in protocol.effectiveness_by_state.items()
        },
        applicable_states=protocol.applicable_states
    )


@app.get("/population/stats", response_model=PopulationStats)
def get_population_statistics():
    """Get aggregate population statistics."""
    loader = get_data_loader()
    individuals = loader.get_individuals_with_trajectory_data()

    if not individuals:
        raise HTTPException(status_code=404, detail="No individuals with trajectory data")

    risk_counts = Counter()
    state_totals = {s: 0.0 for s in STATE_NAMES}
    mfpt_values = []
    primary_types = Counter()
    subtypes = Counter()

    for name in individuals:
        try:
            trajectory = loader.load_individual_trajectory(name)
            ind_data = loader.get_individual_data(name)
            classification = ind_data.get('classification', {})

            # State distribution
            counts = Counter(trajectory)
            total = len(trajectory)
            for s in STATE_NAMES:
                state_totals[s] += counts.get(s, 0) / total

            # Classification
            primary_types[classification.get('primary_type', 'Unknown')] += 1
            subtypes[classification.get('subtype', 'Unknown')] += 1

            # Risk assessment (simplified)
            directing_pct = counts.get('Directing', 0) / total
            if directing_pct > 0.4:
                risk_counts['Critical'] += 1
            elif directing_pct > 0.3:
                risk_counts['High'] += 1
            elif directing_pct > 0.2:
                risk_counts['Moderate'] += 1
            else:
                risk_counts['Low'] += 1

        except Exception:
            continue

    n = len(individuals)
    avg_state = {s: state_totals[s] / n for s in STATE_NAMES}

    return PopulationStats(
        n_individuals=n,
        risk_distribution=dict(risk_counts),
        avg_state_distribution=avg_state,
        avg_mfpt_to_directing=3.5,  # Placeholder
        classification_distribution={
            'primary': dict(primary_types),
            'subtype': dict(subtypes)
        }
    )


@app.post("/simulate", response_model=SimulationResult)
def run_simulation(request: SimulationRequest):
    """Run a what-if simulation for an individual."""
    loader = get_data_loader()
    individuals = loader.get_individuals_with_trajectory_data()

    if request.individual_name not in individuals:
        raise HTTPException(status_code=404, detail=f"Individual '{request.individual_name}' not found")

    try:
        actual_trajectory = loader.load_individual_trajectory(request.individual_name)
        transition_matrix = loader.load_individual_transition_matrix(request.individual_name)

        if request.intervention_time >= len(actual_trajectory):
            raise HTTPException(
                status_code=400,
                detail=f"Intervention time {request.intervention_time} exceeds trajectory length {len(actual_trajectory)}"
            )

        # Run simulations
        simulated_outcomes = []
        initial_state = actual_trajectory[0]

        for _ in range(request.n_simulations):
            sim_traj = simulate_trajectory_api(
                transition_matrix,
                initial_state,
                len(actual_trajectory),
                request.intervention_time,
                request.intervention_strength
            )
            simulated_outcomes.append(sim_traj)

        # Compute statistics
        original_directing = sum(1 for s in actual_trajectory if s == 'Directing') / len(actual_trajectory)

        simulated_directing_rates = [
            sum(1 for s in traj if s == 'Directing') / len(traj)
            for traj in simulated_outcomes
        ]

        avg_simulated = np.mean(simulated_directing_rates)
        std_simulated = np.std(simulated_directing_rates)
        reduction = (original_directing - avg_simulated) / original_directing * 100 if original_directing > 0 else 0

        return SimulationResult(
            original_directing_pct=original_directing * 100,
            simulated_directing_pct=avg_simulated * 100,
            simulated_directing_std=std_simulated * 100,
            potential_reduction_pct=reduction,
            sample_trajectory=simulated_outcomes[0]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def simulate_trajectory_api(
    transition_matrix: np.ndarray,
    initial_state: str,
    n_steps: int,
    intervention_time: int,
    intervention_strength: float
) -> List[str]:
    """Simulate a trajectory with intervention."""
    state_to_idx = {s: i for i, s in enumerate(STATE_NAMES)}
    idx_to_state = {i: s for i, s in enumerate(STATE_NAMES)}

    current_idx = state_to_idx.get(initial_state, 0)
    trajectory = [initial_state]

    matrix = transition_matrix.copy()

    for step in range(1, n_steps):
        # Apply intervention
        if step == intervention_time:
            directing_idx = state_to_idx.get('Directing', 1)
            for i in range(4):
                if i != directing_idx:
                    reduction = matrix[i, directing_idx] * intervention_strength
                    matrix[i, directing_idx] -= reduction
                    for j in range(4):
                        if j != directing_idx:
                            matrix[i, j] += reduction / 3

            matrix = np.clip(matrix, 0, 1)
            for i in range(4):
                matrix[i] = matrix[i] / matrix[i].sum()

        # Sample next state
        probs = matrix[current_idx]
        probs = probs / probs.sum()
        next_idx = np.random.choice(4, p=probs)
        trajectory.append(idx_to_state[next_idx])
        current_idx = next_idx

    return trajectory


# Run with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
