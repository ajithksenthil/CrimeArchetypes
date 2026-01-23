#!/usr/bin/env python3
"""
Batch Intervention Analysis

Runs intervention analysis across all individuals with trajectory data,
generating summary statistics and identifying population-level patterns.

Usage:
    python run_intervention_analysis.py [--output-dir OUTPUT_DIR]
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'dashboard'))

from dashboard.data.loader import DashboardDataLoader, STATE_NAMES
from intervention import (
    BehavioralSCM,
    TrajectoryAnalyzer,
    CounterfactualEngine,
    InterventionOptimizer,
    ClinicalReportGenerator,
    INTERVENTION_PROTOCOLS,
    RiskLevel
)


def analyze_individual(loader: DashboardDataLoader, individual_name: str) -> dict:
    """
    Run full intervention analysis for a single individual.

    Args:
        loader: Data loader instance
        individual_name: Name of the individual

    Returns:
        Dict with analysis results
    """
    try:
        # Load individual data
        trajectory = loader.load_individual_trajectory(individual_name)
        transition_matrix = loader.load_individual_transition_matrix(individual_name)

        # Get classification
        ind_data = loader.get_individual_data(individual_name)
        classification = ind_data.get('classification', {})

        # Create analysis components
        scm = BehavioralSCM(transition_matrix, STATE_NAMES)
        analyzer = TrajectoryAnalyzer(transition_matrix=transition_matrix)
        optimizer = InterventionOptimizer(scm)
        report_gen = ClinicalReportGenerator(transition_matrix, STATE_NAMES)

        # Run comprehensive analysis
        analysis = analyzer.comprehensive_analysis(trajectory)

        # Generate report
        archetype = classification.get('subtype', 'Unknown') if classification else 'Unknown'
        report = report_gen.generate_individual_report(
            individual_id=individual_name,
            trajectory=trajectory,
            archetype=archetype,
            include_retrospective=True
        )

        # Get optimization results
        opt_result = optimizer.find_optimal_timing(
            trajectory=trajectory,
            budget_constraint=20000,
            max_interventions=5
        )

        # State distribution
        state_counts = Counter(trajectory)
        total_events = len(trajectory)
        state_distribution = {
            state: state_counts.get(state, 0) / total_events
            for state in STATE_NAMES
        }

        # Compute key statistics
        result = {
            'name': individual_name,
            'n_events': len(trajectory),
            'state_distribution': state_distribution,
            'dominant_state': max(state_distribution.items(), key=lambda x: x[1])[0],
            'classification': {
                'primary_type': classification.get('primary_type', 'Unknown') if classification else 'Unknown',
                'subtype': archetype
            },
            'risk_level': report.risk_assessment.level.value if report.risk_assessment else 'Unknown',
            'risk_score': report.risk_assessment.score if report.risk_assessment else 0,
            'mfpt_to_directing': report.mfpt_to_directing,
            'n_critical_transitions': len(analysis.get('critical_transitions', [])),
            'n_intervention_windows': len(analysis.get('intervention_windows', [])),
            'n_tipping_points': len(analysis.get('tipping_points', [])),
            'n_missed_opportunities': len(report.missed_opportunities) if report.missed_opportunities else 0,
            'top_recommendations': [
                {
                    'protocol': rec.protocol.name,
                    'expected_benefit': rec.expected_benefit,
                    'cost': rec.cost,
                    'urgency': rec.urgency
                }
                for rec in opt_result.recommendations[:3]
            ] if opt_result.recommendations else [],
            'total_expected_benefit': opt_result.total_expected_benefit if opt_result else 0,
            'success': True
        }

        return result

    except Exception as e:
        return {
            'name': individual_name,
            'success': False,
            'error': str(e)
        }


def run_batch_analysis(data_dir: str, output_dir: str = None):
    """
    Run intervention analysis across all individuals.

    Args:
        data_dir: Path to empirical_study directory
        output_dir: Path for output files (default: creates timestamped directory)
    """
    print("=" * 60)
    print("BATCH INTERVENTION ANALYSIS")
    print("=" * 60)

    # Setup
    loader = DashboardDataLoader(Path(data_dir))
    individuals = loader.get_individuals_with_trajectory_data()

    if not individuals:
        print("ERROR: No individuals with trajectory data found")
        return

    print(f"\nFound {len(individuals)} individuals with trajectory data")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(data_dir) / f"intervention_analysis_{timestamp}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis for each individual
    results = []
    failed = []

    for i, name in enumerate(individuals, 1):
        print(f"\n[{i}/{len(individuals)}] Analyzing: {name}")
        result = analyze_individual(loader, name)

        if result['success']:
            results.append(result)
            risk = result['risk_level']
            n_windows = result['n_intervention_windows']
            print(f"  Risk: {risk} | Windows: {n_windows} | MFPT: {result['mfpt_to_directing']:.1f}")
        else:
            failed.append({'name': name, 'error': result['error']})
            print(f"  FAILED: {result['error']}")

    # Compute population statistics
    print("\n" + "=" * 60)
    print("COMPUTING POPULATION STATISTICS")
    print("=" * 60)

    pop_stats = compute_population_statistics(results)

    # Save results
    print("\nSaving results...")

    # Save individual results
    with open(output_dir / "individual_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save population statistics
    with open(output_dir / "population_statistics.json", 'w') as f:
        json.dump(pop_stats, f, indent=2)

    # Save failed analyses
    if failed:
        with open(output_dir / "failed_analyses.json", 'w') as f:
            json.dump(failed, f, indent=2)

    # Generate summary report
    summary_report = generate_summary_report(results, pop_stats, failed)
    with open(output_dir / "ANALYSIS_SUMMARY.md", 'w') as f:
        f.write(summary_report)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - individual_results.json: {len(results)} individuals")
    print(f"  - population_statistics.json: Aggregate statistics")
    print(f"  - ANALYSIS_SUMMARY.md: Human-readable summary")
    if failed:
        print(f"  - failed_analyses.json: {len(failed)} failures")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Analyzed: {len(results)} / {len(individuals)} individuals")
    print(f"Risk Distribution: {pop_stats['risk_distribution']}")
    print(f"Avg MFPT to Directing: {pop_stats['avg_mfpt_to_directing']:.1f} events")
    print(f"Total Intervention Windows: {pop_stats['total_intervention_windows']}")
    print(f"Most Recommended Protocol: {pop_stats['most_recommended_protocol']}")


def compute_population_statistics(results: list) -> dict:
    """Compute aggregate statistics across all individuals."""
    if not results:
        return {}

    # Risk distribution
    risk_counts = Counter(r['risk_level'] for r in results)

    # State distributions
    avg_state_dist = defaultdict(float)
    for r in results:
        for state, pct in r['state_distribution'].items():
            avg_state_dist[state] += pct
    for state in avg_state_dist:
        avg_state_dist[state] /= len(results)

    # Protocol recommendations
    protocol_counts = Counter()
    for r in results:
        for rec in r['top_recommendations']:
            protocol_counts[rec['protocol']] += 1

    # Aggregate metrics
    mfpt_values = [r['mfpt_to_directing'] for r in results if r['mfpt_to_directing']]
    risk_scores = [r['risk_score'] for r in results if r['risk_score']]

    return {
        'n_analyzed': len(results),
        'risk_distribution': dict(risk_counts),
        'avg_state_distribution': dict(avg_state_dist),
        'avg_mfpt_to_directing': np.mean(mfpt_values) if mfpt_values else 0,
        'std_mfpt_to_directing': np.std(mfpt_values) if mfpt_values else 0,
        'avg_risk_score': np.mean(risk_scores) if risk_scores else 0,
        'total_critical_transitions': sum(r['n_critical_transitions'] for r in results),
        'total_intervention_windows': sum(r['n_intervention_windows'] for r in results),
        'total_missed_opportunities': sum(r['n_missed_opportunities'] for r in results),
        'most_recommended_protocol': protocol_counts.most_common(1)[0][0] if protocol_counts else 'None',
        'protocol_recommendation_counts': dict(protocol_counts),
        'classification_distribution': {
            'primary': dict(Counter(r['classification']['primary_type'] for r in results)),
            'subtype': dict(Counter(r['classification']['subtype'] for r in results))
        }
    }


def generate_summary_report(results: list, pop_stats: dict, failed: list) -> str:
    """Generate a markdown summary report."""
    report = f"""# Batch Intervention Analysis Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Individuals Analyzed:** {len(results)}
- **Failed Analyses:** {len(failed)}
- **Total Events Analyzed:** {sum(r['n_events'] for r in results):,}

## Risk Distribution

| Risk Level | Count | Percentage |
|------------|-------|------------|
"""
    total = len(results)
    for level in ['Critical', 'High', 'Moderate', 'Low']:
        count = pop_stats['risk_distribution'].get(level, 0)
        pct = count / total * 100 if total > 0 else 0
        report += f"| {level} | {count} | {pct:.1f}% |\n"

    report += f"""
## Key Metrics

- **Average MFPT to Directing:** {pop_stats['avg_mfpt_to_directing']:.1f} events (±{pop_stats['std_mfpt_to_directing']:.1f})
- **Average Risk Score:** {pop_stats['avg_risk_score']:.2f}
- **Total Critical Transitions:** {pop_stats['total_critical_transitions']}
- **Total Intervention Windows:** {pop_stats['total_intervention_windows']}
- **Total Missed Opportunities:** {pop_stats['total_missed_opportunities']}

## State Distribution (Population Average)

| State | Percentage |
|-------|------------|
"""
    for state in STATE_NAMES:
        pct = pop_stats['avg_state_distribution'].get(state, 0) * 100
        report += f"| {state} | {pct:.1f}% |\n"

    report += """
## Most Recommended Protocols

| Protocol | Times Recommended |
|----------|-------------------|
"""
    for protocol, count in sorted(
        pop_stats['protocol_recommendation_counts'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]:
        report += f"| {protocol} | {count} |\n"

    report += """
## Classification Distribution

### Primary Types
| Type | Count |
|------|-------|
"""
    for ptype, count in pop_stats['classification_distribution']['primary'].items():
        report += f"| {ptype} | {count} |\n"

    report += """
### Subtypes
| Subtype | Count |
|---------|-------|
"""
    for stype, count in pop_stats['classification_distribution']['subtype'].items():
        report += f"| {stype} | {count} |\n"

    if failed:
        report += f"""
## Failed Analyses ({len(failed)})

| Individual | Error |
|------------|-------|
"""
        for f in failed:
            report += f"| {f['name']} | {f['error'][:50]}... |\n"

    report += """
---
*This analysis is for research purposes only. Clinical decisions should be made by qualified professionals.*
"""
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch intervention analysis")
    parser.add_argument(
        "--data-dir",
        default="empirical_study",
        help="Path to empirical_study directory"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates timestamped directory)"
    )

    args = parser.parse_args()
    run_batch_analysis(args.data_dir, args.output_dir)
