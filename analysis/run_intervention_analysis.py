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

# Visualization imports (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

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

    # Generate comparative analysis
    comparative = generate_comparative_analysis(results, output_dir)

    # Generate visualizations
    generate_visualizations(results, pop_stats, output_dir)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - individual_results.json: {len(results)} individuals")
    print(f"  - population_statistics.json: Aggregate statistics")
    print(f"  - ANALYSIS_SUMMARY.md: Human-readable summary")
    print(f"  - comparative_analysis.json: Cross-archetype analysis")
    print(f"  - COMPARATIVE_ANALYSIS.md: Comparative report")
    print(f"  - figures/: Visualization images")
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


def generate_visualizations(results: list, pop_stats: dict, output_dir: Path):
    """
    Generate visualization figures for the batch analysis.

    Args:
        results: List of individual analysis results
        pop_stats: Population statistics dict
        output_dir: Directory to save figures
    """
    if not VISUALIZATION_AVAILABLE:
        print("Warning: matplotlib/seaborn not available, skipping visualizations")
        return

    print("\nGenerating visualizations...")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'Critical': '#c0392b',
        'High': '#e67e22',
        'Moderate': '#f39c12',
        'Low': '#27ae60'
    }
    state_colors = {
        'Seeking': '#3498db',
        'Directing': '#e74c3c',
        'Conferring': '#27ae60',
        'Revising': '#f39c12'
    }

    # 1. Risk Distribution Pie Chart
    fig, ax = plt.subplots(figsize=(8, 6))
    risk_data = pop_stats['risk_distribution']
    risk_levels = ['Critical', 'High', 'Moderate', 'Low']
    values = [risk_data.get(level, 0) for level in risk_levels]
    risk_colors = [colors[level] for level in risk_levels]

    # Only include non-zero values
    labels_filtered = [l for l, v in zip(risk_levels, values) if v > 0]
    values_filtered = [v for v in values if v > 0]
    colors_filtered = [c for c, v in zip(risk_colors, values) if v > 0]

    if values_filtered:
        ax.pie(values_filtered, labels=labels_filtered, colors=colors_filtered,
               autopct='%1.1f%%', startangle=90, explode=[0.02]*len(values_filtered))
        ax.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
        plt.savefig(figures_dir / 'risk_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - risk_distribution.png")

    # 2. State Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    states = list(STATE_NAMES)
    avg_dist = [pop_stats['avg_state_distribution'].get(s, 0) * 100 for s in states]
    bar_colors = [state_colors.get(s, '#95a5a6') for s in states]

    bars = ax.bar(states, avg_dist, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Average Percentage (%)', fontsize=12)
    ax.set_xlabel('Behavioral State', fontsize=12)
    ax.set_title('Population Average State Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(avg_dist) * 1.2 if avg_dist else 100)

    # Add value labels on bars
    for bar, val in zip(bars, avg_dist):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.savefig(figures_dir / 'state_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - state_distribution.png")

    # 3. MFPT Distribution Histogram
    mfpt_values = [r['mfpt_to_directing'] for r in results if r.get('mfpt_to_directing')]
    if mfpt_values:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(mfpt_values, bins=15, color='#3498db', edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(mfpt_values), color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(mfpt_values):.1f}')
        ax.set_xlabel('Mean First Passage Time to Directing (events)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of MFPT to Directing State', fontsize=14, fontweight='bold')
        ax.legend()
        plt.savefig(figures_dir / 'mfpt_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - mfpt_distribution.png")

    # 4. Protocol Recommendations Bar Chart
    protocol_counts = pop_stats.get('protocol_recommendation_counts', {})
    if protocol_counts:
        fig, ax = plt.subplots(figsize=(12, 6))
        sorted_protocols = sorted(protocol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        protocols = [p[0].replace('_', ' ').title()[:25] for p in sorted_protocols]
        counts = [p[1] for p in sorted_protocols]

        bars = ax.barh(protocols, counts, color='#9b59b6', edgecolor='white')
        ax.set_xlabel('Times Recommended', fontsize=12)
        ax.set_title('Top 10 Recommended Intervention Protocols', fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Highest at top

        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    str(count), va='center', fontsize=10)

        plt.savefig(figures_dir / 'protocol_recommendations.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - protocol_recommendations.png")

    # 5. Risk Score vs MFPT Scatter Plot
    risk_scores = [r['risk_score'] for r in results if r.get('risk_score') and r.get('mfpt_to_directing')]
    mfpt_for_scatter = [r['mfpt_to_directing'] for r in results if r.get('risk_score') and r.get('mfpt_to_directing')]
    risk_levels_scatter = [r['risk_level'] for r in results if r.get('risk_score') and r.get('mfpt_to_directing')]

    if risk_scores and mfpt_for_scatter:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter_colors = [colors.get(rl, '#95a5a6') for rl in risk_levels_scatter]

        ax.scatter(mfpt_for_scatter, risk_scores, c=scatter_colors, s=100, alpha=0.7, edgecolors='white')
        ax.set_xlabel('MFPT to Directing (events)', fontsize=12)
        ax.set_ylabel('Risk Score', fontsize=12)
        ax.set_title('Risk Score vs. Mean First Passage Time', fontsize=14, fontweight='bold')

        # Add legend
        legend_handles = [mpatches.Patch(color=colors[level], label=level)
                         for level in ['Critical', 'High', 'Moderate', 'Low']
                         if level in risk_levels_scatter]
        ax.legend(handles=legend_handles, title='Risk Level')

        plt.savefig(figures_dir / 'risk_vs_mfpt.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - risk_vs_mfpt.png")

    # 6. Classification Distribution Stacked Bar
    classification_dist = pop_stats.get('classification_distribution', {})
    if classification_dist.get('primary') and classification_dist.get('subtype'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Primary types
        primary = classification_dist['primary']
        ax1 = axes[0]
        ax1.bar(primary.keys(), primary.values(), color='#3498db', edgecolor='white')
        ax1.set_title('Primary Classification Types', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)

        # Subtypes
        subtype = classification_dist['subtype']
        ax2 = axes[1]
        sorted_subtypes = sorted(subtype.items(), key=lambda x: x[1], reverse=True)[:10]
        ax2.barh([s[0][:20] for s in sorted_subtypes], [s[1] for s in sorted_subtypes],
                 color='#e74c3c', edgecolor='white')
        ax2.set_title('Top Subtypes', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Count')
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(figures_dir / 'classification_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - classification_distribution.png")

    # 7. Individual Comparison Heatmap (State Distributions)
    if len(results) >= 3:
        fig, ax = plt.subplots(figsize=(12, max(6, len(results) * 0.4)))

        # Create matrix
        names = [r['name'][:20] for r in results]
        states = list(STATE_NAMES)
        matrix = np.array([[r['state_distribution'].get(s, 0) for s in states] for r in results])

        # Create heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states, fontsize=10)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('State Proportion', fontsize=10)

        ax.set_title('Individual State Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Behavioral State', fontsize=12)
        ax.set_ylabel('Individual', fontsize=12)

        plt.savefig(figures_dir / 'individual_comparison_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - individual_comparison_heatmap.png")

    print(f"Visualizations saved to: {figures_dir}")


def generate_comparative_analysis(results: list, output_dir: Path) -> dict:
    """
    Generate comparative analysis across individuals and archetypes.

    Args:
        results: List of individual analysis results
        output_dir: Directory to save analysis files

    Returns:
        Dict with comparative analysis results
    """
    print("\nGenerating comparative analysis...")

    # Group by archetype
    by_primary = defaultdict(list)
    by_subtype = defaultdict(list)

    for r in results:
        primary = r['classification']['primary_type']
        subtype = r['classification']['subtype']
        by_primary[primary].append(r)
        by_subtype[subtype].append(r)

    # Compute archetype-level statistics
    archetype_stats = {}

    for archetype, individuals in by_primary.items():
        mfpt_values = [r['mfpt_to_directing'] for r in individuals if r.get('mfpt_to_directing')]
        risk_scores = [r['risk_score'] for r in individuals if r.get('risk_score')]
        risk_counts = Counter(r['risk_level'] for r in individuals)

        # Average state distribution for this archetype
        avg_state = defaultdict(float)
        for r in individuals:
            for state, val in r['state_distribution'].items():
                avg_state[state] += val
        for state in avg_state:
            avg_state[state] /= len(individuals)

        # Protocol recommendations for this archetype
        protocol_counts = Counter()
        for r in individuals:
            for rec in r.get('top_recommendations', []):
                protocol_counts[rec['protocol']] += 1

        archetype_stats[archetype] = {
            'n_individuals': len(individuals),
            'avg_mfpt': np.mean(mfpt_values) if mfpt_values else 0,
            'std_mfpt': np.std(mfpt_values) if mfpt_values else 0,
            'avg_risk_score': np.mean(risk_scores) if risk_scores else 0,
            'risk_distribution': dict(risk_counts),
            'avg_state_distribution': dict(avg_state),
            'top_protocols': dict(protocol_counts.most_common(5)),
            'avg_intervention_windows': np.mean([r.get('n_intervention_windows', 0) for r in individuals]),
            'avg_critical_transitions': np.mean([r.get('n_critical_transitions', 0) for r in individuals])
        }

    # Subtype-level statistics
    subtype_stats = {}
    for subtype, individuals in by_subtype.items():
        mfpt_values = [r['mfpt_to_directing'] for r in individuals if r.get('mfpt_to_directing')]
        risk_scores = [r['risk_score'] for r in individuals if r.get('risk_score')]

        subtype_stats[subtype] = {
            'n_individuals': len(individuals),
            'avg_mfpt': np.mean(mfpt_values) if mfpt_values else 0,
            'avg_risk_score': np.mean(risk_scores) if risk_scores else 0,
            'primary_types': dict(Counter(r['classification']['primary_type'] for r in individuals))
        }

    # Cross-archetype protocol effectiveness
    protocol_by_archetype = {}
    for archetype, stats in archetype_stats.items():
        for protocol, count in stats['top_protocols'].items():
            if protocol not in protocol_by_archetype:
                protocol_by_archetype[protocol] = {}
            protocol_by_archetype[protocol][archetype] = count

    comparative = {
        'archetype_statistics': archetype_stats,
        'subtype_statistics': subtype_stats,
        'protocol_by_archetype': protocol_by_archetype,
        'highest_risk_archetype': max(archetype_stats.items(),
                                       key=lambda x: x[1]['avg_risk_score'])[0] if archetype_stats else None,
        'lowest_mfpt_archetype': min(archetype_stats.items(),
                                      key=lambda x: x[1]['avg_mfpt'] if x[1]['avg_mfpt'] > 0 else float('inf'))[0] if archetype_stats else None
    }

    # Save comparative analysis
    with open(output_dir / 'comparative_analysis.json', 'w') as f:
        json.dump(comparative, f, indent=2)

    # Generate comparative report
    report = generate_comparative_report(comparative)
    with open(output_dir / 'COMPARATIVE_ANALYSIS.md', 'w') as f:
        f.write(report)

    print(f"  - comparative_analysis.json")
    print(f"  - COMPARATIVE_ANALYSIS.md")

    return comparative


def generate_comparative_report(comparative: dict) -> str:
    """Generate a markdown comparative analysis report."""
    report = f"""# Comparative Intervention Analysis

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Highest Risk Archetype:** {comparative.get('highest_risk_archetype', 'N/A')}
- **Fastest Escalation (Lowest MFPT):** {comparative.get('lowest_mfpt_archetype', 'N/A')}

## Archetype Comparison

"""
    for archetype, stats in comparative.get('archetype_statistics', {}).items():
        report += f"""### {archetype}

- **Individuals:** {stats['n_individuals']}
- **Average Risk Score:** {stats['avg_risk_score']:.2f}
- **Average MFPT to Directing:** {stats['avg_mfpt']:.1f} events (±{stats['std_mfpt']:.1f})
- **Average Intervention Windows:** {stats['avg_intervention_windows']:.1f}
- **Average Critical Transitions:** {stats['avg_critical_transitions']:.1f}

**Risk Distribution:** {stats['risk_distribution']}

**Top Recommended Protocols:**
"""
        for protocol, count in stats['top_protocols'].items():
            report += f"- {protocol}: {count} recommendations\n"
        report += "\n"

    report += """## Subtype Analysis

| Subtype | N | Avg Risk Score | Avg MFPT |
|---------|---|----------------|----------|
"""
    for subtype, stats in sorted(comparative.get('subtype_statistics', {}).items(),
                                  key=lambda x: x[1]['avg_risk_score'], reverse=True):
        report += f"| {subtype[:25]} | {stats['n_individuals']} | {stats['avg_risk_score']:.2f} | {stats['avg_mfpt']:.1f} |\n"

    report += """

## Protocol Effectiveness by Archetype

| Protocol | """ + " | ".join(comparative.get('archetype_statistics', {}).keys()) + """ |
|----------|""" + "|".join(["---"] * len(comparative.get('archetype_statistics', {}))) + """|
"""
    for protocol, archetypes in comparative.get('protocol_by_archetype', {}).items():
        row = [str(archetypes.get(a, 0)) for a in comparative.get('archetype_statistics', {}).keys()]
        report += f"| {protocol[:30]} | " + " | ".join(row) + " |\n"

    report += """
---
*This comparative analysis identifies patterns across archetypes to inform targeted intervention strategies.*
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
