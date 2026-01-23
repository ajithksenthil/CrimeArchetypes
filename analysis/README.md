# Criminal Life Event State Space Analysis

**Empirical Validation of the 4-Animal State Space from Computational Psychodynamics**

## Overview

This project provides a rigorous empirical validation comparing theory-driven (4-Animal State Space) and data-driven (clustering) approaches for modeling criminal behavioral sequences. The key finding is that the **4-Animal framework achieves 69.2% prediction accuracy**, outperforming all data-driven K=4 methods.

## Key Results

| Method | CV Accuracy | Improvement |
|--------|-------------|-------------|
| **4-Animal (Theory)** | **69.2%** | - |
| Agglom. Ward (Data) | 61.8% | +7.4pp |
| Agglom. Complete (Data) | 60.9% | +8.3pp |
| K-Means (Data) | 50.8% | +18.4pp |
| GMM (Data) | 50.8% | +18.4pp |

## Project Structure

```
analysis/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── # MAIN ANALYSIS SCRIPTS
├── research_pipeline.py                # Main reproducible pipeline
├── run_embedding_animal_comparison.py  # 4-Animal vs data-driven comparison
├── empirical_validation_study.py       # Comprehensive clustering study
│
├── # STATISTICAL ANALYSIS
├── statistical_analysis.py             # Bootstrap CIs, effect sizes, significance
├── publication_figures.py              # Generate publication-quality figures
│
├── # SUPPORTING MODULES
├── four_animal_state_space.py          # 4-Animal classifier implementation
├── comprehensive_clustering_comparison.py  # Multiple clustering methods
│
├── # RESULTS
├── empirical_study/
│   ├── four_animal_comparison_*/       # Comparison results
│   │   ├── comparison_results.json
│   │   └── animal_labels.json
│   ├── publication_figures/            # PDF/PNG figures for paper
│   │   ├── figure1_method_comparison.pdf
│   │   ├── figure2_four_animal_analysis.pdf
│   │   ├── figure3_cv_accuracy_boxplot.pdf
│   │   ├── figure4_theoretical_framework.pdf
│   │   └── table1_method_comparison.tex
│   ├── FINAL_COMPARATIVE_ANALYSIS.md   # Summary of findings
│   ├── METHODS_DOCUMENTATION.md        # Methods for publication
│   └── statistical_analysis.json       # Statistical test results
│
└── # DATA (not tracked in git)
    └── ../mnt/data/csv/                # CSV files with life events
```

## Quick Start

### 1. Setup Environment

```bash
cd /Users/ajithsenthil/Desktop/CrimeArchetypes/analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Full Analysis

```bash
python research_pipeline.py --mode full
```

### 3. Generate Publication Figures

```bash
python publication_figures.py
```

### 4. Run Statistical Analysis

```bash
python statistical_analysis.py
```

## Usage for Ongoing Research

### Adding New Criminal Data

1. Prepare CSV with columns: `year, age, event_description`
2. Save as `Type1_NewName.csv` in the data directory
3. Run:

```bash
python research_pipeline.py --mode add_data --new_file path/to/Type1_NewName.csv
```

### Running Custom Analysis

```python
from research_pipeline import ResearchPipeline, AnalysisConfig

config = AnalysisConfig(
    data_directory="/path/to/data",
    output_directory="/path/to/output",
    n_states=4,
    n_cv_folds=5
)

pipeline = ResearchPipeline(config)
results = pipeline.run_full_analysis()
```

## Theoretical Framework

The 4-Animal State Space categorizes behavior along two dimensions:

```
                    EXPLORE
                       │
           Seeking     │     Conferring
         (Self+Explore)│   (Other+Explore)
                       │
    SELF ──────────────┼────────────────── OTHER
                       │
           Revising    │     Directing
         (Self+Exploit)│   (Other+Exploit)
                       │
                    EXPLOIT
```

## Key Findings

1. **Directing dominates** (68.5% of events): Criminal life events are primarily characterized by Other-Exploitation behavior
2. **High persistence**: Once in Directing mode, 76.4% probability of staying there
3. **Attractor dynamics**: All states transition primarily toward Directing
4. **Superior prediction**: Theory-driven classification outperforms data-driven clustering

## Citation

If you use this code or findings, please cite:

```bibtex
@article{senthil2025criminal,
  title={Empirical Validation of the 4-Animal State Space for Criminal Behavioral Sequences},
  author={Senthil, Ajith},
  journal={[Journal]},
  year={2025}
}
```

## License

[Your license here]

## Contact

Ajith Senthil - [email]
