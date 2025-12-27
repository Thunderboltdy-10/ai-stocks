#!/usr/bin/env python3
"""
Main Orchestration Script for Codebase Audit
Runs complete audit: discovery, analysis, categorization, dependency graphing, and reporting
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add audit_core to path
sys.path.insert(0, str(Path(__file__).parent))

from audit_core.file_discovery import FileDiscovery
from audit_core.import_analyzer import ImportAnalyzer
from audit_core.categorizer import Categorizer
from audit_core.dependency_graph import DependencyGraph
from audit_core.model_analyzer import ModelAnalyzer
from audit_core.report_generator import ReportGenerator


def load_config(config_dir):
    """Load all configuration files"""
    config_dir = Path(config_dir)

    with open(config_dir / 'audit_settings.json', 'r') as f:
        settings = json.load(f)

    with open(config_dir / 'categorization_rules.json', 'r') as f:
        categorization_rules = json.load(f)

    with open(config_dir / 'model_patterns.json', 'r') as f:
        model_patterns = json.load(f)

    return settings, categorization_rules, model_patterns


def run_audit(root_dir=None, output_dir=None, skip_models=False, verbose=True):
    """
    Run complete codebase audit

    Args:
        root_dir: Root directory to audit (overrides config)
        output_dir: Output directory (overrides config)
        skip_models: Skip model artifact analysis
        verbose: Print detailed progress
    """
    print("\n" + "="*70)
    print("CODEBASE ARCHAEOLOGICAL AUDIT")
    print("="*70 + "\n")

    start_time = datetime.now()

    # Load configuration
    config_dir = Path(__file__).parent / 'audit_config'
    settings, categorization_rules, model_patterns = load_config(config_dir)

    # Override settings if provided
    if root_dir:
        settings['root_directory'] = root_dir
    if output_dir:
        settings['output_directory'] = output_dir

    root_path = Path(settings['root_directory'])
    output_path = Path(settings['output_directory'])

    print(f"Root Directory: {root_path}")
    print(f"Output Directory: {output_path}")
    print()

    # ========================================================================
    # PHASE 1: FILE DISCOVERY
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1/6: FILE DISCOVERY")
    print("="*70 + "\n")

    exclusion_patterns = categorization_rules.get('exclusion_patterns', [])
    discovery = FileDiscovery(root_path, exclusion_patterns)

    files = discovery.discover_all_files()

    # Export raw inventory
    temp_output = output_path / 'temp'
    temp_output.mkdir(parents=True, exist_ok=True)

    inventory_df = discovery.export_inventory(temp_output / 'file_inventory_raw.csv')

    print(f"\n✓ Phase 1 Complete: {len(files)} files discovered")

    # ========================================================================
    # PHASE 2: IMPORT ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 2/6: IMPORT ANALYSIS")
    print("="*70 + "\n")

    analyzer = ImportAnalyzer(inventory_df)
    result = analyzer.analyze_all_imports()

    import_map = result['import_map']
    unparseable_files = result['unparseable_files']

    # Export import data
    analyzer.export_import_map(temp_output / 'import_summary.csv')

    print(f"\n✓ Phase 2 Complete: {len(import_map)} Python files analyzed")
    print(f"  Unparseable files: {len(unparseable_files)}")

    # ========================================================================
    # PHASE 3: CATEGORIZATION
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 3/6: FILE CATEGORIZATION")
    print("="*70 + "\n")

    categorizer = Categorizer(inventory_df, import_map, categorization_rules)
    categorization_results = categorizer.categorize_all_files()

    categorization_df = categorizer.export_categorization(temp_output / 'categorization_results.csv')

    print(f"\n✓ Phase 3 Complete: {len(categorization_results)} files categorized")

    # ========================================================================
    # PHASE 4: DEPENDENCY GRAPH
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 4/6: DEPENDENCY GRAPH CONSTRUCTION")
    print("="*70 + "\n")

    dep_graph = DependencyGraph(import_map, categorization_df)
    graph = dep_graph.build_graph()

    metrics = dep_graph.calculate_metrics()

    # Export graph data
    graphs_dir = temp_output / 'graphs'
    graphs_dir.mkdir(exist_ok=True)

    graph_data = dep_graph.export_graph_data(temp_output)

    # Visualize graph
    layout = settings.get('dependency_graph_layout', 'spring')
    max_nodes = settings.get('max_graph_nodes', 200)
    dep_graph.visualize_graph(
        graphs_dir / 'dependency_graph_full.png',
        layout=layout,
        max_nodes=max_nodes
    )

    print(f"\n✓ Phase 4 Complete: Graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # ========================================================================
    # PHASE 5: MODEL ARTIFACT ANALYSIS
    # ========================================================================
    if not skip_models:
        print("\n" + "="*70)
        print("PHASE 5/6: MODEL ARTIFACT ANALYSIS")
        print("="*70 + "\n")

        # Find saved_models directory
        models_dir = root_path / 'saved_models'

        if models_dir.exists():
            # Get training and inference scripts from categorization
            training_scripts = categorization_df[
                categorization_df['category'] == 'CORE_TRAINING'
            ]['relative_path'].tolist()

            inference_scripts = categorization_df[
                categorization_df['category'] == 'CORE_INFERENCE'
            ]['relative_path'].tolist()

            # Run model analysis
            model_analyzer = ModelAnalyzer(
                models_dir,
                training_scripts,
                inference_scripts,
                model_patterns
            )

            model_artifacts = model_analyzer.discover_model_artifacts()
            model_results = model_analyzer.export_model_analysis(temp_output)

            model_status_df = model_results['status_df']

            print(f"\n✓ Phase 5 Complete: {len(model_artifacts)} model artifacts analyzed")
        else:
            print(f"Models directory not found: {models_dir}")
            print("Skipping model analysis")
            model_status_df = None
    else:
        print("\n⊘ Phase 5 Skipped: Model analysis disabled")
        model_status_df = None

    # ========================================================================
    # PHASE 6: REPORT GENERATION
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 6/6: COMPREHENSIVE REPORT GENERATION")
    print("="*70 + "\n")

    report_gen = ReportGenerator(
        inventory_df=inventory_df,
        import_map=import_map,
        categorization_df=categorization_df,
        dependency_metrics=metrics,
        model_status_df=model_status_df,
        config=settings
    )

    final_output_dir = report_gen.generate_all_reports(output_path)

    # Move graph to final location
    final_graphs_dir = final_output_dir / 'graphs'
    final_graphs_dir.mkdir(exist_ok=True)

    import shutil
    for graph_file in graphs_dir.glob('*'):
        shutil.copy(graph_file, final_graphs_dir / graph_file.name)

    # Copy other data files
    final_data_dir = final_output_dir / 'data'
    for data_file in temp_output.glob('*.csv'):
        shutil.copy(data_file, final_data_dir / data_file.name)
    for data_file in temp_output.glob('*.json'):
        shutil.copy(data_file, final_data_dir / data_file.name)

    # Clean up temp directory
    shutil.rmtree(temp_output)

    # ========================================================================
    # COMPLETION SUMMARY
    # ========================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "="*70)
    print("AUDIT COMPLETE")
    print("="*70)
    print(f"\n✓ Total Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"\n📁 Results Directory: {final_output_dir}")
    print(f"\n📊 Main Report: {final_output_dir / 'reports' / 'AUDIT_REPORT.md'}")
    print(f"📋 Index File: {final_output_dir / 'INDEX.md'}")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Review the main audit report (AUDIT_REPORT.md)")
    print("2. Check key findings for critical issues")
    print("3. Examine the dependency graph visualization")
    print("4. Review model status matrix for cleanup candidates")
    print("5. Plan cleanup actions based on recommendations")

    print("\n" + "="*70 + "\n")

    return final_output_dir


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Codebase Archaeological Audit - Comprehensive project analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_audit.py
  python run_audit.py --root-dir /path/to/project
  python run_audit.py --skip-models --verbose
  python run_audit.py --output-dir /path/to/output
        """
    )

    parser.add_argument(
        '--root-dir',
        type=str,
        help='Root directory to audit (overrides config)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (overrides config)'
    )

    parser.add_argument(
        '--skip-models',
        action='store_true',
        help='Skip model artifact analysis'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output (default: True)'
    )

    args = parser.parse_args()

    try:
        output_dir = run_audit(
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            skip_models=args.skip_models,
            verbose=args.verbose
        )

        print(f"\n✓ Success! Results saved to: {output_dir}\n")
        return 0

    except Exception as e:
        print(f"\n✗ Error during audit: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
