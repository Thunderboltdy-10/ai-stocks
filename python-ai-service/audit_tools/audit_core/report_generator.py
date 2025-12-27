"""
Report Generator - Compiles analysis results into comprehensive reports
"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


class ReportGenerator:
    """Generates comprehensive audit reports in multiple formats"""

    def __init__(self, inventory_df, import_map, categorization_df,
                 dependency_metrics, model_status_df, config):
        """
        Initialize report generator

        Args:
            inventory_df: File inventory DataFrame
            import_map: Import mapping dictionary
            categorization_df: Categorization results DataFrame
            dependency_metrics: Dependency graph metrics
            model_status_df: Model status DataFrame
            config: Audit configuration
        """
        self.inventory_df = inventory_df
        self.import_map = import_map
        self.categorization_df = categorization_df
        self.dependency_metrics = dependency_metrics
        self.model_status_df = model_status_df
        self.config = config

        self.summary_stats = {}
        self.key_findings = []

    def generate_summary_stats(self):
        """Calculate overall project statistics"""
        print("Generating summary statistics...")

        # Overall stats
        total_files = len(self.inventory_df)
        total_size_mb = self.inventory_df['size_mb'].sum()

        # File type breakdown
        file_type_counts = self.inventory_df['file_type'].value_counts().to_dict()

        # Python-specific stats
        python_files = self.inventory_df[self.inventory_df['file_type'] == 'PYTHON_SOURCE']
        total_python_files = len(python_files)

        # Category stats
        category_counts = self.categorization_df['category'].value_counts().to_dict()
        category_percentages = (self.categorization_df['category'].value_counts(normalize=True) * 100).to_dict()

        # Low confidence categorizations
        low_confidence_count = self.categorization_df[
            self.categorization_df.get('needs_review', False) == True
        ].shape[0] if 'needs_review' in self.categorization_df.columns else 0

        # Dependency stats
        total_imports = sum(len(imports) for imports in self.import_map.values())

        # Model stats
        total_models = len(self.model_status_df) if self.model_status_df is not None and not self.model_status_df.empty else 0
        total_model_size_mb = self.model_status_df['size_mb'].sum() if total_models > 0 else 0

        self.summary_stats = {
            'audit_date': datetime.now().isoformat(),
            'project_root': str(self.config.get('root_directory', '')),
            'total_files': total_files,
            'total_size_mb': round(total_size_mb, 2),
            'total_python_files': total_python_files,
            'total_imports': total_imports,
            'total_models': total_models,
            'total_model_size_mb': round(total_model_size_mb, 2),
            'file_type_breakdown': file_type_counts,
            'category_breakdown': category_counts,
            'category_percentages': {k: round(v, 2) for k, v in category_percentages.items()},
            'low_confidence_count': low_confidence_count
        }

        return self.summary_stats

    def identify_key_findings(self):
        """Identify critical issues and opportunities"""
        print("Identifying key findings...")

        findings = []

        # Finding 1: Root level files
        root_clutter = self.categorization_df[
            self.categorization_df['category'] == 'ROOT_CLUTTER'
        ]
        if not root_clutter.empty:
            findings.append({
                'severity': 'MEDIUM',
                'category': 'Organization',
                'title': 'Root-level clutter detected',
                'description': f'{len(root_clutter)} files found in root directory that should be organized',
                'affected_count': len(root_clutter),
                'recommendation': 'Move files to appropriate subdirectories'
            })

        # Finding 2: Deprecated files
        deprecated = self.categorization_df[
            self.categorization_df['category'] == 'DEPRECATED'
        ]
        if not deprecated.empty:
            deprecated_size = self.categorization_df[
                self.categorization_df['category'] == 'DEPRECATED'
            ]['size_mb'].sum() if 'size_mb' in self.categorization_df.columns else 0

            findings.append({
                'severity': 'LOW',
                'category': 'Cleanup',
                'title': 'Deprecated files present',
                'description': f'{len(deprecated)} deprecated files consuming {deprecated_size:.1f} MB',
                'affected_count': len(deprecated),
                'recommendation': 'Archive or delete deprecated files'
            })

        # Finding 3: Orphaned models
        if self.model_status_df is not None and not self.model_status_df.empty:
            orphaned_models = self.model_status_df[
                self.model_status_df['status'] == 'ORPHANED'
            ]
            if not orphaned_models.empty:
                orphaned_size = orphaned_models['size_mb'].sum()
                findings.append({
                    'severity': 'HIGH',
                    'category': 'Models',
                    'title': 'Orphaned model artifacts',
                    'description': f'{len(orphaned_models)} orphaned models consuming {orphaned_size:.1f} MB',
                    'affected_count': len(orphaned_models),
                    'recommendation': 'Review and delete/archive orphaned models'
                })

        # Finding 4: Unknown categorizations
        unknown = self.categorization_df[
            self.categorization_df['category'] == 'UNKNOWN'
        ]
        if len(unknown) > len(self.categorization_df) * 0.1:  # More than 10%
            findings.append({
                'severity': 'MEDIUM',
                'category': 'Categorization',
                'title': 'High proportion of unknown files',
                'description': f'{len(unknown)} files ({len(unknown)/len(self.categorization_df)*100:.1f}%) could not be categorized',
                'affected_count': len(unknown),
                'recommendation': 'Review and improve categorization rules'
            })

        # Finding 5: Low confidence categorizations
        if self.summary_stats.get('low_confidence_count', 0) > 0:
            findings.append({
                'severity': 'LOW',
                'category': 'Quality',
                'title': 'Low confidence categorizations',
                'description': f'{self.summary_stats["low_confidence_count"]} files categorized with low confidence',
                'affected_count': self.summary_stats['low_confidence_count'],
                'recommendation': 'Review flagged files manually'
            })

        self.key_findings = findings
        return findings

    def generate_markdown_report(self, output_path):
        """Generate comprehensive Markdown report"""
        print("Generating Markdown report...")

        report_lines = [
            "# Codebase Audit Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Project:** {self.config.get('root_directory', 'Unknown')}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"This audit analyzed **{self.summary_stats['total_files']} files** ",
            f"across the project, totaling **{self.summary_stats['total_size_mb']:.1f} MB**.",
            "",
            "### Quick Stats",
            "",
            f"- **Python Files:** {self.summary_stats['total_python_files']}",
            f"- **Total Imports:** {self.summary_stats['total_imports']}",
            f"- **Model Artifacts:** {self.summary_stats['total_models']} ({self.summary_stats['total_model_size_mb']:.1f} MB)",
            f"- **Categories Identified:** {len(self.summary_stats['category_breakdown'])}",
            "",
            "---",
            "",
            "## File Distribution",
            "",
            "### By File Type",
            "",
            "| File Type | Count |",
            "|-----------|-------|"
        ]

        # Add file type table
        for file_type, count in sorted(self.summary_stats['file_type_breakdown'].items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"| {file_type} | {count} |")

        report_lines.extend([
            "",
            "### By Category",
            "",
            "| Category | Count | Percentage |",
            "|----------|-------|------------|"
        ])

        # Add category table
        for category, count in sorted(self.summary_stats['category_breakdown'].items(), key=lambda x: x[1], reverse=True):
            percentage = self.summary_stats['category_percentages'].get(category, 0)
            report_lines.append(f"| {category} | {count} | {percentage:.1f}% |")

        # Add key findings
        report_lines.extend([
            "",
            "---",
            "",
            "## Key Findings",
            ""
        ])

        for finding in self.key_findings:
            severity_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(finding['severity'], '⚪')
            report_lines.extend([
                f"### {severity_emoji} {finding['title']}",
                "",
                f"**Severity:** {finding['severity']}",
                f"**Category:** {finding['category']}",
                "",
                finding['description'],
                "",
                f"**Recommendation:** {finding['recommendation']}",
                ""
            ])

        # Add model analysis if available
        if self.model_status_df is not None and not self.model_status_df.empty:
            report_lines.extend([
                "---",
                "",
                "## Model Artifact Analysis",
                "",
                f"Found **{len(self.model_status_df)} model artifacts** in the project.",
                "",
                "### Model Status Distribution",
                "",
                "| Status | Count |",
                "|--------|-------|"
            ])

            status_counts = self.model_status_df['status'].value_counts()
            for status, count in status_counts.items():
                report_lines.append(f"| {status} | {count} |")

            report_lines.extend([
                "",
                "### Recommendations",
                "",
                "| Recommendation | Count |",
                "|----------------|-------|"
            ])

            rec_counts = self.model_status_df['recommendation'].value_counts()
            for rec, count in rec_counts.items():
                report_lines.append(f"| {rec} | {count} |")

        # Add next steps
        report_lines.extend([
            "",
            "---",
            "",
            "## Recommended Next Steps",
            "",
            "1. **Review Key Findings**: Address high-severity issues first",
            "2. **Organize Root Files**: Move root-level files to appropriate directories",
            "3. **Clean Deprecated Files**: Archive or delete deprecated code",
            "4. **Model Cleanup**: Review and remove orphaned model artifacts",
            "5. **Improve Categorization**: Review files with unknown or low-confidence categories",
            "",
            "---",
            "",
            "## Detailed Outputs",
            "",
            "Additional detailed reports have been generated:",
            "",
            "- `data/file_inventory_complete.csv` - Complete file listing with all metadata",
            "- `data/categorization_results.csv` - Detailed categorization results",
            "- `data/node_metrics.csv` - Dependency graph metrics",
            "- `data/model_status_matrix.csv` - Model artifact analysis",
            "- `graphs/dependency_graph_full.png` - Visual dependency map",
            "",
            f"_Report generated by Codebase Audit Tool on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        ])

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"Markdown report generated: {output_path}")
        return output_path

    def export_all_csvs(self, output_dir):
        """Export all data to CSV files"""
        print("Exporting CSV files...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Complete inventory
        inventory_path = output_dir / 'file_inventory_complete.csv'
        self.categorization_df.to_csv(inventory_path, index=False)
        print(f"  Complete inventory: {inventory_path}")

        # Model status
        if self.model_status_df is not None and not self.model_status_df.empty:
            model_path = output_dir / 'model_status_matrix.csv'
            self.model_status_df.to_csv(model_path, index=False)
            print(f"  Model status: {model_path}")

        return output_dir

    def export_all_json(self, output_dir):
        """Export summary data to JSON"""
        print("Exporting JSON files...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary statistics
        stats_path = output_dir / 'summary_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.summary_stats, f, indent=2, default=str)
        print(f"  Summary stats: {stats_path}")

        # Key findings
        findings_path = output_dir / 'key_findings.json'
        with open(findings_path, 'w') as f:
            json.dump(self.key_findings, f, indent=2)
        print(f"  Key findings: {findings_path}")

        return output_dir

    def export_visualizations(self, output_dir):
        """Generate additional visualizations"""
        print("Generating visualizations...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Category distribution pie chart
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            categories = list(self.summary_stats['category_breakdown'].keys())
            counts = list(self.summary_stats['category_breakdown'].values())

            ax.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
            ax.set_title('File Category Distribution')

            pie_path = output_dir / 'category_distribution.png'
            plt.tight_layout()
            plt.savefig(pie_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Category distribution: {pie_path}")

        except Exception as e:
            print(f"  Error generating category chart: {e}")

        return output_dir

    def generate_all_reports(self, output_root):
        """Generate all reports and exports"""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE AUDIT REPORTS")
        print("="*60 + "\n")

        output_root = Path(output_root)

        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = output_root / f"audit_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        reports_dir = output_dir / 'reports'
        data_dir = output_dir / 'data'
        graphs_dir = output_dir / 'graphs'

        for d in [reports_dir, data_dir, graphs_dir]:
            d.mkdir(exist_ok=True)

        # Generate all outputs
        self.generate_summary_stats()
        self.identify_key_findings()

        # Markdown report
        md_path = reports_dir / 'AUDIT_REPORT.md'
        self.generate_markdown_report(md_path)

        # CSV exports
        self.export_all_csvs(data_dir)

        # JSON exports
        self.export_all_json(data_dir)

        # Visualizations
        self.export_visualizations(graphs_dir)

        # Create index file
        index_lines = [
            "# Audit Results Index",
            "",
            f"Audit completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Reports",
            f"- [Main Report](reports/AUDIT_REPORT.md)",
            "",
            "## Data Files",
            f"- [Complete Inventory](data/file_inventory_complete.csv)",
            f"- [Model Status](data/model_status_matrix.csv)",
            f"- [Summary Statistics](data/summary_stats.json)",
            f"- [Key Findings](data/key_findings.json)",
            "",
            "## Visualizations",
            f"- [Category Distribution](graphs/category_distribution.png)",
            f"- [Dependency Graph](graphs/dependency_graph_full.png)",
        ]

        index_path = output_dir / 'INDEX.md'
        with open(index_path, 'w') as f:
            f.write('\n'.join(index_lines))

        print("\n" + "="*60)
        print("AUDIT REPORTS GENERATED SUCCESSFULLY")
        print("="*60)
        print(f"\nOutput directory: {output_dir}")
        print(f"Main report: {md_path}")
        print(f"\nReview the INDEX.md file for all outputs.")

        return output_dir


if __name__ == '__main__':
    print("ReportGenerator module - use via main orchestration script")
