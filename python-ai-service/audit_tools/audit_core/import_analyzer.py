"""
Import Analyzer - Extracts import statements from Python files
"""
import ast
import json
import sys
from pathlib import Path
import pandas as pd


class ImportAnalyzer:
    """Analyzes Python imports to build dependency maps"""

    def __init__(self, file_inventory_df):
        """
        Initialize the import analyzer

        Args:
            file_inventory_df: DataFrame from FileDiscovery containing file inventory
        """
        self.inventory_df = file_inventory_df

        # Filter for Python files only
        self.python_files = file_inventory_df[
            file_inventory_df['file_type'] == 'PYTHON_SOURCE'
        ].copy()

        self.import_map = {}
        self.unparseable_files = []
        self.stdlib_modules = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()

    def extract_imports(self, filepath):
        """
        Extract all import statements from a Python file

        Args:
            filepath: Path to the Python file

        Returns:
            List of import dictionaries
        """
        imports = []

        try:
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content, filename=str(filepath))

            # Walk the AST to find imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # Handle: import module1, module2
                    for alias in node.names:
                        imports.append({
                            'type': 'import',
                            'module': alias.name,
                            'imported': None,
                            'alias': alias.asname,
                            'line': node.lineno,
                            'level': 0
                        })

                elif isinstance(node, ast.ImportFrom):
                    # Handle: from module import name1, name2
                    module = node.module or ''
                    level = node.level  # 0 = absolute, >0 = relative

                    for alias in node.names:
                        imports.append({
                            'type': 'from',
                            'module': module,
                            'imported': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno,
                            'level': level
                        })

            return imports

        except SyntaxError as e:
            self.unparseable_files.append({
                'filepath': filepath,
                'error_type': 'SyntaxError',
                'error_message': str(e),
                'line_number': e.lineno if hasattr(e, 'lineno') else None
            })
            return []

        except UnicodeDecodeError as e:
            self.unparseable_files.append({
                'filepath': filepath,
                'error_type': 'UnicodeDecodeError',
                'error_message': str(e),
                'line_number': None
            })
            return []

        except Exception as e:
            self.unparseable_files.append({
                'filepath': filepath,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'line_number': None
            })
            return []

    def classify_import_source(self, module_name):
        """
        Classify whether an import is from stdlib, third-party, or local

        Args:
            module_name: Name of the imported module

        Returns:
            'STDLIB', 'THIRD_PARTY', or 'LOCAL'
        """
        if not module_name:
            return 'UNKNOWN'

        # Extract top-level module name
        top_level = module_name.split('.')[0]

        # Check if it's a standard library module
        if top_level in self.stdlib_modules:
            return 'STDLIB'

        # Check common third-party packages
        common_third_party = {
            'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn',
            'torch', 'tensorflow', 'keras', 'xgboost', 'lightgbm',
            'plotly', 'networkx', 'requests', 'flask', 'fastapi',
            'pytest', 'unittest', 'mock', 'nose',
            'yfinance', 'alpaca_trade_api', 'binance'
        }

        if top_level in common_third_party:
            return 'THIRD_PARTY'

        # If it starts with common local patterns, classify as local
        local_patterns = ['models', 'utils', 'data', 'training', 'inference',
                         'evaluation', 'analysis', 'service', 'scripts']

        if top_level in local_patterns:
            return 'LOCAL'

        # Default to third-party for unknown modules
        return 'THIRD_PARTY'

    def analyze_all_imports(self):
        """Analyze imports for all Python files"""
        print(f"Analyzing imports for {len(self.python_files)} Python files...")

        for idx, row in self.python_files.iterrows():
            filepath = row['absolute_path']

            if idx % 10 == 0:
                print(f"  Analyzed {idx}/{len(self.python_files)} files...")

            # Extract imports
            imports = self.extract_imports(filepath)

            # Classify each import
            for imp in imports:
                imp['source'] = self.classify_import_source(imp['module'])

            # Store in map
            self.import_map[row['relative_path']] = imports

        print(f"Import analysis complete.")
        print(f"  Successfully analyzed: {len(self.import_map)} files")
        print(f"  Unparseable files: {len(self.unparseable_files)}")

        return {
            'import_map': self.import_map,
            'unparseable_files': self.unparseable_files
        }

    def detect_circular_imports(self):
        """Detect circular import dependencies (simplified version)"""
        # This is a placeholder for a more sophisticated cycle detection
        # Would require building a full dependency graph
        print("Note: Circular dependency detection will be done in DependencyGraph module")
        return []

    def export_import_map(self, output_path):
        """Export import map to JSON and summary CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export full import map to JSON
        json_path = output_path.parent / (output_path.stem + '_full.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.import_map, f, indent=2)
        print(f"Import map exported to: {json_path}")

        # Create summary CSV
        summary_data = []
        for filepath, imports in self.import_map.items():
            local_imports = [imp for imp in imports if imp['source'] == 'LOCAL']
            third_party_imports = [imp for imp in imports if imp['source'] == 'THIRD_PARTY']
            stdlib_imports = [imp for imp in imports if imp['source'] == 'STDLIB']

            summary_data.append({
                'file': filepath,
                'total_imports': len(imports),
                'local_imports': len(local_imports),
                'third_party_imports': len(third_party_imports),
                'stdlib_imports': len(stdlib_imports),
                'unique_modules': len(set(imp['module'] for imp in imports if imp['module']))
            })

        summary_df = pd.DataFrame(summary_data)
        csv_path = output_path
        summary_df.to_csv(csv_path, index=False)
        print(f"Import summary exported to: {csv_path}")

        # Export unparseable files
        if self.unparseable_files:
            error_df = pd.DataFrame(self.unparseable_files)
            error_path = output_path.parent / 'unparseable_files.csv'
            error_df.to_csv(error_path, index=False)
            print(f"Unparseable files exported to: {error_path}")

        return summary_df


if __name__ == '__main__':
    # Test with a sample DataFrame
    test_data = {
        'absolute_path': [__file__],
        'relative_path': ['audit_core/import_analyzer.py'],
        'file_type': ['PYTHON_SOURCE']
    }

    df = pd.DataFrame(test_data)
    analyzer = ImportAnalyzer(df)
    result = analyzer.analyze_all_imports()

    print(f"\nSample imports from this file:")
    for imp in result['import_map']['audit_core/import_analyzer.py'][:5]:
        print(f"  {imp['type']} {imp['module']} - {imp['source']}")
