"""
Categorization Engine - Classifies files into categories based on patterns
"""
import json
from pathlib import Path
import pandas as pd


class Categorizer:
    """Categorizes files based on path, name, content, and import patterns"""

    def __init__(self, file_inventory_df, import_map, categorization_rules):
        """
        Initialize the categorizer

        Args:
            file_inventory_df: DataFrame with file inventory
            import_map: Dictionary mapping files to their imports
            categorization_rules: Dictionary with categorization rules
        """
        self.inventory_df = file_inventory_df.copy()
        self.import_map = import_map
        self.rules = categorization_rules

        # Track categorization results
        self.categorization_results = []

    def categorize_by_path(self, filepath):
        """Categorize based on file path patterns"""
        path_str = str(filepath).lower()

        for category, config in self.rules.get('path_based_rules', {}).items():
            patterns = config.get('patterns', [])
            confidence = config.get('confidence', 0.5)

            for pattern in patterns:
                if pattern.lower() in path_str:
                    return category, confidence, f"path contains '{pattern}'"

        return None, 0, None

    def categorize_by_name(self, filename):
        """Categorize based on filename patterns"""
        name_lower = filename.lower()

        for category, config in self.rules.get('name_based_rules', {}).items():
            patterns = config.get('patterns', [])
            confidence = config.get('confidence', 0.5)

            for pattern in patterns:
                if pattern.lower() in name_lower:
                    return category, confidence, f"name contains '{pattern}'"

        return None, 0, None

    def categorize_by_content(self, filepath):
        """Categorize based on file content markers"""
        try:
            # Read first 100 lines only for performance
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline() for _ in range(100)]
                content = ''.join(lines).upper()

            for category, config in self.rules.get('content_markers', {}).items():
                markers = config.get('markers', [])
                confidence = config.get('confidence', 0.5)

                for marker in markers:
                    if marker.upper() in content:
                        return category, confidence, f"content contains '{marker}'"

            return None, 0, None

        except Exception:
            return None, 0, None

    def categorize_by_imports(self, filepath, file_imports):
        """Categorize based on import patterns"""
        if not file_imports:
            return None, 0, None

        # Extract module names
        imported_modules = set()
        for imp in file_imports:
            if imp.get('module'):
                imported_modules.add(imp['module'])

        # Check import-based rules
        for category, config in self.rules.get('import_based_rules', {}).items():
            import_patterns = config.get('imports', [])
            confidence = config.get('confidence', 0.5)

            for pattern in import_patterns:
                if any(pattern in module for module in imported_modules):
                    return category, confidence, f"imports '{pattern}'"

        return None, 0, None

    def categorize_by_file_type(self, file_type, filepath):
        """Special categorization for non-Python files"""
        # Model artifacts
        if file_type.startswith('MODEL_'):
            return 'ARTIFACT', 0.95, 'model file type'

        # Data files
        if file_type.startswith('DATA_'):
            return 'ARTIFACT', 0.90, 'data file type'

        # Log files
        if file_type == 'LOG_FILE':
            return 'ARTIFACT', 0.90, 'log file'

        # Batch/shell scripts
        if file_type in ['SCRIPT_BATCH', 'SCRIPT_SHELL']:
            # Check if in deprecated folder or named with old/deprecated
            if 'deprecated' in str(filepath).lower() or 'old' in str(filepath).lower():
                return 'DEPRECATED', 0.95, 'deprecated script'
            return 'UTILITY', 0.70, 'script file'

        # Configuration files
        if file_type.startswith('CONFIG_'):
            return 'CONFIG', 0.85, 'config file type'

        # Documentation
        if file_type.startswith('DOC_'):
            return 'CONFIG', 0.70, 'documentation file'

        # Notebooks
        if file_type == 'NOTEBOOK':
            return 'EXPERIMENTAL', 0.75, 'jupyter notebook'

        return None, 0, None

    def check_if_orphan(self, relative_path):
        """Check if file is an orphan (not imported by anything)"""
        # For Python files, check if it's imported anywhere
        is_imported = False

        for filepath, imports in self.import_map.items():
            for imp in imports:
                if imp.get('source') == 'LOCAL':
                    # Simplified check - would need more sophisticated module resolution
                    module = imp.get('module', '')
                    if module and module.replace('.', '/') in str(relative_path):
                        is_imported = True
                        break

            if is_imported:
                break

        if not is_imported:
            # Check if it's in root directory (root clutter)
            if '/' not in str(relative_path) or str(relative_path).count('/') == 0:
                return 'ROOT_CLUTTER', 0.90, 'root level file not imported'
            return 'ORPHAN', 0.75, 'not imported anywhere'

        return None, 0, None

    def categorize_file(self, row):
        """Apply all categorization rules to a single file"""
        filepath = Path(row['absolute_path'])
        relative_path = row['relative_path']
        filename = row['filename']
        file_type = row['file_type']

        # Collect all possible categorizations
        categories = []

        # 1. File type categorization (for non-Python files)
        if file_type != 'PYTHON_SOURCE':
            cat, conf, rule = self.categorize_by_file_type(file_type, relative_path)
            if cat:
                categories.append((cat, conf, rule))

        # 2. Path-based categorization
        cat, conf, rule = self.categorize_by_path(relative_path)
        if cat:
            categories.append((cat, conf, rule))

        # 3. Name-based categorization
        cat, conf, rule = self.categorize_by_name(filename)
        if cat:
            categories.append((cat, conf, rule))

        # 4. Content-based categorization (for Python files)
        if file_type == 'PYTHON_SOURCE':
            cat, conf, rule = self.categorize_by_content(filepath)
            if cat:
                categories.append((cat, conf, rule))

        # 5. Import-based categorization
        file_imports = self.import_map.get(relative_path, [])
        if file_imports:
            cat, conf, rule = self.categorize_by_imports(filepath, file_imports)
            if cat:
                categories.append((cat, conf, rule))

        # 6. Check if orphan (only for Python files)
        if file_type == 'PYTHON_SOURCE':
            cat, conf, rule = self.check_if_orphan(relative_path)
            if cat:
                categories.append((cat, conf, rule))

        # Select best category
        if categories:
            # Sort by confidence (highest first)
            categories.sort(key=lambda x: x[1], reverse=True)
            category, confidence, matched_rule = categories[0]
            rules_matched = [cat[2] for cat in categories]
        else:
            category = 'UNKNOWN'
            confidence = 0.0
            rules_matched = []

        return {
            'relative_path': relative_path,
            'category': category,
            'confidence': confidence,
            'rules_matched': rules_matched,
            'needs_review': confidence < self.rules.get('low_confidence_threshold', 0.60)
        }

    def categorize_all_files(self):
        """Categorize all files in the inventory"""
        print(f"Categorizing {len(self.inventory_df)} files...")

        results = []
        for idx, row in self.inventory_df.iterrows():
            if idx % 100 == 0:
                print(f"  Categorized {idx}/{len(self.inventory_df)} files...")

            result = self.categorize_file(row)
            results.append(result)

        self.categorization_results = results
        print(f"Categorization complete.")

        # Generate statistics
        df_results = pd.DataFrame(results)
        category_counts = df_results['category'].value_counts()
        print("\nCategory distribution:")
        for cat, count in category_counts.items():
            print(f"  {cat}: {count}")

        low_confidence = df_results[df_results['needs_review']].shape[0]
        print(f"\nLow-confidence categorizations: {low_confidence}")

        return results

    def generate_category_stats(self):
        """Generate statistics about categorization results"""
        if not self.categorization_results:
            return {}

        df = pd.DataFrame(self.categorization_results)

        stats = {}
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            stats[category] = {
                'count': len(cat_df),
                'avg_confidence': cat_df['confidence'].mean(),
                'low_confidence_count': cat_df[cat_df['needs_review']].shape[0]
            }

        return stats

    def export_categorization(self, output_path):
        """Export categorization results"""
        if not self.categorization_results:
            raise ValueError("No categorization results. Run categorize_all_files() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge with inventory
        results_df = pd.DataFrame(self.categorization_results)
        merged_df = self.inventory_df.merge(
            results_df,
            on='relative_path',
            how='left'
        )

        # Export to CSV
        merged_df.to_csv(output_path, index=False)
        print(f"Categorization results exported to: {output_path}")

        # Export category stats
        stats = self.generate_category_stats()
        stats_path = output_path.parent / 'category_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Category statistics exported to: {stats_path}")

        # Export low-confidence files
        low_conf_df = merged_df[merged_df['needs_review'] == True]
        if not low_conf_df.empty:
            review_path = output_path.parent / 'review_needed.csv'
            low_conf_df.to_csv(review_path, index=False)
            print(f"Low-confidence files exported to: {review_path}")

        return merged_df


if __name__ == '__main__':
    print("Categorizer module - use via main orchestration script")
