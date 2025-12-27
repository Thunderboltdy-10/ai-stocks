"""
File Discovery Engine - Scans filesystem and extracts file metadata
"""
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


class FileDiscovery:
    """Discovers and catalogs all files in the target directory"""

    def __init__(self, root_path, exclusion_patterns=None):
        """
        Initialize the file discovery engine

        Args:
            root_path: Root directory to scan
            exclusion_patterns: List of patterns to exclude (e.g., .git, __pycache__)
        """
        self.root_path = Path(root_path).resolve()
        self.exclusion_patterns = exclusion_patterns or []
        self.discovered_files = []
        self.error_log = []

        if not self.root_path.exists():
            raise ValueError(f"Root path does not exist: {self.root_path}")
        if not self.root_path.is_dir():
            raise ValueError(f"Root path is not a directory: {self.root_path}")

    def _should_exclude(self, path):
        """Check if a path matches any exclusion pattern"""
        path_str = str(path)
        for pattern in self.exclusion_patterns:
            if pattern in path_str:
                return True
        return False

    def classify_file_type(self, filepath):
        """Classify file based on extension"""
        ext = Path(filepath).suffix.lower()

        type_map = {
            # Python files
            '.py': 'PYTHON_SOURCE',
            '.pyc': 'PYTHON_BYTECODE',
            '.pyo': 'PYTHON_BYTECODE',

            # Model artifacts
            '.pkl': 'MODEL_PICKLE',
            '.pickle': 'MODEL_PICKLE',
            '.h5': 'MODEL_KERAS',
            '.keras': 'MODEL_KERAS',
            '.pt': 'MODEL_PYTORCH',
            '.pth': 'MODEL_PYTORCH',
            '.ckpt': 'MODEL_CHECKPOINT',
            '.joblib': 'MODEL_JOBLIB',
            '.weights': 'MODEL_WEIGHTS',

            # Configuration files
            '.json': 'CONFIG_JSON',
            '.yaml': 'CONFIG_YAML',
            '.yml': 'CONFIG_YAML',
            '.toml': 'CONFIG_TOML',
            '.ini': 'CONFIG_INI',
            '.cfg': 'CONFIG_CFG',
            '.env': 'CONFIG_ENV',

            # Data files
            '.csv': 'DATA_CSV',
            '.parquet': 'DATA_PARQUET',
            '.feather': 'DATA_FEATHER',
            '.arrow': 'DATA_ARROW',
            '.xlsx': 'DATA_EXCEL',
            '.xls': 'DATA_EXCEL',

            # Scripts
            '.bat': 'SCRIPT_BATCH',
            '.sh': 'SCRIPT_SHELL',
            '.ps1': 'SCRIPT_POWERSHELL',

            # Documentation
            '.md': 'DOC_MARKDOWN',
            '.txt': 'DOC_TEXT',
            '.rst': 'DOC_RST',

            # Notebooks
            '.ipynb': 'NOTEBOOK',

            # Logs
            '.log': 'LOG_FILE',
        }

        return type_map.get(ext, 'OTHER')

    def extract_metadata(self, filepath):
        """Extract metadata for a single file"""
        try:
            path = Path(filepath)
            stat = path.stat()

            # Calculate relative path from root
            try:
                relative_path = path.relative_to(self.root_path)
            except ValueError:
                relative_path = path

            return {
                'absolute_path': str(path.resolve()),
                'relative_path': str(relative_path),
                'filename': path.name,
                'extension': path.suffix.lower(),
                'file_type': self.classify_file_type(filepath),
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 4),
                'created_date': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        except PermissionError as e:
            self.error_log.append((filepath, 'PermissionError', str(e)))
            return None
        except FileNotFoundError as e:
            self.error_log.append((filepath, 'FileNotFoundError', str(e)))
            return None
        except OSError as e:
            self.error_log.append((filepath, 'OSError', str(e)))
            return None

    def discover_all_files(self):
        """Recursively discover all files in the root directory"""
        print(f"Discovering files in: {self.root_path}")
        file_count = 0

        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)

            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude(root_path / d)]

            for filename in files:
                filepath = root_path / filename

                # Skip excluded files
                if self._should_exclude(filepath):
                    continue

                # Extract metadata
                metadata = self.extract_metadata(filepath)
                if metadata:
                    self.discovered_files.append(metadata)
                    file_count += 1

                    if file_count % 100 == 0:
                        print(f"  Discovered {file_count} files...")

        print(f"Discovery complete: {file_count} files found")
        return self.discovered_files

    def export_inventory(self, output_path):
        """Export file inventory to CSV and JSON"""
        if not self.discovered_files:
            raise ValueError("No files discovered yet. Run discover_all_files() first.")

        # Create DataFrame
        df = pd.DataFrame(self.discovered_files)

        # Sort by relative path
        df = df.sort_values('relative_path').reset_index(drop=True)

        # Export to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        csv_path = output_path
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Inventory exported to: {csv_path}")

        # Export to JSON
        json_path = output_path.parent / (output_path.stem + '.json')
        df.to_json(json_path, orient='records', indent=2)
        print(f"Inventory exported to: {json_path}")

        # Export error log if there are errors
        if self.error_log:
            error_df = pd.DataFrame(self.error_log, columns=['filepath', 'error_type', 'error_message'])
            error_path = output_path.parent / 'discovery_errors.csv'
            error_df.to_csv(error_path, index=False)
            print(f"Error log exported to: {error_path}")

        return df


if __name__ == '__main__':
    # Test on a small directory
    test_root = Path(__file__).parent.parent
    exclusion_patterns = ['.git', '__pycache__', '.pytest_cache', '.pyc']

    discovery = FileDiscovery(test_root, exclusion_patterns)
    files = discovery.discover_all_files()

    print(f"\nDiscovered {len(files)} files")
    print(f"Errors encountered: {len(discovery.error_log)}")

    if files:
        print(f"\nSample file:")
        print(json.dumps(files[0], indent=2))
