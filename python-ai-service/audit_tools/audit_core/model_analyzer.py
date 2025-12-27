"""
Model Artifact Analyzer - Analyzes model files and correlates with scripts
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import re


class ModelAnalyzer:
    """Analyzes model artifacts and their relationship with code"""

    def __init__(self, models_directory, training_scripts_list, inference_scripts_list, model_patterns):
        """
        Initialize model analyzer

        Args:
            models_directory: Path to saved models directory
            training_scripts_list: List of training script file paths
            inference_scripts_list: List of inference script file paths
            model_patterns: Dictionary with model naming patterns
        """
        self.models_directory = Path(models_directory)
        self.training_scripts = training_scripts_list
        self.inference_scripts = inference_scripts_list
        self.patterns = model_patterns

        self.discovered_models = []

    def discover_model_artifacts(self):
        """Scan models directory for model files"""
        print(f"Discovering model artifacts in: {self.models_directory}")

        if not self.models_directory.exists():
            print(f"Models directory does not exist: {self.models_directory}")
            return []

        model_extensions = self.patterns.get('file_extensions', {}).keys()

        model_count = 0
        for root, dirs, files in self.models_directory.walk():
            for filename in files:
                filepath = root / filename
                ext = filepath.suffix.lower()

                if ext in model_extensions:
                    metadata = self._extract_model_metadata(filepath)
                    if metadata:
                        self.discovered_models.append(metadata)
                        model_count += 1

        print(f"Discovered {model_count} model artifacts")
        return self.discovered_models

    def _extract_model_metadata(self, filepath):
        """Extract metadata from a model file"""
        try:
            stat = filepath.stat()
            filename = filepath.name
            ext = filepath.suffix.lower()

            # Infer model type
            model_info = self.infer_model_type(filename)

            return {
                'filepath': str(filepath),
                'filename': filename,
                'extension': ext,
                'file_format': self.patterns.get('file_extensions', {}).get(ext, 'unknown'),
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'age_days': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).days,
                'model_type': model_info['model_type'],
                'inferred_symbol': model_info.get('symbol'),
                'inferred_timestamp': model_info.get('timestamp')
            }

        except Exception as e:
            print(f"Error extracting metadata from {filepath}: {e}")
            return None

    def infer_model_type(self, filename):
        """Infer model type from filename patterns"""
        filename_lower = filename.lower()

        # Try to extract symbol (AAPL, TSLA, etc.)
        symbol_match = re.search(r'([A-Z]{2,5})_', filename)
        symbol = symbol_match.group(1) if symbol_match else None

        # Try to extract timestamp
        timestamp_patterns = [
            r'(\d{8})',  # 20241215
            r'(\d{4}-\d{2}-\d{2})',  # 2024-12-15
            r'(\d{14})'  # 20241215143022
        ]

        timestamp = None
        for pattern in timestamp_patterns:
            match = re.search(pattern, filename)
            if match:
                timestamp = match.group(1)
                break

        # Identify model type
        model_type_patterns = self.patterns.get('model_type_patterns', {})

        for model_type, patterns in model_type_patterns.items():
            for pattern in patterns:
                if pattern.lower() in filename_lower:
                    return {
                        'model_type': model_type,
                        'symbol': symbol,
                        'timestamp': timestamp
                    }

        return {
            'model_type': 'UNKNOWN',
            'symbol': symbol,
            'timestamp': timestamp
        }

    def correlate_with_training_scripts(self, model_file):
        """Find training scripts that might have created this model"""
        model_type = model_file['model_type']
        filename = model_file['filename']

        training_patterns = self.patterns.get('training_script_patterns', {}).get(model_type, [])

        matches = []
        for script in self.training_scripts:
            script_lower = script.lower()

            # Check if script name matches model type patterns
            for pattern in training_patterns:
                if pattern.lower() in script_lower:
                    confidence = 0.9
                    matches.append({
                        'script': script,
                        'confidence': confidence,
                        'reason': f"Matches pattern '{pattern}'"
                    })
                    break

            # Check if script mentions the symbol
            if model_file.get('inferred_symbol'):
                if model_file['inferred_symbol'] in script.upper():
                    matches.append({
                        'script': script,
                        'confidence': 0.7,
                        'reason': f"Contains symbol {model_file['inferred_symbol']}"
                    })

        # Return best match if any
        if matches:
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            return matches[0]['script'], matches[0]['confidence']

        return None, 0.0

    def correlate_with_inference_scripts(self, model_file):
        """Find inference scripts that load this model"""
        filename = model_file['filename']
        model_path = model_file['filepath']

        matches = []

        # Check inference scripts for references to this model file
        for script in self.inference_scripts:
            try:
                script_path = Path(script) if not isinstance(script, Path) else script

                # Read script content
                if script_path.exists() and script_path.is_file():
                    with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # Check if filename appears in script
                    if filename in content:
                        matches.append(script)
                        continue

                    # Check if model type loading pattern appears
                    model_type = model_file['model_type']
                    if model_type.lower() in content.lower():
                        # Weak match
                        if model_file.get('inferred_symbol') and model_file['inferred_symbol'] in content.upper():
                            matches.append(script)

            except Exception:
                continue

        return matches

    def detect_orphaned_models(self):
        """Identify models with no associated scripts"""
        print("Detecting orphaned models...")

        orphaned = []

        for model in self.discovered_models:
            training_script, train_conf = self.correlate_with_training_scripts(model)
            inference_scripts = self.correlate_with_inference_scripts(model)

            # Classify orphan severity
            severity = None
            if not training_script and not inference_scripts:
                severity = 'HIGH'  # No training, no inference
            elif not training_script and inference_scripts:
                severity = 'LOW'  # Used in inference but no training script
            elif training_script and not inference_scripts:
                severity = 'MEDIUM'  # Has training but not used

            if severity:
                orphaned.append({
                    'model_file': model['filename'],
                    'model_type': model['model_type'],
                    'size_mb': model['size_mb'],
                    'age_days': model['age_days'],
                    'severity': severity,
                    'has_training_script': training_script is not None,
                    'has_inference_usage': len(inference_scripts) > 0
                })

        print(f"Found {len(orphaned)} orphaned models")
        return orphaned

    def generate_model_status_matrix(self):
        """Generate comprehensive status for all models"""
        print("Generating model status matrix...")

        status_matrix = []

        for model in self.discovered_models:
            training_script, train_conf = self.correlate_with_training_scripts(model)
            inference_scripts = self.correlate_with_inference_scripts(model)

            # Determine status
            if training_script and inference_scripts:
                status = 'ACTIVE'
                recommendation = 'KEEP'
            elif training_script and not inference_scripts:
                status = 'TRAINING_ONLY'
                recommendation = 'REVIEW'  # Maybe experimental
            elif not training_script and inference_scripts:
                status = 'INFERENCE_ONLY'
                recommendation = 'REVIEW'  # Training code might be lost
            else:
                status = 'ORPHANED'
                # Decide based on age and size
                if model['age_days'] > 180 and model['size_mb'] > 10:
                    recommendation = 'DELETE'
                elif model['age_days'] > 90:
                    recommendation = 'ARCHIVE'
                else:
                    recommendation = 'REVIEW'

            status_matrix.append({
                'model_file': model['filename'],
                'model_type': model['model_type'],
                'symbol': model.get('inferred_symbol', 'N/A'),
                'size_mb': model['size_mb'],
                'modified_date': model['modified_date'],
                'age_days': model['age_days'],
                'training_script': training_script if training_script else 'None',
                'training_confidence': train_conf,
                'inference_usage': ', '.join(inference_scripts[:3]) if inference_scripts else 'None',
                'inference_count': len(inference_scripts),
                'status': status,
                'recommendation': recommendation
            })

        return pd.DataFrame(status_matrix)

    def export_model_analysis(self, output_dir):
        """Export model analysis results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate status matrix
        status_df = self.generate_model_status_matrix()

        # Export status matrix
        status_path = output_dir / 'model_status_matrix.csv'
        status_df.to_csv(status_path, index=False)
        print(f"Model status matrix exported to: {status_path}")

        # Export orphaned models
        orphaned = self.detect_orphaned_models()
        orphaned_df = pd.DataFrame(orphaned)
        if not orphaned_df.empty:
            orphaned_path = output_dir / 'orphaned_models.csv'
            orphaned_df.to_csv(orphaned_path, index=False)
            print(f"Orphaned models exported to: {orphaned_path}")

        # Export model type statistics
        stats = {
            'total_models': len(self.discovered_models),
            'total_size_mb': sum(m['size_mb'] for m in self.discovered_models),
            'by_type': {},
            'by_status': status_df['status'].value_counts().to_dict(),
            'by_recommendation': status_df['recommendation'].value_counts().to_dict()
        }

        # Group by model type
        for model_type in status_df['model_type'].unique():
            type_df = status_df[status_df['model_type'] == model_type]
            stats['by_type'][model_type] = {
                'count': len(type_df),
                'total_size_mb': type_df['size_mb'].sum(),
                'avg_age_days': type_df['age_days'].mean()
            }

        stats_path = output_dir / 'model_type_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Model statistics exported to: {stats_path}")

        return {
            'status_df': status_df,
            'orphaned': orphaned,
            'stats': stats
        }


if __name__ == '__main__':
    print("ModelAnalyzer module - use via main orchestration script")
