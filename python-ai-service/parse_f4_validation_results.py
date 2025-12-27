#!/usr/bin/env python3
"""
F4 Validation Results Parser
Parses test logs and creates consolidated JSON and Markdown reports
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Results directory
RESULTS_DIR = Path(__file__).parent / "f4_validation_results"


def extract_regressor_metrics(log_file: Path) -> Dict[str, Any]:
    """Extract metrics from regressor test log"""
    result = {
        "status": "FAILED",
        "sharpe": None,
        "dir_acc": None,
        "mse": None,
        "mae": None,
        "r2": None,
        "warnings": [],
        "errors": [],
        "variance_collapse": False,
        "details": {}
    }

    if not log_file.exists():
        result["errors"].append(f"Log file not found: {log_file}")
        return result

    try:
        content = log_file.read_text()

        # Check for test pass/fail
        if "PASS" in content or "All tests passed" in content or "SUCCESS" in content:
            result["status"] = "PASSED"
        elif "FAIL" in content or "ERROR" in content:
            result["status"] = "FAILED"

        # Extract Sharpe ratio
        sharpe_match = re.search(r"Sharpe[:\s]+([0-9.+-]+)", content, re.IGNORECASE)
        if sharpe_match:
            result["sharpe"] = float(sharpe_match.group(1))

        # Extract directional accuracy
        dir_acc_match = re.search(r"Directional[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if dir_acc_match:
            result["dir_acc"] = float(dir_acc_match.group(1))

        # Extract MSE
        mse_match = re.search(r"MSE[:\s]+([0-9.eE+-]+)", content, re.IGNORECASE)
        if mse_match:
            result["mse"] = float(mse_match.group(1))

        # Extract MAE
        mae_match = re.search(r"MAE[:\s]+([0-9.eE+-]+)", content, re.IGNORECASE)
        if mae_match:
            result["mae"] = float(mae_match.group(1))

        # Extract R2
        r2_match = re.search(r"R2[:\s]+([0-9.+-]+)", content, re.IGNORECASE)
        if r2_match:
            result["r2"] = float(r2_match.group(1))

        # Check for variance collapse
        if "variance collapse" in content.lower() or "std = 0" in content.lower():
            result["variance_collapse"] = True
            result["warnings"].append("Variance collapse detected")

        # Extract warnings
        warning_matches = re.findall(r"WARNING[:\s]+(.+)", content, re.IGNORECASE)
        result["warnings"].extend(warning_matches[:5])  # Limit to 5

        # Extract errors
        error_matches = re.findall(r"ERROR[:\s]+(.+)", content, re.IGNORECASE)
        result["errors"].extend(error_matches[:5])  # Limit to 5

    except Exception as e:
        result["errors"].append(f"Error parsing log: {str(e)}")

    return result


def extract_classifier_metrics(log_file: Path) -> Dict[str, Any]:
    """Extract metrics from classifier test log"""
    result = {
        "status": "FAILED",
        "buy_sharpe": None,
        "sell_sharpe": None,
        "hold_sharpe": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "warnings": [],
        "errors": [],
        "calibration_ok": None,
        "details": {}
    }

    if not log_file.exists():
        result["errors"].append(f"Log file not found: {log_file}")
        return result

    try:
        content = log_file.read_text()

        # Check for test pass/fail
        if "PASS" in content or "All tests passed" in content or "SUCCESS" in content:
            result["status"] = "PASSED"
        elif "FAIL" in content or "ERROR" in content:
            result["status"] = "FAILED"

        # Extract Sharpe ratios for each class
        buy_sharpe_match = re.search(r"Buy.*Sharpe[:\s]+([0-9.+-]+)", content, re.IGNORECASE)
        if buy_sharpe_match:
            result["buy_sharpe"] = float(buy_sharpe_match.group(1))

        sell_sharpe_match = re.search(r"Sell.*Sharpe[:\s]+([0-9.+-]+)", content, re.IGNORECASE)
        if sell_sharpe_match:
            result["sell_sharpe"] = float(sell_sharpe_match.group(1))

        hold_sharpe_match = re.search(r"Hold.*Sharpe[:\s]+([0-9.+-]+)", content, re.IGNORECASE)
        if hold_sharpe_match:
            result["hold_sharpe"] = float(hold_sharpe_match.group(1))

        # Extract accuracy
        acc_match = re.search(r"Accuracy[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if acc_match:
            result["accuracy"] = float(acc_match.group(1))

        # Extract precision
        prec_match = re.search(r"Precision[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if prec_match:
            result["precision"] = float(prec_match.group(1))

        # Extract recall
        recall_match = re.search(r"Recall[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if recall_match:
            result["recall"] = float(recall_match.group(1))

        # Extract F1
        f1_match = re.search(r"F1[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if f1_match:
            result["f1"] = float(f1_match.group(1))

        # Check calibration
        if "calibration ok" in content.lower() or "well calibrated" in content.lower():
            result["calibration_ok"] = True
        elif "calibration" in content.lower() and ("poor" in content.lower() or "failed" in content.lower()):
            result["calibration_ok"] = False
            result["warnings"].append("Calibration issues detected")

        # Extract warnings
        warning_matches = re.findall(r"WARNING[:\s]+(.+)", content, re.IGNORECASE)
        result["warnings"].extend(warning_matches[:5])  # Limit to 5

        # Extract errors
        error_matches = re.findall(r"ERROR[:\s]+(.+)", content, re.IGNORECASE)
        result["errors"].extend(error_matches[:5])  # Limit to 5

    except Exception as e:
        result["errors"].append(f"Error parsing log: {str(e)}")

    return result


def calculate_summary(regressor_results: Dict, classifier_results: Dict) -> Dict[str, Any]:
    """Calculate summary statistics"""
    summary = {
        "regressor_pass_rate": "0/5",
        "classifier_pass_rate": "0/4",
        "avg_regressor_sharpe": None,
        "avg_classifier_sharpe": None,
        "total_warnings": 0,
        "total_errors": 0,
        "symbols_with_variance_collapse": [],
        "symbols_with_calibration_issues": []
    }

    # Regressor pass rate
    regressor_passes = sum(1 for r in regressor_results.values() if r["status"] == "PASSED")
    summary["regressor_pass_rate"] = f"{regressor_passes}/5"

    # Classifier pass rate
    classifier_passes = sum(1 for r in classifier_results.values() if r["status"] == "PASSED")
    summary["classifier_pass_rate"] = f"{classifier_passes}/4"

    # Average regressor Sharpe
    regressor_sharpes = [r["sharpe"] for r in regressor_results.values() if r["sharpe"] is not None]
    if regressor_sharpes:
        summary["avg_regressor_sharpe"] = sum(regressor_sharpes) / len(regressor_sharpes)

    # Average classifier Sharpe (using buy_sharpe as primary metric)
    classifier_sharpes = [r["buy_sharpe"] for r in classifier_results.values() if r["buy_sharpe"] is not None]
    if classifier_sharpes:
        summary["avg_classifier_sharpe"] = sum(classifier_sharpes) / len(classifier_sharpes)

    # Count warnings and errors
    for symbol, result in regressor_results.items():
        summary["total_warnings"] += len(result["warnings"])
        summary["total_errors"] += len(result["errors"])
        if result["variance_collapse"]:
            summary["symbols_with_variance_collapse"].append(symbol)

    for symbol, result in classifier_results.items():
        summary["total_warnings"] += len(result["warnings"])
        summary["total_errors"] += len(result["errors"])
        if result["calibration_ok"] is False:
            summary["symbols_with_calibration_issues"].append(symbol)

    return summary


def generate_markdown_report(regressor_results: Dict, classifier_results: Dict, summary: Dict) -> str:
    """Generate markdown report"""
    report = f"""# F4 Validation Test Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Regressor Tests:** {summary['regressor_pass_rate']} passed
- **Classifier Tests:** {summary['classifier_pass_rate']} passed
- **Average Regressor Sharpe:** {summary['avg_regressor_sharpe']:.3f if summary['avg_regressor_sharpe'] else 'N/A'}
- **Average Classifier Sharpe:** {summary['avg_classifier_sharpe']:.3f if summary['avg_classifier_sharpe'] else 'N/A'}
- **Total Warnings:** {summary['total_warnings']}
- **Total Errors:** {summary['total_errors']}

"""

    if summary['symbols_with_variance_collapse']:
        report += f"\n### ⚠️ Variance Collapse Detected\n"
        report += f"Symbols: {', '.join(summary['symbols_with_variance_collapse'])}\n"

    if summary['symbols_with_calibration_issues']:
        report += f"\n### ⚠️ Calibration Issues Detected\n"
        report += f"Symbols: {', '.join(summary['symbols_with_calibration_issues'])}\n"

    # Regressor results
    report += "\n## Regressor Test Results\n\n"
    report += "| Symbol | Status | Sharpe | Dir Acc | MSE | MAE | R² | Issues |\n"
    report += "|--------|--------|--------|---------|-----|-----|-------|--------|\n"

    for symbol in ["AAPL", "ASML", "IWM", "KO", "MSFT"]:
        result = regressor_results.get(symbol, {})
        status = result.get("status", "N/A")
        status_icon = "✅" if status == "PASSED" else "❌"
        sharpe = f"{result['sharpe']:.3f}" if result.get('sharpe') is not None else "N/A"
        dir_acc = f"{result['dir_acc']:.3f}" if result.get('dir_acc') is not None else "N/A"
        mse = f"{result['mse']:.4f}" if result.get('mse') is not None else "N/A"
        mae = f"{result['mae']:.4f}" if result.get('mae') is not None else "N/A"
        r2 = f"{result['r2']:.3f}" if result.get('r2') is not None else "N/A"

        issues = []
        if result.get('variance_collapse'):
            issues.append("Var collapse")
        if result.get('errors'):
            issues.append(f"{len(result['errors'])} errors")
        if result.get('warnings'):
            issues.append(f"{len(result['warnings'])} warnings")

        issues_str = ", ".join(issues) if issues else "-"

        report += f"| {symbol} | {status_icon} {status} | {sharpe} | {dir_acc} | {mse} | {mae} | {r2} | {issues_str} |\n"

    # Classifier results
    report += "\n## Classifier Test Results\n\n"
    report += "| Symbol | Status | Buy Sharpe | Sell Sharpe | Hold Sharpe | Accuracy | Calibration | Issues |\n"
    report += "|--------|--------|------------|-------------|-------------|----------|-------------|--------|\n"

    for symbol in ["AAPL", "ASML", "IWM", "KO"]:
        result = classifier_results.get(symbol, {})
        status = result.get("status", "N/A")
        status_icon = "✅" if status == "PASSED" else "❌"
        buy_sharpe = f"{result['buy_sharpe']:.3f}" if result.get('buy_sharpe') is not None else "N/A"
        sell_sharpe = f"{result['sell_sharpe']:.3f}" if result.get('sell_sharpe') is not None else "N/A"
        hold_sharpe = f"{result['hold_sharpe']:.3f}" if result.get('hold_sharpe') is not None else "N/A"
        accuracy = f"{result['accuracy']:.3f}" if result.get('accuracy') is not None else "N/A"

        calib = "✅" if result.get('calibration_ok') is True else "❌" if result.get('calibration_ok') is False else "N/A"

        issues = []
        if result.get('calibration_ok') is False:
            issues.append("Calibration")
        if result.get('errors'):
            issues.append(f"{len(result['errors'])} errors")
        if result.get('warnings'):
            issues.append(f"{len(result['warnings'])} warnings")

        issues_str = ", ".join(issues) if issues else "-"

        report += f"| {symbol} | {status_icon} {status} | {buy_sharpe} | {sell_sharpe} | {hold_sharpe} | {accuracy} | {calib} | {issues_str} |\n"

    # Detailed errors and warnings
    if summary['total_errors'] > 0:
        report += "\n## Errors\n\n"
        for symbol, result in {**regressor_results, **classifier_results}.items():
            if result.get('errors'):
                report += f"\n### {symbol}\n"
                for error in result['errors']:
                    report += f"- {error}\n"

    if summary['total_warnings'] > 0:
        report += "\n## Warnings\n\n"
        for symbol, result in {**regressor_results, **classifier_results}.items():
            if result.get('warnings'):
                report += f"\n### {symbol}\n"
                for warning in result['warnings']:
                    report += f"- {warning}\n"

    return report


def main():
    """Main function"""
    print("Parsing F4 Validation Test Results...")
    print("=" * 60)

    # Parse regressor results
    print("\nParsing regressor test results...")
    regressor_results = {}
    for symbol in ["AAPL", "ASML", "IWM", "KO", "MSFT"]:
        log_file = RESULTS_DIR / f"test_regressor_{symbol}.log"
        print(f"  - {symbol}: {log_file.name}")
        regressor_results[symbol] = extract_regressor_metrics(log_file)

    # Parse classifier results
    print("\nParsing classifier test results...")
    classifier_results = {}
    for symbol in ["AAPL", "ASML", "IWM", "KO"]:
        log_file = RESULTS_DIR / f"test_classifier_{symbol}.log"
        print(f"  - {symbol}: {log_file.name}")
        classifier_results[symbol] = extract_classifier_metrics(log_file)

    # Calculate summary
    print("\nCalculating summary statistics...")
    summary = calculate_summary(regressor_results, classifier_results)

    # Create consolidated JSON
    print("\nCreating JSON report...")
    json_output = {
        "generated_at": datetime.now().isoformat(),
        "regressor_results": regressor_results,
        "classifier_results": classifier_results,
        "summary": summary
    }

    json_file = RESULTS_DIR / "VALIDATION_TEST_RESULTS.json"
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"  ✓ Saved: {json_file}")

    # Create markdown report
    print("\nCreating Markdown report...")
    markdown_content = generate_markdown_report(regressor_results, classifier_results, summary)
    markdown_file = RESULTS_DIR / "VALIDATION_TEST_REPORT.md"
    markdown_file.write_text(markdown_content)
    print(f"  ✓ Saved: {markdown_file}")

    print("\n" + "=" * 60)
    print("✅ Report generation complete!")
    print("\nSummary:")
    print(f"  - Regressor Tests: {summary['regressor_pass_rate']} passed")
    print(f"  - Classifier Tests: {summary['classifier_pass_rate']} passed")
    if summary['avg_regressor_sharpe']:
        print(f"  - Avg Regressor Sharpe: {summary['avg_regressor_sharpe']:.3f}")
    if summary['avg_classifier_sharpe']:
        print(f"  - Avg Classifier Sharpe: {summary['avg_classifier_sharpe']:.3f}")
    print(f"  - Total Warnings: {summary['total_warnings']}")
    print(f"  - Total Errors: {summary['total_errors']}")

    if summary['symbols_with_variance_collapse']:
        print(f"\n⚠️  Variance collapse detected in: {', '.join(summary['symbols_with_variance_collapse'])}")
    if summary['symbols_with_calibration_issues']:
        print(f"⚠️  Calibration issues in: {', '.join(summary['symbols_with_calibration_issues'])}")


if __name__ == "__main__":
    main()
