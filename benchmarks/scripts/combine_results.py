from pathlib import Path
import json
import csv
import platform
import psutil
import os


def get_system_resources():
    """
    Retrieve system resource information.
    """
    total_ram_mb = int(psutil.virtual_memory().total / (1024 ** 2))
    return {
        'CPU Cores': os.cpu_count(),
        'Total RAM (MB)': total_ram_mb,
        'OS Version': f"{platform.system()} {platform.release()}",
        'Architecture': platform.machine(),
    }


def extract_api_and_scenario(folder_name):
    """
    Extract API and scenario names from the folder name.

    Example:
        "gpt_default" -> ("Gpt", "Default")
    """
    parts = folder_name.split('_')
    api_name = parts[0].capitalize()
    scenario_name = ' '.join(parts[1:]).capitalize() if len(parts) > 1 else 'Default'
    return api_name, scenario_name


def parse_summary_file(summary_path):
    """
    Parse the summary JSON file to extract benchmark metrics.
    """
    try:
        with summary_path.open('r') as f:
            summary_data = json.load(f)
    except Exception as e:
        print(f"Error reading summary file {summary_path}: {e}")
        return {}

    throughput = summary_data.get('results_request_output_throughput_token_per_s_mean')
    ttft_ms = summary_data.get('results_ttft_s_mean', 0) * 1000
    end_to_end_ms = summary_data.get('results_end_to_end_latency_s_mean', 0) * 1000

    return {
        'Throughput (Tokens/sec)': throughput,
        'Latency (Time to First Token - ms)': ttft_ms,
        'Latency (Time to Complete Response - ms)': end_to_end_ms,
    }


def parse_system_metrics(metrics_path):
    """
    Parse the system metrics JSON file to extract CPU and memory usage.
    Updated for the new container metrics format:
      {
        "max_cpu_percent": <value>,
        "max_memory_mb": <value>
      }
    """
    try:
        with metrics_path.open('r') as f:
            system_data = json.load(f)
    except Exception as e:
        print(f"Error reading system metrics file {metrics_path}: {e}")
        return {}

    metrics = {}
    if "max_cpu_percent" in system_data:
        metrics["CPU Usage (%)"] = system_data["max_cpu_percent"]
    if "max_memory_mb" in system_data:
        # Convert memory from MB to GB
        metrics["RAM Usage (GB)"] = system_data["max_memory_mb"] / 1024
    return metrics


def parse_results(directory):
    """
    Walk through the results directory, parse benchmark and system metrics,
    and organize the data by API and scenario.
    """
    results = {}
    directory = Path(directory)

    for api_folder in directory.iterdir():
        if not api_folder.is_dir():
            continue

        api_name, scenario_name = extract_api_and_scenario(api_folder.name)
        results.setdefault(api_name, {})[scenario_name] = {}

        # Parse summary file (if present)
        summary_files = list(api_folder.glob('*summary.json'))
        if summary_files:
            summary_metrics = parse_summary_file(summary_files[0])
            results[api_name][scenario_name].update(summary_metrics)

        # Parse system metrics file (if present)
        system_metrics_file = api_folder / 'container_metrics.json'
        if system_metrics_file.exists():
            sys_metrics = parse_system_metrics(system_metrics_file)
            results[api_name][scenario_name].update(sys_metrics)

    return results


def write_csv(results, output_file):
    """
    Write the aggregated benchmark results and system information to a CSV file.
    """
    metrics = [
        'Throughput (Tokens/sec)',
        'Latency (Time to First Token - ms)',
        'Latency (Time to Complete Response - ms)',
        'CPU Usage (%)',
        'RAM Usage (GB)'
    ]

    # Sort API names and collect unique scenario names.
    api_names = sorted(results.keys())
    scenarios = sorted({
        scenario for api_data in results.values() for scenario in api_data
    })

    output_file = Path(output_file)
    with output_file.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write system information.
        writer.writerow(['System Info', 'Value'])
        for key, value in get_system_resources().items():
            writer.writerow([key, value])
        writer.writerow([])  # Blank row for separation.

        # Write CSV header for the benchmark metrics.
        header = ['Metric', 'Scenario'] + api_names
        writer.writerow(header)

        # Write data rows for each metric and scenario.
        for metric in metrics:
            for scenario in scenarios:
                row = [metric, scenario]
                for api in api_names:
                    row.append(results.get(api, {}).get(scenario, {}).get(metric, 'N/A'))
                writer.writerow(row)


def main():
    """
    Main function to parse the results and generate the CSV report.
    """
    results_dir = Path('./results')
    output_csv = results_dir / 'aggregated_benchmarks.csv'

    results = parse_results(results_dir)
    write_csv(results, output_csv)
    print(f"Results saved to {output_csv}")


if __name__ == '__main__':
    main()
