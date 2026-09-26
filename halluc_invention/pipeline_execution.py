import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_stratified_pipeline_analysis(json_data_or_path, num_bootstraps=10000, seed=42):
    """
    Groups entries by active operational domain, runs individual clustered bootstrap
    matrices, and maps the localized error vectors directly to a Matplotlib plot.
    """
    np.random.seed(seed)

    # Handle direct parsing or file systems
    if isinstance(json_data_or_path, str) and json_data_or_path.strip().startswith('{'):
        dataset = json.loads(json_data_or_path)
    elif isinstance(json_data_or_path, str):
        with open(json_data_or_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    else:
        dataset = json_data_or_path

    df = pd.DataFrame(dataset.get("cases", []))
    df['score_invention'] = df['acceptability'].apply(lambda x: 1.0 if x == 'acceptable' else 0.0)

    unique_domains = df['domain'].unique()
    plot_data = []

    print("Executing dynamic cluster resamples stratified across domains...")
    for domain in unique_domains:
        sub_df = df[df['domain'] == domain]
        cluster_ids = sub_df['id'].unique()
        n_clusters = len(cluster_ids)

        # Pre-aggregate true sample rates
        means_invention = sub_df.groupby('id')['score_invention'].mean().to_numpy()
        observed_mean = means_invention.mean()

        # Matrix bootstrap fanout choice vector
        resample_matrix = np.random.choice(n_clusters, size=(num_bootstraps, n_clusters), replace=True)
        boot_distributions = means_invention[resample_matrix].mean(axis=1)

        lower_bound = np.percentile(boot_distributions, 2.5)
        upper_bound = np.percentile(boot_distributions, 97.5)

        # Map localized symmetric error deviations
        error_minus = max(0.0, observed_mean - lower_bound)
        error_plus = max(0.0, upper_bound - observed_mean)

        plot_data.append({
            "domain": domain.replace("_", " ").title(),
            "mean": observed_mean * 100,
            "err_y": [[error_minus * 100], [error_plus * 100]],
            "ci": f"[{lower_bound * 100:.1f}%, {upper_bound * 100:.1f}%]"
        })
        print(f" -> {domain}: {observed_mean * 100:.1f}% {plot_data[-1]['ci']}")

    return plot_data


def generate_matplotlib_chart(plot_data):
    """
    Generates a clean, presentation-ready horizontal error bar chart layout.
    """
    domains = [item["domain"] for item in plot_data]
    means = [item["mean"] for item in plot_data]

    # Reshape err columns into asymmetric format (2, N)
    err_low = [item["err_y"][0][0] for item in plot_data]
    err_high = [item["err_y"][1][0] for item in plot_data]
    asymmetric_error = np.array([err_low, err_high])

    # Configure document plot size properties
    fig, ax = plt.subplots(figsize=(10, n_domains := max(4, len(domains) * 0.8)))

    # Render baseline comparison trace path
    ax.axvline(0, color='#d3d3d3', linestyle='--', linewidth=1.5, label='Standard Generation Baseline (0%)')

    # Draw stratified execution error markers
    bars = ax.errorbar(
        means, domains, xerr=asymmetric_error, fmt='o',
        color='#1f77b4', elinewidth=2.5, capsize=6, capthick=2,
        markersize=8, label='Invention Reinterpretation (95% CI)'
    )

    # Aesthetic chart adjustments
    ax.set_xlabel('Acceptability Yield Rate (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Paired Clustered Bootstrap Error Bounds by Academic Domain', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(-5, 105)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

    # Append precise metric tags to track bars visually
    for i, (m, item) in enumerate(zip(means, plot_data)):
        ax.text(m, i + 0.15, f"{m:.1f}% \n{item['ci']}", ha='center', va='bottom', fontsize=9, color='#333333')

    plt.tight_layout()
    plt.savefig('generated/stratified_error_bounds.png', dpi=300)
    print("Horizontal chart profile exported securely to 'generated/stratified_error_bounds.png'.")


# Secure multi-domain execution execution template stub
if __name__ == "__main__":
    # Access structural inputs directly from user storage
    with open("your_dataset.json", "r", encoding="utf-8") as file:
        dataset_obj = json.load(file)

    stratified_metrics = run_stratified_pipeline_analysis(dataset_obj, num_bootstraps=10000)
    generate_matplotlib_chart(stratified_metrics)
