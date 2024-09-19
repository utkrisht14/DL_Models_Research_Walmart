import matplotlib.pyplot as plt
import wandb


def plot_graph(target, pred, last_lookback_days=7):
    """
    Plot the graph based on the actual and predicted price.
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(target[-last_lookback_days:], label=f"Last {last_lookback_days} Days Actuals")
    ax.plot(pred[-last_lookback_days:], label=f"Last {last_lookback_days} Days Predicted")

    # Dynamically set y-axis limits based on data range
    y_min = min(min(target[-last_lookback_days:]), min(pred[-last_lookback_days:]))
    y_max = max(max(target[-last_lookback_days:]), max(pred[-last_lookback_days:]))
    ax.set_ylim(y_min - 5, y_max + 5)

    ax.set_xlabel(f"Last {last_lookback_days} Days")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"Last {last_lookback_days} Days Comparison")
    plt.legend()


def compare_model_graph(actual_label, window_7, window_20, window_50, window_100, last_lookback_days):
    """
    Compare the result of lookback period of all models in a single graph.
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(actual_label[-last_lookback_days:], label=f"Last {last_lookback_days} Days Actual")
    ax.plot(window_7[-last_lookback_days:], label=f"Predicted - 7 days windows")
    ax.plot(window_20[-last_lookback_days:], label=f"Predicted - 20 days windows")
    ax.plot(window_50[-last_lookback_days:], label=f"Predicted - 50 days windows")
    ax.plot(window_100[-last_lookback_days:], label=f"Predicted - 100 days windows")

    # Dynamically set y-axis limits based on data range
    y_min = min(min(actual_label[-last_lookback_days:]), min(window_7[-last_lookback_days:]),
                min(window_20[-last_lookback_days:]), min(window_50[-last_lookback_days:]),
                min(window_100[-last_lookback_days:]))
    y_max = max(max(actual_label[-last_lookback_days:]), max(window_7[-last_lookback_days:]),
                max(window_20[-last_lookback_days:]), max(window_50[-last_lookback_days:]),
                max(window_100[-last_lookback_days:]))
    ax.set_ylim(y_min - 5, y_max + 5)

    ax.set_xlabel(f"Last {last_lookback_days} Days")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"Last {last_lookback_days} days comparison")
    plt.legend()


def log_model_plot_overall_preds(model_name, overall_targets, overall_preds, last_lookback_days):
    """
    Logs the graph for individual models and their predictions.
    """
    # Start a new W&B run for plotting
    wandb.init(project="Walmart_Research_Plots", name=model_name)

    # Plot the graph
    plot_graph(overall_targets, overall_preds, last_lookback_days=last_lookback_days)

    # Save the plot locally
    plot_filename = f"{model_name.lower().replace(' ', '_')}_plot.png"
    plt.savefig(plot_filename)

    # Log the plot to W&B
    wandb.log({f"{model_name} Plot": wandb.Image(plot_filename)})

    # Close the plot
    plt.close()

    # End the W&B run
    wandb.finish()
