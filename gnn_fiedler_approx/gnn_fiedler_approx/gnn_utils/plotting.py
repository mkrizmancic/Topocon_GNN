import pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from plotly.colors import find_intermediate_color




def create_combined_histogram(df, bars, line, option="boxplot", title="", xlabel=None):
    """
    Create a chart for visualizing the distribution of a metric across a dataset labels.

    Specifically, this function creates a histogram of the frequency of dataset labels (bars)
    and a line plot of the average error according to some metric (line) across the dataset labels.
    For example, you can see how much the model makes mistakes for poorly or well connected graphs.
    """
    max_val = int(df[bars].max())
    if max_val == 1:
        nbins = 51
        rangebins = (-0.01, max_val + 0.01)
    elif bars == "Nodes":
        nbins = max_val
        rangebins = (0.5, max_val + 0.5)
    else:
        nbins = int(df[bars].max() * 5) + 1
        rangebins = (-0.1, df[bars].max() + 0.1)

    df[bars] = df[bars].round(5)

    hist = np.histogram(df[bars], range=rangebins, bins=nbins)
    bin_edges = np.round(hist[1], 3)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    temp_df = df[[bars, line]].copy()
    temp_df["bin"] = pd.cut(df[bars], bins=bin_edges) # type: ignore
    stats = temp_df.groupby("bin", observed=False)[line].agg(['mean', 'std'])
    mean_per_bin = stats['mean']
    std_per_bin = stats['std']

    fig = go.Figure()
    xlabel = bars if xlabel is None else xlabel

    # Add histogram (count)
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist[0],
            name="Num. examples",
            hovertemplate="Range: %{customdata[0]} - %{customdata[1]}<br>Count: %{y}<extra></extra>",
            customdata=[(str(round(bin_edges[i], 3)), str(round(bin_edges[i + 1], 3))) for i in range(len(bin_edges) - 1)],
        )
    )

    if option in ["mean", "std", "ci"]:
        def _add_mean_line(name_suffix="", line_width=2):
            """Helper function to add the mean line trace"""
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=mean_per_bin,
                mode='lines+markers',
                name=f"Mean {line}" + (f" {name_suffix}" if name_suffix else ""),
                line=dict(color=px.colors.qualitative.Plotly[1], width=line_width),
                marker=dict(size=8),
                yaxis="y2"
            ))

        def _add_error_bars(upper_bound, lower_bound, error_name):
            """Helper function to add asymmetric error bars"""
            # Calculate asymmetric error arrays
            error_upper = np.array(upper_bound) - np.array(mean_per_bin)
            error_lower = np.array(mean_per_bin) - np.array(lower_bound)

            # Add scatter plot with error bars
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=mean_per_bin,
                mode='lines+markers',
                name=f"Mean {line}",
                line=dict(color=px.colors.qualitative.Plotly[1], width=2),
                marker=dict(size=8),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=error_upper,
                    arrayminus=error_lower,
                    visible=True,
                    color=px.colors.qualitative.Plotly[1],
                    width=3
                ),
                yaxis="y2",
                hovertemplate=(
                    f'{xlabel}: %{{x}}<br>' +
                    f'Mean {line}: %{{y:.3f}}<br>' +
                    f'{error_name}<br>' +
                    '<extra></extra>'
                )
            ))

        def _update_dual_axis_layout(y2_title):
            """Helper function to update layout for dual y-axes"""
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title="Count",
                yaxis2=dict(
                    title=y2_title,
                    overlaying='y',
                    side='right'
                ),
                bargap=0,
            )

        # Handle different options
        if option == "mean":
            _add_mean_line()
            _update_dual_axis_layout(f"Average {line}")

        elif option == "std":
            upper_bound = mean_per_bin + std_per_bin
            lower_bound = mean_per_bin - std_per_bin

            _add_error_bars(upper_bound, lower_bound, "Mean ± Std Deviation")
            _update_dual_axis_layout(f"{line} Metrics")

        elif option == "ci":
            from scipy import stats

            # Calculate 95% confidence intervals for each bin
            confidence_level = 0.95
            ci_lower = []
            ci_upper = []

            for bin_label in temp_df["bin"].cat.categories:
                bin_data = temp_df.loc[temp_df["bin"] == bin_label, line]

                if len(bin_data) > 1:
                    # Calculate standard error
                    mean_val = bin_data.mean()
                    std_val = bin_data.std()
                    count = len(bin_data)
                    se = std_val / np.sqrt(count)

                    # Get t-critical value for 95% CI
                    t_critical = stats.t.ppf((1 + confidence_level) / 2, count - 1)
                    margin_error = t_critical * se

                    ci_lower.append(mean_val - margin_error)
                    ci_upper.append(mean_val + margin_error)
                else:
                    # If only one data point or no data, use the mean value
                    mean_val = bin_data.mean() if len(bin_data) > 0 else np.nan
                    ci_lower.append(mean_val)
                    ci_upper.append(mean_val)

            _add_error_bars(ci_upper, ci_lower, "Mean with 95% Confidence Interval")
            _update_dual_axis_layout(f"{line} with 95% CI")

    elif option == "boxplot":
        # Add boxplots for errors
        only_first = True
        for i, bin_label in enumerate(temp_df["bin"].cat.categories):
            # Extract errors for the current bin
            bin_errors = temp_df.loc[temp_df["bin"] == bin_label, line]

            # Add a boxplot for each bin
            fig.add_trace(go.Box(
                y=bin_errors,
                x=[bin_centers[i]] * len(bin_errors),  # Align with bin center
                name="Error",
                marker=dict(color=px.colors.qualitative.Plotly[1]),
                boxmean=True,  # Show mean as a marker
                showlegend=(len(bin_errors) > 0 and only_first),  # Show legend only for the first boxplot
                legendgroup="Error",
                width=(bin_edges[1] - bin_edges[0]) * 0.8,
                yaxis="y2"
            ))

            if len(bin_errors) > 0:
                only_first = False

        # Update layout for the plot
        fig.update_layout(
            title=title,
            xaxis=dict(title=xlabel, tickvals=bin_centers),
            yaxis=dict(title="Count (log)", type="log"),
            yaxis2=dict(
                title=f"{line} Distribution",
                overlaying='y',
                side='right'
            ),
            bargap=0,  # Controls the gap between bars
            barmode='overlay',  # Overlay the bars on top of each other
            boxmode='group',  # Group the boxplots
        )

    return fig


def plot_feature_distribution(nodes_df: pd.DataFrame, feature: str, option: str, save: bool = False):
    plot_options = {
        "spread_markers": (plot_feature_distribution_1, f"min_max_error_{feature}.pdf"),
        "2d_markers": (plot_feature_distribution_2, f"graph_minmax_error_marker_{feature}.pdf"),
        "spread_lines": (plot_feature_distribution_3, f"graph_minmax_error_line_{feature}.pdf"),
        "deviation_markers": (plot_feature_distribution_4, f"graph_error_deviation_{feature}.pdf"),
    }

    if option in plot_options:
        plot_func, fig_name = plot_options[option]
        fig = plot_func(nodes_df, feature)
    else:
        raise ValueError(f"Unknown plotting option: {option}. \nValid options are:\n- " + "\n- ".join(plot_options.keys()))

    fig.show()

    if save:
        image_path = pathlib.Path().cwd().parent / "results" / "feature_influence"
        image_path.mkdir(parents=True, exist_ok=True)
        image_path /= fig_name
        fig.write_image(image_path, width=1920, height=1080)


def plot_feature_distribution_1(nodes_df, feature: str):
    fig = px.scatter(nodes_df, x=f"{feature}_min", y=f"{feature}_max",
                    color="abs(Error %)", # size="Nodes",
                    title="Node Degree vs. Absolute Error Percentage",
                    hover_data={"Nodes": True, "Graph": False, "True": True})
    # fig.update_traces(marker=dict(size=5))

    cs = "Viridis"
    mee = 20   # max expected error
    moef = 5    # max outlier error factor
    cbmr = 0.9  # color bar max range, 90% of max outlier error, 10% for even higher outliers
    fig.update_layout(
        coloraxis=dict(
            cmin=0,
            cmax=moef * mee / cbmr,
            colorscale=[[0, sample_colorscale(cs, [0])[0]],  # zero error
                        [0.5 * cbmr / moef, sample_colorscale(cs, [0.5])[0]],  # 1/2 of max expected error = 10%
                        [1.0 * cbmr / moef, sample_colorscale(cs, [1])[0]],  # max expected error = 20%
                        [1.0 * cbmr, 'rgb(255, 0, 0, 1)'],
                        [1.0 * cbmr, 'rgb(255, 0, 0, 1)'],
                        [1, 'rgb(255, 0, 255, 1)']] #sample_colorscale(cs, [1])[0]]],
            )
        )

    min_subset = nodes_df[nodes_df["Nodes"] <= 10][f"{feature}_min"].min()
    max_subset = nodes_df[nodes_df["Nodes"] <= 10][f"{feature}_max"].max()
    min_whole = nodes_df[f"{feature}_min"].min()
    max_whole = nodes_df[f"{feature}_max"].max()

    fig.add_hline(y=max_subset, line_color="black", line_width=1)
    fig.add_vline(x=min_subset, line_color="black", line_width=1)
    fig.add_vrect(x0=min_whole, x1=min_subset, fillcolor="red", opacity=0.05)
    fig.add_hrect(y0=max_subset, y1=max_whole, fillcolor="red", opacity=0.05)
    return fig


def plot_feature_distribution_2(nodes_df, feature: str):
    plot_df = nodes_df[["Graph", "abs(Error %)", "Nodes", "True"]].copy()
    plot_df[feature] = nodes_df[f"{feature}_min"]
    plot_df2 = nodes_df[["Graph", "abs(Error %)", "Nodes", "True"]].copy()
    plot_df2[feature] = nodes_df[f"{feature}_max"]
    plot_df = pd.concat([plot_df, plot_df2], ignore_index=True)

    fig = px.scatter(plot_df, x="Graph", y=feature,
                    color="abs(Error %)",
                    title="Node Degree vs. Absolute Error Percentage",
                    hover_data={"Nodes": True, "Graph": False, "True": True})
    fig.update_xaxes(showticklabels=False)
    fig.update_traces(marker=dict(size=3))

    cs = "Viridis"
    mee = 20   # max expected error
    moef = 5    # max outlier error factor
    cbmr = 0.9  # color bar max range, 90% of max outlier error, 10% for even higher outliers
    fig.update_layout(
        coloraxis=dict(
            cmin=0,
            cmax=moef * mee / cbmr,
            colorscale=[[0, sample_colorscale(cs, [0])[0]],  # zero error
                        [0.5 * cbmr / moef, sample_colorscale(cs, [0.5])[0]],  # 1/2 of max expected error = 10%
                        [1.0 * cbmr / moef, sample_colorscale(cs, [1])[0]],  # max expected error = 20%
                        [1.0 * cbmr, 'rgb(255, 0, 0, 1)'],
                        [1.0 * cbmr, 'rgb(255, 0, 0, 1)'],
                        [1, 'rgb(255, 0, 255, 1)']] #sample_colorscale(cs, [1])[0]]],
            )
        )
    fig.add_hline(y=nodes_df[nodes_df["Nodes"] <= 10][f"{feature}_max"].max(), line_color="black", line_width=1)
    fig.add_hline(y=nodes_df[nodes_df["Nodes"] <= 10][f"{feature}_min"].min(), line_color="black", line_width=1)

    return fig


def plot_feature_distribution_3(nodes_df, feature: str):
    fig = go.Figure()
    plot_df = nodes_df.copy()

    cs = "Viridis"
    mee = 20   # max expected error
    moef = 5    # max outlier error factor
    cbmr = 0.9  # color bar max range

    # Define color scale breakpoints
    cmax = moef * mee / cbmr

    # Create bins for error percentages
    n_bins = 100
    bin_edges = np.linspace(0, cmax, n_bins + 1)

    # Assign each row to a bin
    plot_df['color_bin'] = pd.cut(plot_df['abs(Error %)'], bins=bin_edges, labels=False, include_lowest=True)
    plot_df['color_bin'] = plot_df['color_bin'].fillna(n_bins - 1)  # Put values > cmax in last bin

    # Sample colorscale at bin positions
    bin_colors = []
    for i in range(n_bins):
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2

        if bin_center <= mee:
            norm_val = bin_center / mee
            color_val = sample_colorscale(cs, [norm_val])[0]
        elif bin_center <= moef * mee:
            norm_val = (bin_center - mee) / (moef * mee - mee)
            color_val = find_intermediate_color(sample_colorscale(cs, [1])[0], 'rgb(255, 0, 0)', norm_val, colortype='rgb')
        else:
            color_val = 'rgb(255, 0, 255)'

        bin_colors.append(color_val)

    # Group by color bin and create one trace per bin
    for bin_idx in sorted(plot_df['color_bin'].unique()):
        bin_data = plot_df[plot_df['color_bin'] == bin_idx]

        # Create x, y coordinates for vertical lines
        x_coords = []
        y_coords = []
        hover_text = []

        for idx, row in bin_data.iterrows():
            x_coords.extend([idx, idx, None])  # x position for start, end, and break
            y_coords.extend([row[f"{feature}_min"], row[f"{feature}_max"], None])
            hover_text.extend([
                f"Graph: {row['Graph']}<br>Nodes: {row['Nodes']}<br>True: {row['True']:.5f}<br>abs(Error %): {row['abs(Error %)']:.2f}<br>{feature}: {row[f'{feature}_min']:.3f}",
                f"Graph: {row['Graph']}<br>Nodes: {row['Nodes']}<br>True: {row['True']:.5f}<br>abs(Error %): {row['abs(Error %)']:.2f}<br>{feature}: {row[f'{feature}_max']:.3f}",
                ""
            ])

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color=bin_colors[int(bin_idx)], width=1),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # no actual data
            mode="markers",
            marker=dict(coloraxis="coloraxis", showscale=True),
            hoverinfo="none",
            showlegend=False
        ))

    # Add horizontal reference lines
    fig.add_hline(y=nodes_df[nodes_df["Nodes"] <= 10][f"{feature}_max"].max(),
                  line_color="black", line_width=1)
    fig.add_hline(y=nodes_df[nodes_df["Nodes"] <= 10][f"{feature}_min"].min(),
                  line_color="black", line_width=1)

    fig.update_xaxes(showticklabels=False, title_text="Graph")
    fig.update_yaxes(title_text=feature)
    fig.update_layout(title="Node Feature Range vs. Absolute Error Percentage")

    fig.update_layout(
        coloraxis=dict(
            colorbar=dict(title="abs(Error %)"),
            cmin=0,
            cmax=moef * mee / cbmr,
            colorscale=[[0, sample_colorscale(cs, [0])[0]],  # zero error
                        [0.5 * cbmr / moef, sample_colorscale(cs, [0.5])[0]],  # 1/2 of max expected error = 10%
                        [1.0 * cbmr / moef, sample_colorscale(cs, [1])[0]],  # max expected error = 20%
                        [1.0 * cbmr, 'rgb(255, 0, 0, 1)'],
                        [1.0 * cbmr, 'rgb(255, 0, 0, 1)'],
                        [1, 'rgb(255, 0, 255, 1)']] #sample_colorscale(cs, [1])[0]]],
            )
        )

    return fig


def plot_feature_distribution_4(nodes_df, feature: str):
    plot_df = nodes_df[["Graph", "abs(Error %)", "Nodes", "True"]].copy()
    feature_max = nodes_df[nodes_df["Nodes"] <= 10][f"{feature}_max"].max()
    feature_min = nodes_df[nodes_df["Nodes"] <= 10][f"{feature}_min"].min()
    plot_df[f"{feature}_max_dev"] = np.maximum(nodes_df[f"{feature}_max"] - feature_max, 0)
    plot_df[f"{feature}_min_dev"] = np.minimum(nodes_df[f"{feature}_min"] - feature_min, 0)

    fig = go.Figure(data=go.Scatter(
        x=plot_df["Graph"],
        y=plot_df["abs(Error %)"],
        mode='markers',
        marker=dict(
            color=plot_df[f"{feature}_max_dev"],
            coloraxis="coloraxis",
            size=5,
            line=dict(
                color=plot_df[f"{feature}_min_dev"],
                coloraxis="coloraxis",
                width=2,
                ),
        ),
        # hover_data={"Nodes": True, "Graph": False, "True": True}
        hovertemplate=
        # "Graph: %{x}<br>abs(Error %): %{y:.2f}<br>" +
        "Nodes: %{customdata[0]}<br>True: %{customdata[1]:.5f}<br>" +
        "Max dev.: %{marker.color:.3f}<br>Min dev.: %{marker.line.color:.3f}<br>" +
        "<extra></extra>",
        customdata=np.column_stack((plot_df["Nodes"], plot_df["True"]))
    ))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(type="log")
    fig.update_layout(
        coloraxis=dict(
            colorscale="Spectral",
            cmid=0,
            colorbar=dict(title=f"{feature} deviation"),
        )
    )

    return fig
