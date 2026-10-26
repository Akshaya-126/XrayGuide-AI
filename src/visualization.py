import plotly.graph_objects as go


def plot_predictions(predictions: dict):
    """Build a bar chart of the model's raw outputs."""
    items = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in items]
    scores = [float(item[1]) for item in items]

    fig = go.Figure(go.Bar(x=scores, y=labels, orientation="h"))
    fig.update_layout(
        title="Pretrained model outputs (not calibrated probabilities)",
        xaxis_title="Raw model score",
        yaxis_title="Model label",
        height=max(420, 24 * len(labels)),
        margin=dict(l=10, r=10, t=55, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig
