import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import networkx as nx
import pandas as pd
import numpy as np


def draw_network(
    G,
    data,
    # Core visualization
    color_by=None,
    color_type='auto',
    
    # Color customization
    palette='tab20',
    cmap='viridis',
    color_map=None,
    
    # Continuous-specific
    vmin=None,
    vmax=None,
    na_color='gray',
    neg_color='pink',
    show_colorbar=True,
    
    # Size
    size_by=None,
    size_scale=1.0,
    size_range=(2, 300),
    default_size=20,
    
    # Node borders
    border_color_by=None,
    border_palette='Set2',
    border_color_map=None,
    border_width=1,
    
    # Labels
    label_by=None,
    label_size=8,
    label_color='black',
    
    # Figure
    figsize=(8, 8),
    alpha=0.9,
    title=None,
    
    # Graph settings
    remove_self_loops=True,
    layout='neato',
    
    # Output
    save_path=None,
    show_legend=False,
    legend_title=None,
    
    **kwargs
):
    """
    Draw a network graph with flexible node styling.
    
    Parameters
    ----------
    G : networkx.Graph
        The graph to visualize
    data : pd.DataFrame
        Node attributes (index should match node IDs)
    color_by : str, optional
        Column name to color nodes by
    color_type : {'auto', 'categorical', 'continuous'}
        How to interpret color_by column
    palette : str
        Matplotlib palette for categorical data
    cmap : str or Colormap
        Matplotlib colormap for continuous data
    color_map : dict, optional
        Manual mapping of values to colors
    vmin : float, optional
        Minimum value for continuous color scale
    vmax : float, optional
        Maximum value for continuous color scale
    na_color : str
        Color for NaN values in continuous data
    neg_color : str
        Color for negative values in continuous data
    show_colorbar : bool
        Whether to show colorbar for continuous data
    size_by : str, optional
        Column name to size nodes by
    size_scale : float
        Scaling factor for node sizes
    size_range : tuple
        (min, max) size for nodes
    default_size : int
        Default node size when size_by is not specified
    border_color_by : str, optional
        Column name for node border colors
    border_palette : str
        Palette for categorical border colors
    border_color_map : dict, optional
        Manual mapping for border colors
    border_width : float
        Width of node borders
    label_by : str, optional
        Column name to use for node labels
    label_size : int
        Font size for node labels
    label_color : str
        Color for node labels
    figsize : tuple
        Figure size as (width, height)
    alpha : float
        Transparency of nodes
    title : str, optional
        Title to display at top of plot
    remove_self_loops : bool
        Whether to remove self-loops from graph
    layout : str
        Layout algorithm: 'neato' (default), or other graphviz layouts 
        ('dot', 'fdp', 'sfdp', 'circo', 'twopi') if pygraphviz is installed.
        NetworkX layouts always available: 'spring', 'kamada_kawai', 'circular', 'random'.
        Falls back to 'spring' if pygraphviz not installed.
    save_path : str, optional
        Path to save figure
    show_legend : bool
        Whether to display legend
    legend_title : str, optional
        Title for the legend
        
    Returns
    -------
    dict or None
        Color mapping if color_by is specified, else None
    """
    
    # Remove self-loops if requested
    if remove_self_loops:
        G.remove_edges_from(nx.selfloop_edges(G))
    
    print(f"{nx.number_connected_components(G)} connected components")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute layout
    if layout in ['spring', 'kamada_kawai', 'circular', 'random']:
        # Use NetworkX built-in layouts
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G, seed=42)
    else:
        # Try graphviz layouts (neato, dot, fdp, sfdp, circo, twopi, etc.)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog=layout)
        except ImportError:
            print(f"Warning: pygraphviz not installed. '{layout}' layout unavailable.")
            print(f"         Falling back to 'spring' layout.")
            print(f"         Install pygraphviz for neato/dot/fdp layouts: pip install pygraphviz")
            pos = nx.spring_layout(G, seed=42)
    
    # Determine color type if auto
    if color_by is not None and color_type == 'auto':
        color_type = _infer_color_type(data[color_by])
    
    # Handle node colors
    node_colors = None
    key_to_color = None
    norm = None
    color_mapper = None
    
    if color_by is not None:
        if color_type == 'categorical':
            node_colors, key_to_color = _get_categorical_colors(
                G, data, color_by, palette, color_map
            )
        elif color_type == 'continuous':
            node_colors, norm, color_mapper = _get_continuous_colors(
                G, data, color_by, cmap, vmin, vmax, na_color, neg_color
            )
    else:
        node_colors = ['#888888'] * len(G.nodes)
    
    # Handle node sizes
    node_sizes = _get_node_sizes(
        G, data, size_by, size_scale, size_range, default_size
    )
    
    # Handle border colors
    edge_colors = None
    border_key_to_color = None
    if border_color_by is not None:
        edge_colors, border_key_to_color = _get_categorical_colors(
            G, data, border_color_by, border_palette, border_color_map
        )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3, ax=ax)
    
    # Draw nodes
    nx.draw(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=edge_colors if edge_colors else None,
        alpha=alpha,
        linewidths=border_width,
        ax=ax,
        **kwargs
    )
    
    # Draw node labels if requested
    if label_by is not None:
        labels = {node: str(data.loc[node, label_by]) if node in data.index else str(node) 
                  for node in G.nodes}
        nx.draw_networkx_labels(
            G, pos, labels,
            font_size=label_size,
            font_color=label_color,
            ax=ax
        )
    
    # Add colorbar for continuous data
    if color_type == 'continuous' and show_colorbar and color_mapper is not None:
        sm = plt.cm.ScalarMappable(cmap=color_mapper, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.01, pad=0.01)
        cbar.set_label(color_by if legend_title is None else legend_title)
    
    ax.axis('off')
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Show legend for categorical data
    legend_path = None
    if show_legend and color_type == 'categorical' and key_to_color is not None:
        legend_path = _save_legend(
            key_to_color, 
            title=legend_title if legend_title else color_by,
            save_path=save_path
        )
    
    border_legend_path = None
    if show_legend and border_color_by is not None and border_key_to_color is not None:
        border_legend_path = _save_legend(
            border_key_to_color, 
            title=f"{border_color_by} (border)",
            save_path=save_path,
            suffix="_border"
        )
    
    # Save or show
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        if legend_path:
            print(f"Legend saved to {legend_path}")
        if border_legend_path:
            print(f"Border legend saved to {border_legend_path}")
    else:
        plt.show()
    
    plt.close(fig)
    
    return key_to_color


def _infer_color_type(series):
    """Infer whether data is categorical or continuous."""
    if pd.api.types.is_numeric_dtype(series):
        # Check if it looks categorical (few unique values)
        n_unique = series.nunique()
        if n_unique <= 20:
            return 'categorical'
        else:
            return 'continuous'
    else:
        return 'categorical'


def _get_categorical_colors(G, data, color_col, palette, color_map):
    """Get categorical color mapping for nodes."""
    # Get unique categories
    categories = data[color_col]
    unique_cats = list(dict.fromkeys(categories))
    
    # Assign colors
    if color_map is not None:
        key_to_color = color_map
    else:
        key_to_color = _assign_colors(unique_cats, palette)
    
    # Map nodes to colors
    node_to_color = {
        node: key_to_color.get(data.loc[node, color_col], '#888888')
        for node in G.nodes if node in data.index
    }
    
    node_colors = [node_to_color.get(n, '#888888') for n in G.nodes]
    
    return node_colors, key_to_color


def _get_continuous_colors(G, data, color_col, cmap, vmin, vmax, na_color, neg_color):
    """Get continuous color mapping for nodes."""
    # Get colormap
    if isinstance(cmap, str):
        color_mapper = plt.cm.get_cmap(cmap)
    else:
        color_mapper = cmap
    
    # Get node values
    node_values = data[color_col].reindex([n for n in G.nodes])
    
    # Determine vmin/vmax
    if vmin is None:
        vmin = node_values.min(skipna=True)
    if vmax is None:
        vmax = node_values.max(skipna=True)
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Map colors
    colors = []
    for v in node_values:
        if pd.isna(v):
            colors.append(na_color)
        elif v < 0 and neg_color is not None:
            colors.append(neg_color)
        else:
            colors.append(color_mapper(norm(v)))
    
    return colors, norm, color_mapper


def _get_node_sizes(G, data, size_by, size_scale, size_range, default_size):
    """Get node sizes."""
    if size_by is None:
        return [default_size] * len(G.nodes)
    
    size_series = data[size_by] * size_scale
    size_series = size_series.clip(*size_range)
    node_size_map = size_series.to_dict()
    node_sizes = [node_size_map.get(n, size_range[0]) for n in G.nodes]
    
    return node_sizes


def _assign_colors(categories, palette='tab20'):
    """Assign colors to categories."""
    unique = list(dict.fromkeys(categories))
    
    if len(unique) > 20:
        # Use custom 50-color palette
        discrete_palette = [
            "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", 
            "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff", 
            "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", 
            "#000080", "#808080", "#FFFFFF", "#000000", "#a9a9a9", "#1aff1a", 
            "#ff1a1a", "#1a1aff", "#ff8c00", "#9932cc", "#ff1493", "#00ced1",
            "#ff6347", "#2e8b57", "#3cb371", "#b0e0e6", "#4682b4", "#d2691e", 
            "#b22222", "#00fa9a", "#dda0dd", "#98fb98", "#cd5c5c", "#20b2aa", 
            "#ff00ff", "#ffb6c1", "#4169e1", "#dc143c", "#00bfff", "#7b68ee", 
            "#5f9ea0", "#ffd700"
        ] * 5
        cmap = ListedColormap(discrete_palette, name='custom50')
    else:
        cmap = plt.get_cmap(palette)
    
    colors = [mcolors.to_hex(cmap(i % cmap.N)) for i in range(len(unique))]
    return dict(zip(unique, colors))


def _save_legend(color_map, title=None, save_path=None, suffix=""):
    """Save a legend for categorical colors to a separate file."""
    handles = [Patch(color=color, label=label) for label, color in color_map.items()]
    n_cols = len(color_map)
    n_rows = 1
    
    if n_cols > 5:
        n_rows = 6
        n_cols = (n_cols + 1) // 6
    
    fig = plt.figure(figsize=(6, 1 * n_rows))
    plt.legend(handles=handles, loc='center', ncol=n_cols, frameon=False, title=title)
    plt.axis('off')
    
    # Generate legend filename
    if save_path:
        # Remove extension and add .legend.png
        import os
        base_path = os.path.splitext(save_path)[0]
        legend_save_path = f"{base_path}{suffix}.legend.png"
        fig.savefig(legend_save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return legend_save_path
    else:
        plt.show()
        plt.close(fig)
        return None