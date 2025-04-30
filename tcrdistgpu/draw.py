import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
# Requires pygraphviz, libgraphviz-dev
import networkx as nx 
import pandas as pd 
import numpy as np 


def draw_categories(
    G,
    data,
    color_col='conditional',
    palette='tab20',
    pallete_list = None,
    width=8,
    height=8,
    size_col=None,
    size_scale = 1.0,
    size_clip=(2, 300),
    remove_self_edges=True,
    edge_color_col=None,  # <- NEW: optional column to map node edge colors
    edge_key_to_color = None,
    legend = False,
    edge_palette='Set2',  # <- NEW: palette for node edge colors
    **kwargs):
    
    if remove_self_edges:
        G.remove_edges_from(nx.selfloop_edges(G))

    print(nx.number_connected_components(G), "connected components")
    plt.figure(1, figsize=(width, height))
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")

    # Color fill (node color)
    node_to_color, key_to_color = get_node_to_color_map(data=data, col=color_col, palette=palette, palette_list = pallete_list)

    # Optional: node border (edgecolor)
    if edge_color_col is not None:
      if edge_key_to_color is not None:
          node_to_edgecolor = data[edge_color_col].map(edge_key_to_color).to_dict()
      elif edge_color_col:
          edge_key_to_color = assign_colors(data[edge_color_col], palette=edge_palette)
          node_to_edgecolor = data[edge_color_col].map(edge_key_to_color).to_dict()
      else:
          node_to_edgecolor = {}
    else:
      node_to_edgecolor = {}


    alpha = kwargs.pop("alpha", 0.9)
    edge_color = kwargs.pop("edge_color", "gray")

    # Draw edges first
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3)

    if size_col:
        size_series = data[size_col]
        size_series = size_series * size_scale
        size_series = size_series.clip(*size_clip)
        node_size_map = size_series.to_dict()
        node_sizes = [node_size_map.get(n, size_clip[0]) for n in G.nodes]
    else:
        node_sizes = [kwargs.pop("node_size", 20)] * len(G.nodes)
    print(node_sizes)
    # Draw nodes
    nx.draw(
        G, pos,
        node_color=[node_to_color.get(n, '#888888') for n in G.nodes],
        edgecolors=[node_to_edgecolor.get(n, '#000000') for n in G.nodes] if edge_color_col else None,
        edge_color = edge_color,
        node_size=node_sizes,
        alpha=alpha,
        linewidths=1,
        **kwargs
    )

    plt.axis('off')
    plt.show()
    if legend:
      show_legend(key_to_color, title=color_col)
      if edge_color_col:
          show_legend(edge_key_to_color, title=edge_color_col)

def draw_continuous(
    G,
    data,
    width=8,
    height=8,
    continuous_color = 'continuous_color',
    cmap = plt.cm.viridis,
    na_color = "gray",
    neg_color = "pink",
    remove_self_edges=True,
    size_col=None,
    size_scale = 1.0,
    size_clip=(2, 300),
    alpha = .5,
    vmin=-2,  # <- NEW
    vmax=3,  # <- NEW
    **kwargs):
    if remove_self_edges:
        G.remove_edges_from(nx.selfloop_edges(G))

    print(nx.number_connected_components(G), "connected components")
    #plt.figure(1, figsize=(width, height))
    fig, ax = plt.subplots(figsize=(width, height))  # <-- create fig and ax


    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    node_values = data[continuous_color].iloc[[i for i in G.nodes]]

    # Normalize non-NaN values
    if vmin is None:
        vmin = node_values.min(skipna=True)
    if vmax is None:
        vmax = node_values.max(skipna=True)

    norm = plt.Normalize(
      vmin=vmin,
      vmax=vmax)

    colors = []
    for v in node_values:
        if np.isnan(v):
            colors.append(na_color)
        elif v < 0:
            colors.append(neg_color)
        else:
            colors.append(cmap(norm(v)))

    if size_col:
        size_series = data[size_col].copy()
        size_series = size_series * size_scale
        size_series = size_series.clip(*size_clip)
        node_size_map = size_series.to_dict()
        node_sizes = [node_size_map.get(n, size_clip[0]) for n in G.nodes]
    else:
        node_sizes = kwargs.pop("node_size", 20)

    # Draw edges first
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3)

    # Draw nodes
    nx.draw(
        G, pos,
        node_color=colors,
        node_size=node_sizes,
        edge_color = "lightgray",
        alpha=alpha,
        linewidths=1,
        **kwargs)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.01, pad=0.01)
    cbar.set_label(continuous_color)

    plt.axis('off')
    plt.show()



def assign_colors(categories, palette = "tab20", palette_list = None):
  unique = list(dict.fromkeys(categories))  # preserves order
  # Your custom 50-color discrete palette
  if palette_list is not None:
    cmap = ListedColormap(palette_list, name='custom50')
  elif len(unique) > 20 or palette == "custom":
    discrete_palette_50 = [
      "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe",
      "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080",
      "#FFFFFF", "#000000", "#a9a9a9", "#1aff1a", "#ff1a1a", "#1a1aff", "#ff8c00", "#9932cc", "#ff1493", "#00ced1",
      "#ff6347", "#2e8b57", "#3cb371", "#b0e0e6", "#4682b4", "#d2691e", "#b22222", "#00fa9a", "#dda0dd", "#98fb98",
      "#cd5c5c", "#20b2aa", "#ff00ff", "#ffb6c1", "#4169e1", "#dc143c", "#00bfff", "#7b68ee", "#5f9ea0", "#ffd700"]
    discrete_palette_50 = discrete_palette_50 * 5
    cmap = ListedColormap(discrete_palette_50, name='custom50')
  else:
    cmap = plt.get_cmap( palette )  # can also use 'Set3', 'tab20', etc.
  colors = [mcolors.to_hex(cmap(i % cmap.N)) for i in range(len(unique))]
  return dict(zip(unique, colors))

def get_node_to_color_map(data, col = 'epitope', palette = 'tab20', palette_list= None):
  node_to_key = {i:key for i,key in zip(data.index, data[col])}
  key_to_color = assign_colors(data[col], palette = palette, palette_list = palette_list)
  node_to_color = {i:key_to_color[key] for i,key in node_to_key.items()}
  return node_to_color, key_to_color

def show_legend(color_map, title=None):
  handles = [Patch(color=color, label=label) for label, color in color_map.items()]
  n_cols = len(color_map)
  n_rows = 1
  if n_cols > 5:
    n_rows = 6
    n_cols = (n_cols + 1) // 6 # Distribute columns across rows evenly

  plt.figure(figsize=(6, 1 * n_rows)) # Adjust figure height based on rows
  plt.legend(handles=handles, loc='center', ncol=n_cols, frameon=False, title=title)
  plt.axis('off')
  plt.show()



# def draw_basic(G, data, color_col = 'epitope', palette = 'tab20', width = 8, height = 8, remove_self_edges = True, **kwargs):
#   if remove_self_edges:
#     G.remove_edges_from(nx.selfloop_edges(G))
#   print(nx.number_connected_components(G), "connected components")
#   plt.figure(1, figsize=(width, height))
#   pos = nx.nx_agraph.graphviz_layout(G, prog="neato") # layout graphs with positions using graphviz neato
#   C = (G.subgraph(c) for c in nx.connected_components(G))

#   node_to_color, key_to_color = get_node_to_color_map(data=data, col = color_col, palette = palette)

# # Provide defaults but allow overrides via kwargs
#   node_size = kwargs.pop("node_size", 20)
#   edge_color = kwargs.pop("edge_color", "gray")
#   alpha = kwargs.pop("alpha", 0.9)
#   for g in C:
#       nx.draw(g, pos,
#               alpha=alpha,
#               node_size= node_size,
#               edge_color = edge_color,
#               node_color = [node_to_color.get(i) for i in g.nodes] ,
#               with_labels=False,
#               **kwargs  )
#   plt.show()
#   show_legend(color_map=key_to_color, title=None)
#   print(key_to_color)
