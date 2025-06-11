import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib import cm
from anytree import PreOrderIter

class TreeAttentionVisualizer:
    def __init__(self, root, node_names, attn_weights):
        self.root = root
        self.node_names = node_names
        self.attn_weights = attn_weights

    def _hierarchy_pos(self, G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        pos = {}
        def _recurse(G, node, width, vert_gap, vert_loc, xcenter, pos):
            children = list(G.successors(node))
            if not children:
                pos[node] = (xcenter, vert_loc)
            else:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _recurse(G, child, dx, vert_gap, vert_loc - vert_gap, nextx, pos)
                pos[node] = (xcenter, vert_loc)
            return pos
        return _recurse(G, root, width, vert_gap, vert_loc, xcenter, pos)

    def show(self, label_angle=30):
        G = nx.DiGraph()
        node_colors = {}
        node_sizes = {}

        norm = Normalize(vmin=0, vmax=max(self.attn_weights))
        cmap = cm.viridis
        attn_dict = dict(zip(self.node_names, self.attn_weights))

        for node in PreOrderIter(self.root):
            G.add_node(node.name)
            if node.parent:
                G.add_edge(node.parent.name, node.name)

            if node.is_leaf and node.name in attn_dict:
                attn = attn_dict[node.name]
                node_colors[node.name] = cmap(norm(attn))
                node_sizes[node.name] = 1500 + 3000 * attn
            else:
                node_colors[node.name] = "#DDDDDD"
                node_sizes[node.name] = 800

        pos = self._hierarchy_pos(G, self.root.name)

        fig, ax = plt.subplots(figsize=(20, 8))
        nx.draw(
            G, pos, ax=ax,
            node_color=[node_colors[n] for n in G.nodes()],
            node_size=[node_sizes[n] for n in G.nodes()],
            edge_color='gray',
            with_labels=False
        )

        for node_name, (x, y) in pos.items():
            ax.text(x, y, node_name,
                    rotation=label_angle,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=9, fontweight='bold')

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Attention weight (leaf nodes only)", fontsize=10)

        ax.set_title("Conference Tree with Attention Weights", fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
