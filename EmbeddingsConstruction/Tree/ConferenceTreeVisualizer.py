import matplotlib.pyplot as plt
import networkx as nx

from anytree import RenderTree, PreOrderIter
from .ConferenceNode import ConferenceNode

class ConferenceTreeVisualizer:
    def __init__(self, root: ConferenceNode):
        self.root = root

    def show_text_tree(self):
        print("\n📂 Conference Tree Structure:\n")
        for pre, _, node in RenderTree(self.root):
            print(f"{pre}{node.name} ({node.node_type})")

    def show_networkx_tree(self, label_angle=30):
        def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.successors(root))
            if len(children) != 0:
                dx = width / len(children)
                nextx = xcenter - width/2 - dx/2
                for child in children:
                    nextx += dx
                    pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                        vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
            return pos

        G = nx.DiGraph()
        for node in PreOrderIter(self.root):
            G.add_node(node.name, type=node.node_type)
            if node.parent:
                G.add_edge(node.parent.name, node.name)

        pos = hierarchy_pos(G, self.root.name)

        plt.figure(figsize=(20, 8))
        nx.draw(
            G, pos,
            with_labels=False,
            arrows=True,
            node_size=2000,
            node_color='lightblue',
            edge_color='gray'
        )

        for node_name, (x, y) in pos.items():
            plt.text(
                x, y, node_name,
                rotation=label_angle,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=9,
                fontweight='bold'
            )

        # plt.title("Conference Tree Structure", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
