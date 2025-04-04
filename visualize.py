import matplotlib.pyplot as plt
import networkx as nx

def plot_birth_death_schematic(n, k, s):
    """Plot schematic birth-death process matching your reference image"""
    plt.figure(figsize=(10, 3))
    G = nx.DiGraph()
    
    # States from 0 to n failures
    states = [str(i) for i in range(n+1)]
    G.add_nodes_from(states)
    
    # Add transitions
    for i in range(n):
        # Failure transitions (→)
        if i < n - s:
            G.add_edge(states[i], states[i+1], label=f"(n-{i})λ" if i < 3 else "...")
        elif i == n - s:
            G.add_edge(states[i], states[i+1], label=f"(n-s)λ")
        else:
            G.add_edge(states[i], states[i+1], label="λ")
    
    for i in range(1, n+1):
        # Repair transitions (←)
        if i <= s:
            G.add_edge(states[i], states[i-1], label=f"{i}μ" if i < 3 else "...")
        else:
            G.add_edge(states[i], states[i-1], label="sμ")
    
    # Position nodes in a line
    pos = {state: (i, 0) for i, state in enumerate(states)}
    
    # Draw elements
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
    nx.draw_networkx_labels(G, pos)
    
    # Draw edges with different styles
    edge_labels = nx.get_edge_attributes(G, 'label')
    curved_edges = [edge for edge in G.edges() if edge[0] != states[0]]
    straight_edges = [edge for edge in G.edges() if edge[0] == states[0]]
    
    nx.draw_networkx_edges(G, pos, edgelist=straight_edges, arrowstyle='->', width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=curved_edges, arrowstyle='->', 
                          width=1.5, connectionstyle='arc3,rad=0.2')
    
    # Add labels with manual adjustments
    for edge, label in edge_labels.items():
        x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
        y = 0.1 if edge[0] < edge[1] else -0.1  # λ above, μ below
        plt.text(x, y, label, ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', pad=1))
    
    # System status indicators
    for i in range(n+1):
        plt.text(pos[states[i]][0], 0.2, 
                "UP" if (n - i) >= k else "DOWN",
                ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='#4daf4a' if (n - i) >= k else '#e41a1c', 
                         edgecolor='black', pad=1))
    
    plt.title(f"Birth-Death Process for {k}-out-of-{n} System (s={s} repairmen)")
    plt.xlim(-0.5, n+0.5)
    plt.ylim(-0.5, 0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
plot_birth_death_schematic(n=5, k=2, s=2)