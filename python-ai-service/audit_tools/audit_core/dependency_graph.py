"""
Dependency Graph Builder - Creates and analyzes dependency graphs using NetworkX
"""
import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


class DependencyGraph:
    """Builds and analyzes file dependency graphs"""

    def __init__(self, import_map, categorization_df):
        """
        Initialize dependency graph builder

        Args:
            import_map: Dictionary mapping files to their imports
            categorization_df: DataFrame with file categorization
        """
        self.import_map = import_map
        self.categorization_df = categorization_df
        self.graph = nx.DiGraph()
        self.metrics = {}

    def build_graph(self):
        """Build the dependency graph from import map"""
        print("Building dependency graph...")

        # Add all Python files as nodes
        for idx, row in self.categorization_df.iterrows():
            if row.get('file_type') == 'PYTHON_SOURCE':
                relative_path = row['relative_path']

                self.graph.add_node(
                    relative_path,
                    category=row.get('category', 'UNKNOWN'),
                    filename=row.get('filename', ''),
                    size_mb=row.get('size_mb', 0)
                )

        # Add edges for LOCAL imports
        edge_count = 0
        for filepath, imports in self.import_map.items():
            for imp in imports:
                if imp.get('source') == 'LOCAL' and imp.get('module'):
                    # Try to resolve the imported module to a file path
                    # This is simplified - real implementation would need module resolution
                    module = imp['module']

                    # Convert module path to file path
                    possible_paths = [
                        f"{module.replace('.', '/')}.py",
                        f"{module.replace('.', '/')}/__init__.py"
                    ]

                    for target_path in possible_paths:
                        if self.graph.has_node(target_path):
                            self.graph.add_edge(filepath, target_path)
                            edge_count += 1
                            break

        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    def calculate_metrics(self):
        """Calculate centrality and importance metrics"""
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Build graph first.")
            return {}

        print("Calculating graph metrics...")

        # In-degree centrality (how many files depend on this)
        in_degree_cent = nx.in_degree_centrality(self.graph)

        # Out-degree centrality (how many dependencies this file has)
        out_degree_cent = nx.out_degree_centrality(self.graph)

        # Betweenness centrality (how critical is this file for connections)
        betweenness_cent = nx.betweenness_centrality(self.graph)

        # PageRank (importance score)
        try:
            pagerank = nx.pagerank(self.graph)
        except:
            pagerank = {node: 0 for node in self.graph.nodes()}

        # Store metrics as node attributes
        for node in self.graph.nodes():
            self.graph.nodes[node]['in_degree'] = in_degree_cent.get(node, 0)
            self.graph.nodes[node]['out_degree'] = out_degree_cent.get(node, 0)
            self.graph.nodes[node]['betweenness'] = betweenness_cent.get(node, 0)
            self.graph.nodes[node]['pagerank'] = pagerank.get(node, 0)

        self.metrics = {
            'in_degree_centrality': in_degree_cent,
            'out_degree_centrality': out_degree_cent,
            'betweenness_centrality': betweenness_cent,
            'pagerank': pagerank
        }

        # Identify top nodes
        top_in_degree = sorted(in_degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 most depended-upon files:")
        for node, score in top_in_degree:
            print(f"  {node}: {score:.3f}")

        return self.metrics

    def detect_cycles(self):
        """Detect circular dependencies"""
        print("Detecting circular dependencies...")

        try:
            cycles = list(nx.simple_cycles(self.graph))
            print(f"Found {len(cycles)} circular dependencies")

            # Classify by severity
            classified_cycles = []
            for cycle in cycles:
                severity = 'LOW' if len(cycle) == 2 else ('MEDIUM' if len(cycle) <= 4 else 'HIGH')

                # Check if any CORE files are involved
                categories = [self.graph.nodes[node].get('category', '') for node in cycle]
                if any('CORE' in cat for cat in categories):
                    severity = 'HIGH'

                classified_cycles.append({
                    'cycle': cycle,
                    'length': len(cycle),
                    'severity': severity,
                    'categories': categories
                })

            return classified_cycles

        except Exception as e:
            print(f"Error detecting cycles: {e}")
            return []

    def detect_orphans(self):
        """Detect orphaned files"""
        print("Detecting orphaned files...")

        orphans = []
        for node in self.graph.nodes():
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)

            # True orphans: no imports and not imported
            if in_deg == 0 and out_deg == 0:
                orphans.append({
                    'filepath': node,
                    'type': 'ISOLATED',
                    'category': self.graph.nodes[node].get('category', '')
                })

            # Potential dead code: not imported but has imports
            elif in_deg == 0 and out_deg > 0:
                # Skip known entry points
                filename = self.graph.nodes[node].get('filename', '')
                if not any(pattern in filename for pattern in ['main.py', 'train_', '__main__']):
                    orphans.append({
                        'filepath': node,
                        'type': 'POTENTIAL_DEAD_CODE',
                        'category': self.graph.nodes[node].get('category', '')
                    })

        print(f"Found {len(orphans)} orphaned files")
        return orphans

    def extract_subgraph(self, category):
        """Extract subgraph for a specific category"""
        # Get nodes of this category
        category_nodes = [
            node for node in self.graph.nodes()
            if self.graph.nodes[node].get('category') == category
        ]

        if not category_nodes:
            return None

        # Extract subgraph
        subgraph = self.graph.subgraph(category_nodes).copy()
        return subgraph

    def visualize_graph(self, output_path, layout="spring", max_nodes=200):
        """Create a visualization of the dependency graph"""
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. Cannot visualize.")
            return

        print(f"Generating graph visualization...")

        # If graph is too large, show only most important nodes
        if self.graph.number_of_nodes() > max_nodes:
            print(f"Graph has {self.graph.number_of_nodes()} nodes, limiting to top {max_nodes}")

            # Get top nodes by PageRank
            pagerank = self.metrics.get('pagerank', {})
            if pagerank:
                top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                top_node_names = [node for node, score in top_nodes]
                graph_to_plot = self.graph.subgraph(top_node_names)
            else:
                # Just take first max_nodes
                nodes = list(self.graph.nodes())[:max_nodes]
                graph_to_plot = self.graph.subgraph(nodes)
        else:
            graph_to_plot = self.graph

        # Create figure
        fig, ax = plt.subplots(figsize=(20, 20))

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(graph_to_plot, k=0.5, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(graph_to_plot)
        elif layout == "hierarchical":
            try:
                pos = nx.nx_agraph.graphviz_layout(graph_to_plot, prog='dot')
            except:
                pos = nx.spring_layout(graph_to_plot)
        else:
            pos = nx.spring_layout(graph_to_plot)

        # Define colors for categories
        color_map = {
            'CORE_TRAINING': '#FF6B6B',
            'CORE_INFERENCE': '#4ECDC4',
            'CORE_EVALUATION': '#45B7D1',
            'CORE_DATA_PIPELINE': '#96CEB4',
            'UTILITY': '#FFEAA7',
            'TEST': '#DFE6E9',
            'DEPRECATED': '#636E72',
            'ORPHAN': '#2D3436',
            'UNKNOWN': '#FD79A8',
            'CONFIG': '#A29BFE',
            'EXPERIMENTAL': '#FDCB6E'
        }

        # Assign colors based on category
        node_colors = []
        for node in graph_to_plot.nodes():
            category = graph_to_plot.nodes[node].get('category', 'UNKNOWN')
            node_colors.append(color_map.get(category, '#95A5A6'))

        # Node sizes based on importance (PageRank)
        node_sizes = []
        for node in graph_to_plot.nodes():
            pagerank = graph_to_plot.nodes[node].get('pagerank', 0)
            # Scale: 100 to 1000
            size = 100 + (pagerank * 5000)
            node_sizes.append(min(size, 1000))

        # Draw graph
        nx.draw_networkx_nodes(
            graph_to_plot, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            ax=ax
        )

        nx.draw_networkx_edges(
            graph_to_plot, pos,
            edge_color='#BDC3C7',
            arrows=True,
            arrowsize=10,
            width=0.5,
            alpha=0.5,
            ax=ax
        )

        # Add labels for high importance nodes
        high_importance_nodes = {
            node: node.split('/')[-1]  # Just filename
            for node in graph_to_plot.nodes()
            if graph_to_plot.nodes[node].get('pagerank', 0) > 0.01
        }

        nx.draw_networkx_labels(
            graph_to_plot, pos,
            labels=high_importance_nodes,
            font_size=6,
            ax=ax
        )

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=cat)
            for cat, color in sorted(color_map.items())
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

        ax.set_title("Dependency Graph", fontsize=16)
        ax.axis('off')

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
        print(f"Graph visualization saved to: {output_path}")
        plt.close()

    def export_graph_data(self, output_dir):
        """Export graph data to various formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export graph to JSON
        graph_data = nx.node_link_data(self.graph)
        json_path = output_dir / 'dependency_graph.json'
        with open(json_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"Graph data exported to: {json_path}")

        # Export node metrics to CSV
        metrics_data = []
        for node in self.graph.nodes():
            metrics_data.append({
                'filepath': node,
                'category': self.graph.nodes[node].get('category', ''),
                'in_degree': self.graph.in_degree(node),
                'out_degree': self.graph.out_degree(node),
                'in_degree_centrality': self.graph.nodes[node].get('in_degree', 0),
                'out_degree_centrality': self.graph.nodes[node].get('out_degree', 0),
                'betweenness': self.graph.nodes[node].get('betweenness', 0),
                'pagerank': self.graph.nodes[node].get('pagerank', 0)
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('pagerank', ascending=False)
        metrics_path = output_dir / 'node_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Node metrics exported to: {metrics_path}")

        # Export cycles
        cycles = self.detect_cycles()
        cycles_path = output_dir / 'circular_dependencies.json'
        with open(cycles_path, 'w') as f:
            json.dump(cycles, f, indent=2)
        print(f"Circular dependencies exported to: {cycles_path}")

        # Export orphans
        orphans = self.detect_orphans()
        orphans_df = pd.DataFrame(orphans)
        if not orphans_df.empty:
            orphans_path = output_dir / 'orphaned_files.csv'
            orphans_df.to_csv(orphans_path, index=False)
            print(f"Orphaned files exported to: {orphans_path}")

        return {
            'metrics_df': metrics_df,
            'cycles': cycles,
            'orphans': orphans
        }


if __name__ == '__main__':
    print("DependencyGraph module - use via main orchestration script")
