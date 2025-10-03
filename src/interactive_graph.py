import pickle
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.offline as pyo
import webbrowser
from pathlib import Path
from loguru import logger
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class InteractiveKnowledgeGraph:
    """
    Create and display interactive knowledge graph visualization.
    """
    
    def __init__(self, graph_path='data/output/cancer_knowledge_graph.pkl'):
        """Load the knowledge graph."""
        logger.info("Loading knowledge graph for interactive visualization...")
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Load additional data
        self.metadata_df = pd.read_csv('data/output/kg_patient_metadata.csv')
        self.protein_info_df = pd.read_csv('data/output/kg_protein_info.csv')
        
        logger.info(f"Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # Separate node types
        self.nodes_by_type = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            self.nodes_by_type[data['node_type']].append(node)
    
    def create_interactive_visualization(self, output_file='data/output/interactive_knowledge_graph.html'):
        """Create an interactive visualization using Plotly."""
        logger.info("Creating interactive visualization...")
        
        # Create a focused subgraph for better interactivity
        key_proteins = self.protein_info_df[self.protein_info_df['is_key_protein']]['protein'].tolist()
        key_protein_nodes = [f'protein_{p}' for p in key_proteins]
        
        # Get MSI-H patients
        msi_h_patients = []
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'patient' and data.get('is_msi_h') == 1:
                msi_h_patients.append(node)
        
        # Create subgraph with focused nodes
        interactive_nodes = (key_protein_nodes + msi_h_patients + 
                           self.nodes_by_type['pathway'] + 
                           self.nodes_by_type['subtype'] + 
                           self.nodes_by_type['mutation'])
        
        subgraph = self.graph.subgraph(interactive_nodes).copy()
        
        # Create layout
        pos = nx.spring_layout(subgraph, k=3, iterations=100, seed=42)
        
        # Prepare node traces by type
        node_traces = {}
        color_map = {
            'patient': 'red',
            'protein': 'green', 
            'pathway': 'orange',
            'subtype': 'purple',
            'mutation': 'blue'
        }
        
        for node_type in color_map:
            node_traces[node_type] = go.Scatter(
                x=[], y=[], mode='markers+text',
                marker=dict(size=[], color=color_map[node_type], opacity=0.8),
                text=[], textposition="middle center",
                hovertemplate='<b>%{text}</b><br>Type: ' + node_type + '<br>Connections: %{customdata}<extra></extra>',
                name=node_type.title(),
                showlegend=True,
                customdata=[]
            )
        
        # Add nodes to traces
        for node in subgraph.nodes():
            x, y = pos[node]
            node_data = subgraph.nodes[node]
            node_type = node_data['node_type']
            
            node_traces[node_type]['x'] += tuple([x])
            node_traces[node_type]['y'] += tuple([y])
            
            # Node size based on connections
            degree = subgraph.degree(node)
            size = max(15, min(40, degree * 3))
            node_traces[node_type]['marker']['size'] += tuple([size])
            
            # Node label and hover info
            if node_type == 'patient':
                # Get patient subtype for better labeling
                subtype = node_data.get('proteomic_subtype', 'Unknown')
                label = f"MSI-H-{subtype}" if subtype != 'Unknown' else "MSI-H"
                node_traces[node_type]['customdata'] += tuple([f"{degree} (Subtype: {subtype})"])
            else:
                label = node.split('_')[1] if '_' in node else node
                node_traces[node_type]['customdata'] += tuple([str(degree)])
            
            node_traces[node_type]['text'] += tuple([label])
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in subgraph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.8, color='rgba(125,125,125,0.3)'),
            hoverinfo='none', showlegend=False
        )
        
        # Create the figure
        fig = go.Figure(data=[edge_trace] + list(node_traces.values()),
                       layout=go.Layout(
                           title=dict(
                               text='Interactive Cancer Knowledge Graph<br><sub>🎯 Key proteins, MSI-H patients, pathways, subtypes, and mutations</sub>',
                               font=dict(size=16)
                           ),
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=60),
                           annotations=[ 
                               dict(
                                   text="🔴 MSI-H patients | 🟢 Key proteins | 🟠 Pathways | 🟣 Subtypes | 🔵 Mutations",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor='left', yanchor='bottom',
                                   font=dict(color='gray', size=12)
                               ),
                               dict(
                                   text="💡 Hover over nodes for details | 🖱️ Drag to explore | 🔍 Zoom in/out",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.995, y=-0.002,
                                   xanchor='right', yanchor='bottom',
                                   font=dict(color='gray', size=10)
                               )
                           ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white',
                           height=700
                       ))
        
        # Save the interactive plot
        fig.write_html(output_file)
        
        logger.info(f"✅ Interactive graph saved: {output_file}")
        logger.info(f"   Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        
        return fig, output_file
    
    def open_in_browser(self, html_file):
        """Open the interactive graph in the default browser."""
        if Path(html_file).exists():
            abs_path = Path(html_file).absolute()
            file_url = f"file://{abs_path}"
            
            print(f"🌐 Opening interactive knowledge graph in browser...")
            print(f"🔗 URL: {file_url}")
            
            webbrowser.open(file_url)
            
            print("✅ Interactive graph opened!")
            print("\n🎯 WHAT YOU'LL SEE:")
            print("  🔴 Red nodes: MSI-H patients (your classification targets)")
            print("  🟢 Green nodes: Key proteins from Zhang et al. paper")
            print("  🟠 Orange nodes: Biological pathways (DNA repair, EMT, etc.)")
            print("  🟣 Purple nodes: Proteomic subtypes (A, B, C, D, E)")
            print("  🔵 Blue nodes: Key mutations (BRAF, KRAS, TP53, POLE)")
            print("\n💡 INTERACTIVE FEATURES:")
            print("  • Hover over any node to see details")
            print("  • Click and drag nodes to explore connections")
            print("  • Use mouse wheel to zoom in/out")
            print("  • Click legend items to hide/show node types")
            print("\n🎯 ZHANG'S KEY FINDING:")
            print("  Look for MSI-H patients with different protein connection patterns")
            print("  This shows why proteomics distinguishes B (good) vs C (poor) prognosis!")
            
        else:
            print(f"❌ Interactive graph file not found: {html_file}")
    
    def create_and_open(self):
        """Create the interactive visualization and open it in browser."""
        fig, html_file = self.create_interactive_visualization()
        self.open_in_browser(html_file)
        return fig, html_file

def main():
    """Main function to create and open interactive visualization."""
    interactive_kg = InteractiveKnowledgeGraph()
    interactive_kg.create_and_open()

if __name__ == "__main__":
    main()