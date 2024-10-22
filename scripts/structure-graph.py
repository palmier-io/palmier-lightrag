import json
from typing import Dict, List, Tuple
from sortedcontainers import SortedList
import networkx as nx
import re
import os
import plotly.graph_objects as go
from enum import IntFlag, auto
from functools import total_ordering
import community


class SymbolRole(IntFlag):

    # auto() automatically assigns powers of two (1, 2, 4, 8, etc.) to each enum member
    UnspecifiedSymbolRole = 0
    Definition = auto()
    Import = auto()
    WriteAccess = auto()
    ReadAccess = auto()
    Generated = auto()
    Test = auto()
    ForwardDefinition = auto()

@total_ordering
class Occurrence:
    def __init__(self, file_path: str, range_start: Tuple[int, int], range_end: Tuple[int, int], symbol: str):
        self.file_path = file_path
        self.range_start = range_start
        self.range_end = range_end
        self.symbol = symbol

    def __eq__(self, other):
        if not isinstance(other, Occurrence):
            return NotImplemented
        return (self.file_path, self.range_start, self.range_end, self.symbol) == \
               (other.file_path, other.range_start, other.range_end, other.symbol)

    def __lt__(self, other):
        if not isinstance(other, Occurrence):
            return NotImplemented
        return (self.file_path, self.range_start, self.range_end, self.symbol) < \
               (other.file_path, other.range_start, other.range_end, other.symbol)

    def __repr__(self):
        return f"Occurrence(file_path='{self.file_path}', range_start={self.range_start}, range_end={self.range_end}, symbol='{self.symbol}')"

def get_symbol_type(symbol: str) -> str:
    # Module pattern
    if re.search(r'^[^/]+/$', symbol):
        return 'Module'
    
    # Class pattern
    if re.search(r'[^/]+/[^#]+#$', symbol):
        return 'Class'
    
    # Field pattern
    if re.search(r'[^/]+/[^#]+#[^(]+$', symbol):
        return 'Field'
    
    # Method pattern
    if re.search(r'[^/]+/[^#]+#[^(]+\(\)\.?$', symbol):
        return 'Method'
    
    # Function pattern
    if re.search(r'[^/]+/[^(]+\(\)\.?$', symbol):
        return 'Function'
    
    # Parameter pattern
    if re.search(r'[^/]+/[^(]+\(\)\.\([^)]+\)\.?$', symbol):
        return 'Parameter'
    
    # Constant pattern
    if re.search(r'^[^/]+/[^()#]+\.?$', symbol):
        return 'Constant'
    
    # Local pattern
    if symbol.startswith('local '):
        return 'Local'
    
    # Meta pattern
    if re.search(r'[^/]+/[^/]+:$', symbol):
        return 'Meta'
    
    return 'Unknown'

accepted_symbol_types = ['Class', 'Field', 'Method', 'Function']

# Read the JSON file
with open('scip.json', 'r') as file:
    index = json.load(file)

references = {}
definitions = {}
symbol_types = {}
symbol_occurrences = SortedList()

def get_symbol_roles_str(roles_int: int) -> str:
    roles = SymbolRole(roles_int)
    return ' | '.join(role.name for role in SymbolRole if role in roles and role != SymbolRole.UnspecifiedSymbolRole)

for document in index['documents']:
    for occurrence in document['occurrences']:
        symbol = occurrence.get('symbol', '')
        symbol_type = get_symbol_type(symbol)
        symbol_types[symbol] = symbol_type
        
        ranges = occurrence['range']
        enclosing_range = occurrence.get('enclosingRange', None)
        if enclosing_range:
            range_start = tuple(enclosing_range[:2])
            range_end = tuple(enclosing_range[2:])
        else:
            range_start = tuple(ranges[:2])
            range_end = tuple(ranges[2:]) if len(ranges) == 4 else (ranges[0], ranges[2])
                
        # Populate references and definitions
        if symbol and symbol_type in accepted_symbol_types:
            symbol_occurrences.add(Occurrence(document['relativePath'], range_start, range_end, symbol))

            if occurrence.get('symbolRoles', 0) & 0x1:  # Check if Definition role is set
                if symbol not in definitions:
                    definitions[symbol] = []
                definitions[symbol].append((document['relativePath'], range_start))
            else:
                if symbol not in references:
                    references[symbol] = []
                references[symbol].append((document['relativePath'], range_start))

# Create a graph
G = nx.DiGraph()

# Add nodes (definitions)
for symbol, def_list in definitions.items():
    if symbol_types[symbol] in accepted_symbol_types:
        for file_path, range_start in def_list:
            G.add_node(symbol, type=str(get_symbol_type(symbol)), file_path=str(file_path), range_start=str(range_start))

# Function to check if a reference is within an occurrence's range
def is_within_range(ref_start, occ_start, occ_end):
    return occ_start <= ref_start < occ_end

# Add edges based on references within each occurrence's range
for occurrence in symbol_occurrences:
    if occurrence.symbol in definitions:
        source_nodes = [node for node, attr in G.nodes(data=True) if node == occurrence.symbol and attr.get('file_path', '') == occurrence.file_path]
        
        for source_node in source_nodes:
            # Find all references within this occurrence's range
            for ref_symbol, ref_list in references.items():
                for ref_file, ref_start in ref_list:
                    if ref_file == occurrence.file_path and is_within_range(ref_start, occurrence.range_start, occurrence.range_end):
                        # Find the corresponding definition nodes for the reference
                        target_nodes = [node for node in G.nodes() if node == ref_symbol]
                        for target_node in target_nodes:
                            # Check if the edge already exists before adding it
                            if not G.has_edge(source_node, target_node):
                                G.add_edge(source_node, target_node)
                                print(f"Adding edge from {source_node} to {target_node}")

# Write to GraphML file
nx.write_graphml(G, "symbol_graph.graphml")

print("Graph has been saved as 'symbol_graph.graphml'")

# Analysis and visualization
print(f"Total nodes in graph: {G.number_of_nodes()}")
print(f"Total edges in graph: {G.number_of_edges()}")

node_types = {}
for node, attr in G.nodes(data=True):
    node_type = attr.get('type', 'Unknown')
    node_types[node_type] = node_types.get(node_type, 0) + 1

print("Node types:")
for node_type, count in node_types.items():
    print(f"  {node_type}: {count}")

print("\nSample of edges:")
for i, (source, target) in enumerate(list(G.edges())[:10]):
    print(f"Edge {i}: {source} -> {target}")
    print(f"  Source type: {G.nodes[source].get('type', 'Unknown')}")
    print(f"  Target type: {G.nodes[target].get('type', 'Unknown')}")

# Load the graph from GraphML file
G = nx.read_graphml("symbol_graph.graphml")

# Simplify node labels
for node in G.nodes():
    G.nodes[node]['simplified_name'] = node.split()[-1]

# Cluster nodes by file_path
file_path_groups = {}
for node, data in G.nodes(data=True):
    file_path = data.get('file_path', 'Unknown')
    if file_path not in file_path_groups:
        file_path_groups[file_path] = []
    file_path_groups[file_path].append(node)

# Assign positions to nodes
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Create edge trace
edge_x = []
edge_y = []
edge_text = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_text.append(f"{G.nodes[edge[0]]['simplified_name']} -> {G.nodes[edge[1]]['simplified_name']}")

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='text',
    text=edge_text,
    mode='lines+markers',
    marker=dict(
        symbol='arrow',
        size=10,
        angleref='previous',
        standoff=5
    )
)

# Create node trace
node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# Color nodes by file_path and add node info
node_colors = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_colors.append(len(adjacencies[1]))
    node_info = G.nodes[adjacencies[0]]
    node_text.append(f"Name: {node_info['simplified_name']}<br>"
                     f"Type: {node_info['type']}<br>"
                     f"File: {node_info['file_path']}")

node_trace.marker.color = node_colors
node_trace.text = node_text

# Create the figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Symbol Graph',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Python code: <a href='https://github.com/yourusername/yourrepository'> Your Repository</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[dict(label="Show Arrows",
                                      method="restyle",
                                      args=[{"marker.symbol": "arrow"}]),
                                 dict(label="Hide Arrows",
                                      method="restyle",
                                      args=[{"marker.symbol": "circle"}])]
                    )]
                ))

fig.show()
