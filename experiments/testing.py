import networkx as nx

G = nx.from_graph6_bytes("[???????????????O?F~`~zN~zn~~n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".encode("ascii"))
print(G.number_of_nodes(), G.number_of_edges())