# SHGLNN
Algorithm: Context-Aware Self-Supervised Learning for Heterogeneous Hypergraph with Classification Fine-Tuning

Input: Graph dataset G = {G_1, G_2, ..., G_n}
Output: Graph-level classification predictions Y_pred

1: Initialize model parameters θ

2: for each epoch do
3:     for each graph G_i in G do
4:         # Step 1: Hypergraph Construction
5:         E_intra, E_inter = generate_hyperedges(G_i)
        
6:         # Step 2: Hypergraph Convolutions
7:         H_node, H_edge = hypergraph_convolution(E_intra, E_inter)
        
8:         # Step 3: Context-Aware Attentive Graph-Level Pooling
9:         H_graph = attentive_pooling(H_node, H_edge)
10:    end for
11: end for

12: # Step 4: High-Order-Aware Graph Augmentation
13: G_augmented_list = high_order_augmentation(G)

14: # Step 5: Self-Supervised Contrastive Learning
15: for each G_aug in G_augmented_list do
16:    H_aug_node, H_aug_edge = hypergraph_convolution(E_intra, E_inter)
17:    H_aug_graph = attentive_pooling(H_aug_node, H_aug_edge)
18:    Loss = contrastive_loss(H_graph, H_aug_graph)
19:    θ = optimizer.update(Loss, θ)
20: end for

21: # Step 6: Graph-Level Classification
22: Y_pred = []
23: for each test graph G_test in G_test do
24:    E_intra_test, E_inter_test = generate_hyperedges(G_test)
25:    H_node_test, H_edge_test = hypergraph_convolution(E_intra_test, E_inter_test)
26:    H_graph_test = attentive_pooling(H_node_test, H_edge_test)
27:    Y_pred_test = classifier(H_graph_test)
28:    Y_pred.append(Y_pred_test)
29: end for

30: return Y_pred
