# Self-supervised Heterogeneous Hypergraph Neural Network for Graph-level Classification (SHGLNN)

**Input**: Graph dataset G = {G_1, G_2, ..., G_n}

**Output**: Graph-level classification predictions Y_pred

1: Initialize model parameters θ

2: // Main training loop
   for each epoch do
3:    // Iterate over each graph in the dataset
      for each graph G_i in G do
4:       // Step 1: Hypergraph Construction
         E_intra, E_inter = generate_hyperedges(G_i)
        
5:       // Step 2: Hypergraph Convolutions
         H_node, H_edge = hypergraph_convolution(E_intra, E_inter)
        
6:       // Step 3: Context-Aware Attentive Graph-Level Pooling
         H_graph = attentive_pooling(H_node, H_edge)
      end for
   end for

7: // Step 4: High-Order-Aware Graph Augmentation
   G_augmented_list = high_order_augmentation(G)

8: // Step 5: Self-Supervised Contrastive Learning
   for each G_aug in G_augmented_list do
9:    H_aug_node, H_aug_edge = hypergraph_convolution(E_intra, E_inter)
10:   H_aug_graph = attentive_pooling(H_aug_node, H_aug_edge)
11:   Loss = contrastive_loss(H_graph, H_aug_graph)
12:   θ = optimizer.update(Loss, θ)
   end for

13: // Step 6: Graph-Level Classification
   Y_pred = []
   for each test graph G_test in G_test do
14:   E_intra_test, E_inter_test = generate_hyperedges(G_test)
15:   H_node_test, H_edge_test = hypergraph_convolution(E_intra_test, E_inter_test)
16:   H_graph_test = attentive_pooling(H_node_test, H_edge_test)
17:   Y_pred_test = classifier(H_graph_test)
18:   Y_pred.append(Y_pred_test)
   end for

19: return Y_pred
