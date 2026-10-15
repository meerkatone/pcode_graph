- Implement CDGv2 with BB :

   * Each basic block becomes a dedicated node linked to its operations by Membership edges
   * Control edges go from the block's terminator (or from the block node itself when there is no terminator, as in fallthrough) to the destination block nodes
   * Control edges are kept between nodes whose ordering constraints were not captured by Data edges (load/store and calls)

- Check CDVv2 helps to reducing graph diameter while keeping CFG invariant by instruction reordering. Measure impact on number of edges.
- Create categories for syscall_names in gnn_exporters