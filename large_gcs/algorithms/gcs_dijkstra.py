from collections import defaultdict
import heapq as heap
import matplotlib.pyplot as plt
from large_gcs.algorithms.search_algorithm import SearchAlgorithm
from large_gcs.graph.graph import Graph


class GcsDijkstra(SearchAlgorithm):
    def __init__(self, graph: Graph):
        self._graph = graph

    def run(self):
        visited = set()
        pq = []
        node_dists = defaultdict(lambda: float("inf"))
        source_name = self._graph.source_name
        target_name = self._graph.target_name

        visited_subgraph = Graph()

        node_dists[source_name] = 0
        heap.heappush(pq, (0, source_name))

        while len(pq) > 0:
            _dist, node = heap.heappop(pq)
            if node in visited:
                continue

            visited.add(node)
            # Add node to the visited subgraph along with all of its incoming and outgoing edges to the visited subgraph
            visited_subgraph.add_vertex(self._graph.vertices[node], node)
            edges = self._graph.incident_edges(node)
            for edge in edges:
                if edge.u in visited and edge.v in visited:
                    visited_subgraph.add_edge(edge)

            if node == source_name:
                visited_subgraph.set_source(source_name)

            print("New subgraph after adding node from heap:")
            plt.figure()
            visited_subgraph.plot_sets()
            visited_subgraph.plot_edges()
            visited_subgraph.plot_set_labels()
            plt.show()

            if node == target_name:
                break
            edges = self._graph.outgoing_edges(node)
            for edge in edges:
                neighbor = edge.v
                if neighbor not in visited:
                    # Add neighbor and edge temporarily to the visited subgraph
                    visited_subgraph.add_vertex(
                        self._graph.vertices[neighbor], neighbor
                    )
                    visited_subgraph.add_edge(edge)
                    visited_subgraph.set_target(neighbor)

                    sol = visited_subgraph.solve_shortest_path()
                    new_dist = sol.cost

                    print("temp subgraph after adding relaxed node:")
                    plt.figure()
                    visited_subgraph.plot_sets()
                    visited_subgraph.plot_edges()
                    visited_subgraph.plot_set_labels()
                    plt.show()

                    # Remove neighbor and associated edges from the visited subgraph
                    visited_subgraph.remove_vertex(neighbor)

                    if new_dist < node_dists[neighbor]:
                        node_dists[neighbor] = new_dist
                        heap.heappush(pq, (new_dist, neighbor))

        # Solve GCS for a final time to extract the path
        visited_subgraph.set_target(target_name)
        sol = visited_subgraph.solve_shortest_path()
        plt.figure(figsize=(5, 5))
        visited_subgraph.plot_sets()
        visited_subgraph.plot_edges()
        visited_subgraph.plot_path(
            sol.path, color="orangered", linestyle=":", linewidth=2
        )
        print(f"Cost: {sol.cost}")
