from pydrake.all import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    Cost,
    Binding,
    Point as Point,
    HPolyhedron,
    L2NormCost,
    eq,
)
import numpy as np

gcs = GraphOfConvexSets()

m1_A = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
m1_b = [-0.23333045, -0.23333045]

m2_A = [[-0.70710678, 0.0, 0.70710678, 0.0], [0.0, -0.70710678, 0.0, 0.70710678]]
m2_b = [-1.24921309, -1.24921309]

sets = [
    Point([-1, -1, -1, -1]),
    HPolyhedron(m1_A, m1_b),
    HPolyhedron(m2_A, m2_b),
    Point([-1.5, -1.5, -0.5, -0.5]),
]

vertex_names = ["s", "m1", "m2", "t"]
gcs_vertices = {name: gcs.AddVertex(set, name) for name, set in zip(vertex_names, sets)}

# Add vertex cost
A = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])
b = np.array([0, 0])
vertex_path_length_cost = L2NormCost(A, b)
for v in gcs_vertices.values():
    binding = Binding[Cost](vertex_path_length_cost, v.x())
    v.AddCost(binding)

edges = {"s": ["m1"], "m1": ["m2", "t"], "m2": ["m1"]}

gcs_edges = []

print("Edge Constraints:")
for u, vs in edges.items():
    for v in vs:
        e = gcs.AddEdge(gcs_vertices[u], gcs_vertices[v])
        gcs_edges.append(e)
        # Add constant edge cost
        e.AddCost(1)
        # Add edge position continuity constraint
        xu, xv = e.xu(), e.xv()
        constraints = eq(xu[[1, 3]], xv[[0, 2]])
        for c in constraints:
            print(c)
            e.AddConstraint(c)

options = GraphOfConvexSetsOptions()
options.convex_relaxation = False

result = gcs.SolveShortestPath(
    gcs_vertices["s"],
    gcs_vertices["t"],
    options,
)

assert result.is_success()


def print_results(result, gcs):
    print(f"optimal cost {result.get_optimal_cost()}")
    for e in gcs.Edges():
        print(f"{e.u().name()} -> {e.v().name()}, flow: {result.GetSolution(e.phi())}")
    for v in gcs.Vertices():
        print(v.name(), result.GetSolution(v.x()))


print("\nGraph with m2 solution:")
print_results(result, gcs)
print("This DOES NOT satisfy the edge constraint betwen s and m1, and m1 and t")

print("\nGraph without m2 solution:")
gcs.RemoveVertex(gcs_vertices["m2"])

result = gcs.SolveShortestPath(
    gcs_vertices["s"],
    gcs_vertices["t"],
    options,
)

assert result.is_success()
print_results(result, gcs)
print("This satisfies the edge constraint betwen s and m1, and m1 and t")
