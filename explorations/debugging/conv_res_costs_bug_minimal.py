from pydrake.all import GraphOfConvexSets, GraphOfConvexSetsOptions
from pydrake.all import Point as Point

gcs = GraphOfConvexSets()

sets = [
    Point([-5, -1]),
    Point([0, 0]),
    Point([3, 1]),
]

vertex_names = ["s", "m", "t"]
gcs_vertices = {name: gcs.AddVertex(set, name) for name, set in zip(vertex_names, sets)}

# Add vertex cost
for v in gcs_vertices.values():
    v.AddCost(2)

edges = {"s": ["m"], "m": ["t"]}

gcs_edges = []

for u, vs in edges.items():
    for v in vs:
        e = gcs.AddEdge(gcs_vertices[u], gcs_vertices[v])
        gcs_edges.append(e)
        # Add constant edge cost
        e.AddCost(1)

options = GraphOfConvexSetsOptions()
options.convex_relaxation = False

print("standard solve:")
result = gcs.SolveShortestPath(
    gcs_vertices["s"],
    gcs_vertices["t"],
    options,
)

assert result.is_success()


def print_results(result, gcs):
    print(f"optimal cost {result.get_optimal_cost()}")
    for e in gcs.Edges():
        print(
            f"{e.u().name()} -> {e.v().name()}, flow: {result.GetSolution(e.phi())}, cost: {e.GetSolutionCost(result)}"
        )
    for v in gcs.Vertices():
        print(v.name(), result.GetSolution(v.x()), v.GetSolutionCost(result))


print_results(result, gcs)

print("\nConvex restriction solve:")
result = gcs.SolveConvexRestriction(gcs_edges, options)

assert result.is_success()
print_results(result, gcs)
