# Simple Best-First Search

graph = {
    'A': ['B', 'C', 'D'],
    'B': ['E', 'F'],
    'C': ['G'],
    'D': ['H'],
    'E': [],
    'F': [],
    'G': [],
    'H': []
}

heuristic = {
    'A': 5,
    'B': 4,
    'C': 2,
    'D': 6,
    'E': 7,
    'F': 3,
    'G': 1,  # Goal
    'H': 8
}

def best_first_search(start, goal):
    visited = []
    queue = [(heuristic[start], start)]  # (priority, node)

    while queue:
        queue.sort()  # sort by heuristic
        h, node = queue.pop(0)
        print("Visiting:", node)
        visited.append(node)

        if node == goal:
            print("Goal Reached!")
            return

        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append((heuristic[neighbor], neighbor))

# Run
best_first_search('A', 'G')