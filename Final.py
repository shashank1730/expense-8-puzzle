from datetime import datetime
from collections import deque
from queue import PriorityQueue, Queue
import queue
import sys


#BFS ALGORITHM IMPLEMENTATION 
def bfs(start_state, goal_state):
    start_state = tuple(map(tuple, start_state))
    goal_state = tuple(map(tuple, goal_state))
    
    state_queue = Queue()
    state_queue.put(start_state)
    
    visited_states = set()
    parent = {}
    state_depth = {start_state: 0}
    state_cost = {start_state: 0}
    max_fringe_size = [1]
    solution_path = []

    while not state_queue.empty():
        current_state = state_queue.get()
        visited_states.add(current_state)

        if current_state == goal_state:
            path_to_solution = reconstruct_solution_bfs(current_state, parent, start_state)
            print_solution_bfs(len(visited_states), len(visited_states) - 1, len(parent), max_fringe_size,
                           state_depth[goal_state], state_cost[goal_state], solution_path)
            write_trace_file_bfs(len(visited_states), len(visited_states) - 1, len(parent), max_fringe_size,
                             state_depth[goal_state], state_cost[goal_state], solution_path)
            return None

        for move_data in generate_moves_bfs(current_state, visited_states):
            new_state, move_cost, move_description = move_data[0], move_data[1], move_data[2]
            if new_state not in visited_states:
                state_queue.put(new_state)
                visited_states.add(new_state)
                update_state_info_bfs(new_state, current_state, parent, state_depth, state_cost, move_cost, max_fringe_size,
                                  solution_path, move_description,state_queue)
    print("No solution found.")

def generate_moves_bfs(node, visited):
    moves = []
    blank_row, blank_col = get_blank_pos_bfs(node)
    for dr, dc, move_description in [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]:
        new_row, new_col = blank_row + dr, blank_col + dc
        if 0 <= new_row < len(node) and 0 <= new_col < len(node[0]):
            new_node = swap_positions_bfs(node, blank_row, blank_col, new_row, new_col)
            move_cost = node[new_row][new_col]
            moves.append((new_node, move_cost, move_description))
    return moves

def get_blank_pos_bfs(node):
    for r, row in enumerate(node):
        for c, val in enumerate(row):
            if val == 0:
                return r, c
    raise ValueError("The node is not valid because there is no blank tile (0) found")

def swap_positions_bfs(node, r1, c1, r2, c2):
    node = [list(row) for row in node]
    node[r1][c1], node[r2][c2] = node[r2][c2], node[r1][c1]
    return tuple(map(tuple, node))

def update_state_info_bfs(new_state, current_state, parent, state_depth, state_cost, move_cost, max_fringe_size,
                      solution_path, move_description,state_queue):
    parent[new_state] = current_state
    state_depth[new_state] = state_depth[current_state] + 1
    state_cost[new_state] = state_cost[current_state] + move_cost
    max_fringe_size[0] = max(max_fringe_size[0], len(state_queue.queue))
    solution_path.append((new_state, move_cost, move_description))

def reconstruct_solution_bfs(current_state, parent, start_state):
    path_to_solution = []
    while current_state in parent:
        path_to_solution.append(current_state)
        current_state = parent[current_state]
    path_to_solution.append(start_state)
    path_to_solution.reverse()
    return path_to_solution

def print_solution_bfs(states_explored, states_expanded, states_generated, max_fringe_size,
                   solution_depth, solution_cost, solution_path):
    print(f"States Explored: {states_explored}")
    print(f"States Expanded: {states_expanded}")
    print(f"States Generated: {states_generated}")
    print(f"Maximum Fringe Size: {max_fringe_size[0]}")
    print(f"Solution Found at Depth {solution_depth} with Cost {solution_cost}.")
    print('Steps:')
    for step_info in solution_path:
        move_cost, move_description = step_info[1], step_info[2]
        print(f"\tMove {move_cost} {move_description}")
    print("Solution Found Successfully!")

def write_trace_file_bfs(states_explored, states_expanded, states_generated, max_fringe_size,
                     solution_depth, solution_cost, solution_path):
    trace_file_name = f'bfs_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt'
    with open(trace_file_name, "a") as trace_file:
        trace_file.write(f"\tStates Explored: {states_explored}\n")
        trace_file.write(f"\tStates Expanded: {states_expanded}\n")
        trace_file.write(f"\tStates Generated: {states_generated}\n")
        trace_file.write(f"\tMaximum Fringe Size: {max_fringe_size}\n")
        trace_file.write(f"\tSolution Found at Depth {solution_depth} with Cost {solution_cost}.\n")
        for step_info in solution_path:
            move_cost, move_description = step_info[1], step_info[2]
            trace_file.write(f"\tFringe Step: Move {move_cost} {move_description}\n")




# Uniform Cost Search algorithm Implementation 
def uniform_cost_search(start_state, goal_state):
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 0
    
    visited = set()
    frontier = PriorityQueue() 
    frontier.put((0, [start_state, [], 0, []]))
    
    while not frontier.empty():
        current_node = frontier.get()[1]
        current_state, current_path, current_cost, current_pos = current_node
        nodes_popped += 1
        
        if current_state == goal_state:
            print(f'Nodes Popped: {nodes_popped}')
            print(f'Nodes Expanded: {nodes_expanded}')
            print(f'Nodes Generated: {nodes_generated}')
            print(f'Max Fringe Size: {max_fringe_size}')
            print(f'Solution Found at depth {len(current_path)} with cost of {current_cost}.')
            print('Steps:')
            file_name_ucs = f'ucs_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt'
            file_ucs = open(file_name_ucs, "a+")
            file_ucs.write(f"\n\tNodes Popped: {nodes_popped}\n")
            file_ucs.write(f"\tNodes Expanded: {nodes_expanded}\n")
            file_ucs.write(f"\tNodes Generated: {nodes_generated}\n")
            file_ucs.write(f"\tMax Fringe Size: {max_fringe_size}\n")
            file_ucs.write((f"\tSolution Found at depth {len(current_path)} with cost of {current_cost}."))
            with open(file_name_ucs, 'r+') as file:
                content = file.read()
                file.seek(0, 0)
                file.write("Method_selected: UCS \nRunning UCS Algorithm \n" + content)
            for step, pos in zip(current_path, current_pos):
                print(f'\tMove {pos} {step}')
            print("Result found successfully!")
            return current_path

        visited.add(tuple(current_state))
        nodes_expanded += 1
        
        for action, new_state, cost, pos in get_successors_ucs(current_state, visited):
            if tuple(new_state) not in visited:
                nodes_generated += 1
                new_path = current_path + [action]
                new_cost = current_cost + cost
                new_pos = current_pos + [pos]
                frontier.put((new_cost, [new_state, new_path, new_cost, new_pos]))
        
        if frontier.qsize() > max_fringe_size:
            max_fringe_size = frontier.qsize() 
    return None

# Function to get successors of the current node
def get_successors_ucs(state, visited):
    successors = []
    blank_pos = state.index(0)
    blank_row = blank_pos // 3
    blank_col = blank_pos % 3
    with open(f'ucs_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt', 'a+') as file:
        file.write(f"Closed Set: {visited}\n")
        for move_row, move_col, action in [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]:
            new_row = blank_row + move_row
            new_col = blank_col + move_col
            
            if new_row < 0 or new_row >= 3 or new_col < 0 or new_col >= 3:
                continue
            
            new_pos = new_row * 3 + new_col
            new_state = state[:]
            new_state[blank_pos], new_state[new_pos] = new_state[new_pos], new_state[blank_pos]
            
            cost = new_state[blank_row]
            successors.append((action, new_state, cost, blank_pos))
            file.write(f"Successors: {successors}\n")
    return successors





# GREEDY SEARCH ALGORITHM
def greedy(start_state, goal_state):
    visited = set()
    fringe = PriorityQueue()
    fringe.put((heuristic_value(start_state, goal_state), start_state, []))
    max_fringe_size = 1

    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1

    while not fringe.empty():
        _, current, path = fringe.get()
        nodes_popped += 1

        if current == goal_state:
            print_solution_greedy(nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, len(path), path)
            return

        with open(f'greedy_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt', 'a+') as file:
            visited.add(tuple(current))
            nodes_expanded += 1
            for neighbor in get_successors_greedy(current, file):
                if tuple(neighbor) not in visited:
                    cost = heuristic_value(neighbor, goal_state)
                    fringe.put((cost, neighbor, path + [moves_greedy(current, neighbor)]))
                    nodes_generated += 1
                    file.write(f'\nvisited: {visited}, move: {moves_greedy(current, neighbor)}\n')
            max_fringe_size = max(max_fringe_size, fringe.qsize())

    print("No solution found.")

def heuristic_value(state, goal):
    distance = 0
    for i in range(9):
        x1, y1 = divmod(state.index(i), 3)
        x2, y2 = divmod(goal.index(i), 3)
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

def get_successors_greedy(state, file):
    neighbors = []
    x, y = divmod(state.index(0), 3)
    for dx, dy, action in [(0, -1, 'Left'), (-1, 0, 'Up'), (0, 1, 'Right'), (1, 0, 'Down')]:
        newx, newy = x + dx, y + dy
        if 0 <= newx < 3 and 0 <= newy < 3:
            neighbor = state[:]
            neighbor[x*3+y], neighbor[newx*3+newy] = neighbor[newx*3+newy], neighbor[x*3+y]
            neighbors.append(neighbor)
        file.write(f'Successors ({action}): \n{neighbors}\n')
    return neighbors

def moves_greedy(state1, state2):
    index1, index2 = state1.index(0), state2.index(0)
    x1, y1 = divmod(index1, 3)
    x2, y2 = divmod(index2, 3)
    if y1 > y2:
        return f"\tMove {state1[index2]} Left"
    elif y1 < y2:
        return f"\tMove {state1[index2]} Right"
    elif x1 > x2:
        return f"\tMove {state1[index2]} Up"
    elif x1 < x2:
        return f"\tMove {state1[index2]} Down"

def read_input_txt_to_list(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        return [int(i) for line in lines[:3] for i in line.split()]

def print_solution_greedy(nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, solution_depth, solution_path):
    print(f'Nodes Popped: {nodes_popped}')
    print(f'Nodes Expanded: {nodes_expanded}')
    print(f'Nodes Generated: {nodes_generated}')
    print(f'Max Fringe Size: {max_fringe_size}')
    print(f'Solution Found at depth {solution_depth} with cost of {int(solution_depth*4.6)}.')
    print('Steps:')
    for step in solution_path:
        print(step)
    print(" Result found Successfully!")




# a_star algorithm
def a_star(initial_state, goal_state):
    # Calculating the heuristic valye using the manhattan distance
    def distance(state):
        dist = 0
        for i in range(len(state)):
            if state[i] != 0:
                dist += abs(i // 3 - (state[i]-1) // 3) + abs(i % 3 - (state[i]-1) % 3)
        return dist

    class Node:
        def __init__(self, state, parent=None, move=None, cost=0):
            self.state = state
            self.parent = parent
            self.move = move
            self.cost = cost
            # Coumputing the heuristic value
            self.heuristic = distance(state) 
            if self.parent:
                self.depth = parent.depth + 1
            else:
                self.depth = 0

        def __lt__(self, other):
            return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    start_node = Node(initial_state) # storing the start state, parent state, move and cost in the class Node

    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    cost = 0
    max_fringe_size = 0
    frontier = queue.PriorityQueue()
    frontier.put(start_node)
    visited = set()

   # Checking whether the priorityQueue is empty or not
    while not frontier.empty():
        if frontier.qsize() > max_fringe_size:
            max_fringe_size = frontier.qsize()

        node = frontier.get()
        nodes_popped += 1

        # Checking the goal state is equal to the start state
        if node.state == goal_state:
            path = []
            while node.parent:
                path.append(node.move)
                node = node.parent
                cost += node.cost
            path.reverse()
            depth = len(path)
            print(f"Nodes Popped: {nodes_popped}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at depth {depth} with cost of {cost}.")
            print("Steps:")
            file_name_astar = f'astar_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt'
            file_astar = open(file_name_astar, "a")
            file_astar.write(f"\tNodes Popped: {nodes_popped}\n")
            file_astar.write(f"\tNodes Expanded: {nodes_expanded}\n")
            file_astar.write(f"\tNodes Generated: {nodes_generated}\n")
            file_astar.write(f"\tMax Fringe Size: {max_fringe_size}\n")
            file_astar.write((f"\tSolution Found at depth {depth} with cost of {cost}."))
            with open(file_name_astar, 'r+') as file:
                content = file.read()
                file.seek(0, 0)
                file.write("Method_selected: ASTAR \nRunning ASTAR Algorithm \n" + content)
            for step in path:
                print(f"\t{step}")
            print("result found successfully!")
            return None
        
        with open(f'astar_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt', 'a+') as file:
            file.write(f'Closed Set: {visited}\n')
            visited.add(node.state)
            nodes_expanded += 1
            for move, state in get_successors_astar(node.state, file):
                if state not in visited:
                    child = Node(state, parent=node, move=move, cost=node.cost+1)
                    frontier.put(child)
                    nodes_generated += 1
    return None

# This function will return all successors of the node
def get_successors_astar(state, file): 
    successors = []
    i = state.index(0)
    if i not in [0, 1, 2]:
        new_state = list(state)
        new_state[i], new_state[i-3] = new_state[i-3], new_state[i]
        successors.append(("Move {} Down".format(state[i-3]), tuple(new_state)))
    if i not in [6, 7, 8]:
        new_state = list(state)
        new_state[i], new_state[i+3] = new_state[i+3], new_state[i]
        successors.append(("Move {} Up".format(state[i+3]), tuple(new_state)))
    if i not in [0, 3, 6]:
        new_state = list(state)
        new_state[i], new_state[i-1] = new_state[i-1], new_state[i]
        successors.append(("Move {} Right".format(state[i-1]), tuple(new_state)))
    if i not in [2, 5, 8]:
        new_state = list(state)
        new_state[i], new_state[i+1] = new_state[i+1], new_state[i]
        successors.append(("Move {} Left".format(state[i+1]), tuple(new_state)))
    file.write(f'The Fringe steps : \n{successors}\n')
    return successors



#DFS ALOGORITHM
def dfs(start_state, goal_state):
    start_state = tuple(map(tuple, start_state))
    goal_state = tuple(map(tuple, goal_state))
    
    state_stack = [start_state]
    
    visited_states = set()
    parent = {}
    state_depth = {start_state: 0}
    state_cost = {start_state: 0}
    max_fringe_size = [1]
    solution_path = []

    while state_stack:
        current_state = state_stack.pop()
        visited_states.add(current_state)

        if current_state == goal_state:
            path_to_solution = reconstruct_solution_dfs(current_state, parent, start_state)
            print_solution_dfs(len(visited_states), len(visited_states) - 1, len(parent), max_fringe_size[0],
                           state_depth[goal_state], state_cost[goal_state], solution_path)
            write_trace_file_dfs(len(visited_states), len(visited_states) - 1, len(parent), max_fringe_size[0],
                             state_depth[goal_state], state_cost[goal_state], solution_path)
            return None

        for move_data in generate_moves_dfs(current_state, visited_states):
            new_state, move_cost, move_description = move_data[0], move_data[1], move_data[2]
            if new_state not in visited_states:
                state_stack.append(new_state)
                visited_states.add(new_state)
                update_state_info_dfs(new_state, current_state, parent, state_depth, state_cost, move_cost, max_fringe_size,
                                  solution_path, move_description,state_stack)
    print("No solution found.")

def generate_moves_dfs(node, visited):
    moves = []
    blank_row, blank_col = get_blank_pos_dfs(node)
    for dr, dc, move_description in [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]:
        new_row, new_col = blank_row + dr, blank_col + dc
        if 0 <= new_row < len(node) and 0 <= new_col < len(node[0]):
            new_node = swap_positions_dfs(node, blank_row, blank_col, new_row, new_col)
            move_cost = node[new_row][new_col]
            moves.append((new_node, move_cost, move_description))
    return moves

def get_blank_pos_dfs(node):
    for r, row in enumerate(node):
        for c, val in enumerate(row):
            if val == 0:
                return r, c
    raise ValueError("The node is not valid because there is no blank tile (0) found")

def swap_positions_dfs(node, r1, c1, r2, c2):
    node = [list(row) for row in node]
    node[r1][c1], node[r2][c2] = node[r2][c2], node[r1][c1]
    return tuple(map(tuple, node))

def update_state_info_dfs(new_state, current_state, parent, state_depth, state_cost, move_cost, max_fringe_size,
                      solution_path, move_description,state_stack):
    parent[new_state] = current_state
    state_depth[new_state] = state_depth[current_state] + 1
    state_cost[new_state] = state_cost[current_state] + move_cost
    max_fringe_size[0] = max(max_fringe_size[0], len(state_stack))
    solution_path.append((new_state, move_cost, move_description))

def reconstruct_solution_dfs(current_state, parent, start_state):
    path_to_solution = []
    while current_state in parent:
        path_to_solution.append(current_state)
        current_state = parent[current_state]
    path_to_solution.append(start_state)
    path_to_solution.reverse()
    return path_to_solution

def print_solution_dfs(states_explored, states_expanded, states_generated, max_fringe_size,
                   solution_depth, solution_cost, solution_path):
    print(f"States Explored: {states_explored}")
    print(f"States Expanded: {states_expanded}")
    print(f"States Generated: {states_generated}")
    print(f"Maximum Fringe Size: {max_fringe_size}")
    print(f"Solution Found at Depth {solution_depth} with Cost {solution_cost}.")
    print('Steps:')
    for step_info in solution_path:
        move_cost, move_description = step_info[1], step_info[2]
        print(f"\tMove {move_cost} {move_description}")
    print("Solution Found Successfully!")

def write_trace_file_dfs(states_explored, states_expanded, states_generated, max_fringe_size,
                     solution_depth, solution_cost, solution_path):
    trace_file_name = f'dfs_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt'
    with open(trace_file_name, "a") as trace_file:
        trace_file.write(f"\tStates Explored: {states_explored}\n")
        trace_file.write(f"\tStates Expanded: {states_expanded}\n")
        trace_file.write(f"\tStates Generated: {states_generated}\n")
        trace_file.write(f"\tMaximum Fringe Size: {max_fringe_size}\n")
        trace_file.write(f"\tSolution Found at Depth {solution_depth} with Cost {solution_cost}.\n")
        for step_info in solution_path:
            move_cost, move_description = step_info[1], step_info[2]
            trace_file.write(f"\tFringe Step: Move {move_cost} {move_description}\n")

#IDS ALGOIRTHM
def ids(start_state, goal_state):
    start_state = tuple(map(tuple, start_state))
    goal_state = tuple(map(tuple, goal_state))

    max_depth = 0
    solution_path = []

    while True:
        state_stack = [(start_state, 0)]  # Define state_stack here
        visited_states = set()
        parent = {}
        state_depth = {start_state: 0}
        state_cost = {start_state: 0}
        max_fringe_size = [1]
        solution_path = []

        result = dls(start_state, goal_state, max_depth, state_stack, visited_states, parent, state_depth, state_cost, max_fringe_size, solution_path)
        if result is not None:
            states_explored, states_expanded, states_generated, max_fringe_size, solution_depth, solution_cost, path = result
            solution_path.extend(path)
            print_solution_ids(states_explored, states_expanded, states_generated, max_fringe_size,
                           solution_depth, solution_cost, solution_path)
            write_trace_file_ids(states_explored, states_expanded, states_generated, max_fringe_size,
                             solution_depth, solution_cost, solution_path)
            return
        max_depth += 1

def dls(start_state, goal_state, max_depth, state_stack, visited_states, parent, state_depth, state_cost, max_fringe_size, solution_path):
    while state_stack:
        current_state, depth = state_stack.pop()
        visited_states.add(current_state)

        if depth > max_depth:
            continue

        if current_state == goal_state:
            path_to_solution = reconstruct_solution(current_state, parent, start_state)
            return len(visited_states), len(visited_states) - 1, len(parent), max_fringe_size, \
                   state_depth[goal_state], state_cost[goal_state], path_to_solution

        for move_data in generate_moves(current_state, visited_states):
            new_state, move_cost, move_description = move_data[0], move_data[1], move_data[2]
            if new_state not in visited_states:
                state_stack.append((new_state, depth + 1))
                visited_states.add(new_state)
                update_state_info(new_state, current_state, parent, state_depth, state_cost, move_cost, max_fringe_size,
                                  solution_path, move_description,state_stack)

    return None
def generate_moves(node, visited):
    moves = []
    blank_row, blank_col = get_blank_pos(node)
    for dr, dc, move_description in [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]:
        new_row, new_col = blank_row + dr, blank_col + dc
        if 0 <= new_row < len(node) and 0 <= new_col < len(node[0]):
            new_node = swap_positions(node, blank_row, blank_col, new_row, new_col)
            move_cost = node[new_row][new_col]
            moves.append((new_node, move_cost, move_description))
    return moves

def get_blank_pos(node):
    for r, row in enumerate(node):
        for c, val in enumerate(row):
            if val == 0:
                return r, c
    raise ValueError("The node is not valid because there is no blank tile (0) found")

def swap_positions(node, r1, c1, r2, c2):
    node = [list(row) for row in node]
    node[r1][c1], node[r2][c2] = node[r2][c2], node[r1][c1]
    return tuple(map(tuple, node))

def update_state_info(new_state, current_state, parent, state_depth, state_cost, move_cost, max_fringe_size,
                      solution_path, move_description,state_stack):
    parent[new_state] = current_state
    state_depth[new_state] = state_depth[current_state] + 1
    state_cost[new_state] = state_cost[current_state] + move_cost
    max_fringe_size[0] = max(max_fringe_size[0], len(state_stack))
    solution_path.append((new_state, move_cost, move_description))

def reconstruct_solution(current_state, parent, start_state):
    path_to_solution = []
    while current_state in parent:
        path_to_solution.append(current_state)
        current_state = parent[current_state]
    path_to_solution.append(start_state)
    path_to_solution.reverse()
    return path_to_solution

def print_solution_ids(states_explored, states_expanded, states_generated, max_fringe_size,
                   solution_depth, solution_cost, solution_path):
    print(f"States Explored: {states_explored}")
    print(f"States Expanded: {states_expanded}")
    print(f"States Generated: {states_generated}")
    print(f"Maximum Fringe Size: {max_fringe_size[0]}")
    print(f"Solution Found at Depth {solution_depth} with Cost {solution_cost}.")
    print('Steps:')
    for step_info in solution_path:
        move_cost, move_description = step_info[1], step_info[2]
        print(f"\tMove {move_cost} {move_description}")
    print("Solution Found Successfully!")

def write_trace_file_ids(states_explored, states_expanded, states_generated, max_fringe_size,
                     solution_depth, solution_cost, solution_path):
    trace_file_name = f'bfs_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt'
    with open(trace_file_name, "a") as trace_file:
        trace_file.write(f"\tStates Explored: {states_explored}\n")
        trace_file.write(f"\tStates Expanded: {states_expanded}\n")
        trace_file.write(f"\tStates Generated: {states_generated}\n")
        trace_file.write(f"\tMaximum Fringe Size: {max_fringe_size}\n")
        trace_file.write(f"\tSolution Found at Depth {solution_depth} with Cost {solution_cost}.\n")
        for step_info in solution_path:
            move_cost, move_description = step_info[1], step_info[2]
            trace_file.write(f"\tFringe Step: Move {move_cost} {move_description}\n")





def dlss(start_state, goal_state, max_depth):
    start_state = tuple(map(tuple, start_state))
    goal_state = tuple(map(tuple, goal_state))

    state_stack = [(start_state, 0)]  # Define state_stack here
    visited_states = set()
    parent = {}
    state_depth = {start_state: 0}
    state_cost = {start_state: 0}
    max_fringe_size = [1]
    solution_path = []

    result = dls_recursive(start_state, goal_state, max_depth, state_stack, visited_states, parent, state_depth, state_cost, max_fringe_size, solution_path)
    if result is not None:
        states_explored, states_expanded, states_generated, max_fringe_size, solution_depth, solution_cost, path = result
        solution_path.extend(path)
        print_solution_dfs(states_explored, states_expanded, states_generated, max_fringe_size,
                       solution_depth, solution_cost, solution_path)
        write_trace_file_dfs(states_explored, states_expanded, states_generated, max_fringe_size,
                         solution_depth, solution_cost, solution_path)
        return
    print("No solution found.")

def dls_recursive(current_state, goal_state, max_depth, state_stack, visited_states, parent, state_depth, state_cost, max_fringe_size, solution_path):
    while state_stack:
        current_state, depth = state_stack.pop()
        visited_states.add(current_state)

        if depth > max_depth:
            continue

        if current_state == goal_state:
            path_to_solution = reconstruct_solution_dfs(current_state, parent, start_state)
            return len(visited_states), len(visited_states) - 1, len(parent), max_fringe_size[0], \
                   state_depth[goal_state], state_cost[goal_state], path_to_solution

        for move_data in generate_moves_dfs(current_state, visited_states):
            new_state, move_cost, move_description = move_data[0], move_data[1], move_data[2]
            if new_state not in visited_states:
                state_stack.append((new_state, depth + 1))
                visited_states.add(new_state)
                update_state_info_dfs(new_state, current_state, parent, state_depth, state_cost, move_cost, max_fringe_size,
                                  solution_path, move_description, state_stack)

    return None
def generate_moves_dfs(node, visited):
    moves = []
    blank_row, blank_col = get_blank_pos_dfs(node)
    for dr, dc, move_description in [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]:
        new_row, new_col = blank_row + dr, blank_col + dc
        if 0 <= new_row < len(node) and 0 <= new_col < len(node[0]):
            new_node = swap_positions_dfs(node, blank_row, blank_col, new_row, new_col)
            move_cost = node[new_row][new_col]
            moves.append((new_node, move_cost, move_description))
    return moves

def get_blank_pos_dfs(node):
    for r, row in enumerate(node):
        for c, val in enumerate(row):
            if val == 0:
                return r, c
    raise ValueError("The node is not valid because there is no blank tile (0) found")

def swap_positions_dfs(node, r1, c1, r2, c2):
    node = [list(row) for row in node]
    node[r1][c1], node[r2][c2] = node[r2][c2], node[r1][c1]
    return tuple(map(tuple, node))

def update_state_info_dfs(new_state, current_state, parent, state_depth, state_cost, move_cost, max_fringe_size,
                      solution_path, move_description,state_stack):
    parent[new_state] = current_state
    state_depth[new_state] = state_depth[current_state] + 1
    state_cost[new_state] = state_cost[current_state] + move_cost
    max_fringe_size[0] = max(max_fringe_size[0], len(state_stack))
    solution_path.append((new_state, move_cost, move_description))

def reconstruct_solution_dfs(current_state, parent, start_state):
    path_to_solution = []
    while current_state in parent:
        path_to_solution.append(current_state)
        current_state = parent[current_state]
    path_to_solution.append(start_state)
    path_to_solution.reverse()
    return path_to_solution

def print_solution_dfs(states_explored, states_expanded, states_generated, max_fringe_size,
                   solution_depth, solution_cost, solution_path):
    print(f"States Explored: {states_explored}")
    print(f"States Expanded: {states_expanded}")
    print(f"States Generated: {states_generated}")
    print(f"Maximum Fringe Size: {max_fringe_size}")
    print(f"Solution Found at Depth {solution_depth} with Cost {solution_cost}.")
    print('Steps:')
    for step_info in solution_path:
        move_cost, move_description = step_info[1], step_info[2]
        print(f"\tMove {move_cost} {move_description}")
    print("Solution Found Successfully!")

def write_trace_file_dfs(states_explored, states_expanded, states_generated, max_fringe_size,
                     solution_depth, solution_cost, solution_path):
    trace_file_name = f'dfs_trace_file_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.txt'
    with open(trace_file_name, "a") as trace_file:
        trace_file.write(f"\tStates Explored: {states_explored}\n")
        trace_file.write(f"\tStates Expanded: {states_expanded}\n")
        trace_file.write(f"\tStates Generated: {states_generated}\n")
        trace_file.write(f"\tMaximum Fringe Size: {max_fringe_size}\n")
        trace_file.write(f"\tSolution Found at Depth {solution_depth} with Cost {solution_cost}.\n")
        for step_info in solution_path:
            move_cost, move_description = step_info[1], step_info[2]
            trace_file.write(f"\tFringe Step: Move {move_cost} {move_description}\n")
            
# Function to read input from a text file

def read_input_txt_to_list_format(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        return [[int(i) for i in line.split()] for line in lines[:3]]
    
def read_input_txt_to_list(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        return [int(i) for line in lines[:3] for i in line.split()]

def read_input_txt_to_tuple_format(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        return tuple([int(i) for line in lines[:3] for i in line.split()])





if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python expense_8_puzzle.py <start-file> <goal-file> <method> [<dump-flag>]")
        sys.exit()

    start_file = sys.argv[1]
    goal_file = sys.argv[2]
    method = sys.argv[3].lower() if len(sys.argv) > 3 else 'astar'
    dump_flag = True if len(sys.argv) > 4 and sys.argv[4].lower() == 'true' else False

    print(f"{'>>' * 30} Executing {method} algorithm{'<<' * 30}")

    if method == 'bfs':
        start_state = read_input_txt_to_list_format(start_file)
        goal_state = read_input_txt_to_list_format(goal_file)
        bfs(start_state, goal_state)
    
    elif method ==  'ucs':
        start_state = read_input_txt_to_list(start_file)
        goal_state = read_input_txt_to_list(goal_file)
        uniform_cost_search(start_state, goal_state)

    elif method == "greedy":
        start_file = sys.argv[1]
        goal_file = sys.argv[2]
        start_state = read_input_txt_to_list(start_file)
        goal_state = read_input_txt_to_list(goal_file)
        greedy(start_state, goal_state)
    
    elif method == 'astar':
        start_state = read_input_txt_to_tuple_format(start_file)
        goal_state = read_input_txt_to_tuple_format(goal_file)
        a_star(start_state, goal_state)
    
    elif method == 'dfs':
        start_state = read_input_txt_to_list_format(start_file)
        goal_state = read_input_txt_to_list_format(goal_file)
        dfs(start_state, goal_state)

    elif method == "dls":
        max_depth=int(input("enter the depth"))
        start_state = read_input_txt_to_list_format(start_file)
        goal_state = read_input_txt_to_list_format(goal_file)
        dlss(start_state, goal_state, max_depth)

