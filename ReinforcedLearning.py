# Define the parameters given in the problem
gamma = 0.9           # Discount factor
x, y = 0.25, 0.25     # Transition probabilities
epsilon = 0.0001      # Termination threshold

# Define the reward function R(s, a) for each state and action
# Assuming rewards are given as per the task description
rewards = {
    'S0': {'a0': 0, 'a1': 0, 'a2': 0},
    'S1': {'a0': 0, 'a1': 0, 'a2': 0},
    'S2': {'a0': 1, 'a1': 0, 'a2': 0},
    'S3': {'a0': 10, 'a1': 0, 'a2': 0}
}

# Define the transition probabilities T(s, a, s')
transitions = {
    'S0': {'a0': [0, 0, 0, 0], 'a1': [0, 1, 0, 0], 'a2': [0, 0, 1, 0]},
    'S1': {'a0': [0, 1 - x, 0, x], 'a1': [0, 0, 0, 0], 'a2': [0, 0, 0, 0]},
    'S2': {'a0': [1 - y, 0, 0, y], 'a1': [0, 0, 0, 0], 'a2': [0, 0, 0, 0]},
    'S3': {'a0': [1, 0, 0, 0], 'a1': [0, 0, 0, 0], 'a2': [0, 0, 0, 0]}
}

# Initialize V(s) for all states
V = {'S0': 0, 'S1': 0, 'S2': 0, 'S3': 0}

# Value iteration algorithm
def value_iteration():
    global V
    iteration = 0
    while True:
        # Store the old values of V to check convergence
        V_old = V.copy()
        max_delta = 0  # Track the maximum change for termination
        
        # Loop through each state to update V(s)
        for s in V:
            q_values = []  # Store Q(s, a) values for each action
            
            # Loop through each action available in state s
            for a in rewards[s]:
                # Compute Q(s, a) using the formula
                q_value = rewards[s][a]  # Start with immediate reward R(s, a)
                
                # Add gamma * sum over all s' of P(s' | s, a) * V(s')
                for next_state, prob in zip(V, transitions[s][a]):
                    q_value += gamma * prob * V_old[next_state]
                
                q_values.append(q_value)  # Store Q(s, a)
            
            # Update V(s) to max Q(s, a) for the best action a
            V[s] = max(q_values)
            
            # Check the change for convergence
            max_delta = max(max_delta, abs(V[s] - V_old[s]))
        
        iteration += 1  # Track the iteration number
        
        # Check if we have reached the termination condition
        if max_delta < epsilon:
            print(f"Converged after {iteration} iterations.")
            break

# Run value iteration
value_iteration()

# Display the optimal value function V* and policy pi*
print("Optimal Values (V*):", V)