# Import the PuLP module for linear programming
import pulp
import math


def create_and_solve_model(max_items=20):
    """
    Creates and defines the linear optimization model.

    Returns:
        myModel (pulp.LpProblem): The formulated linear programming problem.
    """

    # Load data
    with open("Linear Programming/bin_packing_data_Schwerin1_BPP1.txt", "r") as file:
        # Read and parse the number of items
        num_items = int(file.readline().strip())
        
        # Read and parse the bin capacity
        bin_capacity = int(file.readline().strip())
        
        # Read the remaining lines as weights
        weights = [int(line.strip()) for line in file]

    # Verify the loaded data
    # print(f"Number of items: {num_items}")
    # print(f"Bin capacity: {bin_capacity}")
    # print(f"Weights: {weights}, len={len(weights)}")

    # Limit number of items (to enable solvability)
    weights = weights[:max_items]

    for idx, w in enumerate(weights):
        print(f"({idx},{w})")

    # Define parameters
    m = len(weights)    # Number of items
    I = [i for i in range(m)] # Set of items
    s = weights # Size of items
    
    B = bin_capacity    # Bin capacity
    n = math.ceil(sum(s)/B) # Number of available bin (lower bound)
    J = [j for j in range(n)]   # Set of bins

    # Define decision variables
    x = pulp.LpVariable.dicts(name="x", indices=(I,J), cat="Binary")    # If item i is put into bin j
    y = pulp.LpVariable.dicts(name="y", indices=(J), cat="Binary")      # If bin j is used

    # Initialize the problem for maximization
    myModel = pulp.LpProblem(name="myModel", sense=pulp.LpMinimize) # Minimize

    # Objective function
    myModel += pulp.lpSum(y[j] for j in J), "Total_Bins"

    # Constraints:
    # Capacity per bin
    for j in J:
        myModel += (pulp.lpSum(x[i][j]*s[i] for i in I) <= B*y[j], f"Constraint_Bin_{j}")

    # Assign every item to exactly one bin
    for i in I:
        myModel += (pulp.lpSum(x[i][j] for j in J) == 1, f"Constraint_Item_{i}")

    # Solve model
    solve_model(myModel, msg=False)
    
    # Print statistics
    for j in J:
        content = [(i, s[i]) for i in I if pulp.value(x[i][j])==1]
        print(f"Bin {j}: {content}")


def solve_model(myModel, msg=False, timeLimit=None, gapRel=None, gapAbs=None):
    """
    Solves the linear model using PuLP"s default solver (CBC solver).

    Args:
        myModel (pulp.LpProblem): The linear programming model to solve.
        msg (bool, optional): Whether to display solver output messages. Defaults to False.
        timeLimit (float or int, optional): Time limit for the solver in seconds. Defaults to None.
        gapRel (float, optional): Relative gap tolerance. Defaults to None.
        gapAbs (float, optional): Absolute gap tolerance. Defaults to None.

    """
    # Create a solver instance with optional parameters
    solver = pulp.PULP_CBC_CMD(
        msg=msg,                
        timeLimit=timeLimit,    
        gapRel=gapRel,          
        gapAbs=gapAbs           
    )

    # Solve the problem using the specified solver
    myModel.solve(solver)


if __name__ == "__main__":
    # Create and solve model
    create_and_solve_model(max_items=40)
