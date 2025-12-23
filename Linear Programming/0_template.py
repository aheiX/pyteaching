# Import the PuLP module for linear programming
import pulp


def create_model():
    """
    Creates and defines the linear optimization model.

    Returns:
        myModel (pulp.LpProblem): The formulated linear programming problem.
    """
    # Load and define data
    # Example: A = [1, 2, 3]

    # Define decision variables:
    # Example 1: x1 = pulp.LpVariable(name="x1", lowBound=0)
    # Example 2: x = pulp.LpVariable.dicts(name="x", indices=(A,B,C), cat="Binary") # With A, B, and C being sets

    # Initialize the problem for maximization
    myModel = pulp.LpProblem(name="myModel", sense=pulp.LpMaximize)   # Maximize
    # myModel = pulp.LpProblem(name="myModel", sense=pulp.LpMinimize) # Minimize

    # Objective function
    # Example: myModel += 3 * x1 + 5 * x2, "Total_Profit"

    # Constraints:
    # Example: myModel += x1 <= 4, "Constraint_1"
    # for i in I:
    #     myModel += (pulp.lpSum(x[i][j] for j in J) <= 1, f"Constraint_{i}")

    return myModel


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

    # Note: When calling myModel.solve() without specifying a solver, the default solver 
    # (PuLP"s default CBC solver) is used with its default settings, such as the default 
    # time limit and other parameters.


def print_statistics(myModel):
    """
    Prints the status, objective value, and variables" values.

    Args:
        myModel (pulp.LpProblem): The solved linear model.
    """

    # Display solution status
    print("Status:", pulp.LpStatus[myModel.status])

    # Display the objective function"s value
    print("Objective Value:", pulp.value(myModel.objective))

    # Display each decision variable"s value
    for v in myModel.variables():
        print(f"  {v.name} = {v.varValue}")


if __name__ == "__main__":
    # Create model
    myModel = create_model()

    # Solve (via default)
    solve_model(myModel)
    
    # Print statistics
    print_statistics(myModel)