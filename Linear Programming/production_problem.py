# Import the PuLP module for linear programming
import pulp


def create_model_explicit():
    """
    Creates and defines the linear optimization model.

    Returns:
        myModel (pulp.LpProblem): The formulated linear programming problem.
    """

    # Define decision variables:
    # x1 and x2 represent the quantities of products 1 and 2 to produce
    # lowerBound=0 indicates production quantities cannot be negative
    x1 = pulp.LpVariable(name='x1', lowBound=0)
    x2 = pulp.LpVariable(name='x2', lowBound=0)

    # Initialize the problem for maximization
    myModel = pulp.LpProblem(name='Production_Problem_Explicit', sense=pulp.LpMaximize)

    # Objective function: Maximize profit
    # Profit contributions from each product
    myModel += 3 * x1 + 5 * x2, 'Total_Profit'

    # Constraints:
    # Machine 1 capacity constraint
    myModel += x1 <= 4, 'Machine_1_Capacity'

    # Machine 2 capacity constraint
    myModel += 2 * x2 <= 12, 'Machine_2_Capacity'

    # Machine 3 capacity constraint
    myModel += 3 * x1 + 2 * x2 <= 18, 'Machine_3_Capacity'

    return myModel


def create_model_generic():
    """
    Creates and defines a generic version of the original production problem using dictionaries.

    Returns:
        myModel (pulp.LpProblem): The formulated linear programming problem.
    """

    # Define sets
    products = ['Product 1', 'Product 2']
    machines = ['Machine 1', 'Machine 2', 'Machine 3']

    # Define parameters
    profit_per_product = {'Product 1': 3, 'Product 2': 5}
    capacity_per_machine = {'Machine 1': 4, 'Machine 2': 12, 'Machine 3': 18}
    process_time = {'Product 1': {'Machine 1': 1, 'Machine 2': 0, 'Machine 3': 3},
                    'Product 2': {'Machine 1': 0, 'Machine 2': 2, 'Machine 3': 2},}

    # Define decision variables
    x = pulp.LpVariable.dict(name='x', indices=(products), lowBound=0)

    # Initialize the problem for maximization
    myModel = pulp.LpProblem(name='Production_Problem_Generic', sense=pulp.LpMaximize)

    # Objective function: Maximize profit
    myModel += pulp.lpSum(profit_per_product[i] * x[i] for i in products), 'Total_Profit'

    # Constraints:
    for j in machines:
        myModel += pulp.lpSum(x[i]*process_time[i][j] for i in products) <= capacity_per_machine[j], 'Machine_' + str(j) + '_Capacity'
    
    # # Solve
    # myModel.solve()

    # # Print statistics
    # print('Status:', pulp.LpStatus[myModel.status])
    # print('Objective Value:', pulp.value(myModel.objective))
    # for i in products:
    #     print(f" {i} = {pulp.value(x[i])}")
    # print()

    return myModel

def solve_model_with_default_solver(myModel):
    """
    Solves the linear model using PuLP's default solver (CBC solver).

    Args:
        myModel (pulp.LpProblem): The linear programming model to solve.
    """
    # Create a solver instance with optional parameters
    solver = pulp.PULP_CBC_CMD(
        msg=False,        # msg=False suppresses solver output messages
        timeLimit=5,      # Limit the solver's runtime to 5 seconds
        gapRel=0.001,     # Relative optimality gap
        gapAbs=10         # Absolute optimality gap
    )

    # Solve the problem using the specified solver
    myModel.solve(solver)

    # Note: When calling myModel.solve() without specifying a solver, the default solver 
    # (PuLP's default CBC solver) is used with its default settings, such as the default 
    # time limit and other parameters.


def solve_model_with_cplex(myModel):
    """
    Solves the linear model using CPLEX solver via PuLP's CPLEX_PY interface.

    Args:
        myModel (pulp.LpProblem): The linear programming model to solve.
    """
    # Create a CPLEX solver instance with parameters
    solver = pulp.CPLEX_PY(
        msg=False,          # # msg=False suppresses solver output messages
        timeLimit=5,        # Limit runtime to 5 seconds
        gapRel=0.001        # Relative optimality gap
        # Note: gapAbs parameter is not available for CPLEX_PY
    )

    # Solve the problem
    myModel.solve(solver)


def print_statistics(myModel):
    """
    Prints the status, objective value, and variables' values.

    Args:
        myModel (pulp.LpProblem): The solved linear model.
    """

    # Display solution status
    print('Status:', pulp.LpStatus[myModel.status])

    # Display the objective function's value
    print('Objective Value:', pulp.value(myModel.objective))

    # Display each decision variable's value
    for v in myModel.variables():
        print(f'  {v.name} = {v.varValue}')


if __name__ == "__main__":
    # Solve the model using the default solver
    # print('Solve via default solver')
    # model = create_model_explicit()
    # solve_model_with_default_solver(model)
    # print_statistics(model)

    print()

    # Solve the model using CPLEX solver
    print('Solve via CPLEX')
    model = create_model_explicit()
    model = create_model_generic()
    solve_model_with_cplex(model)
    print_statistics(model)