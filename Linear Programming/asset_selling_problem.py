# Import the PuLP module for linear programming
import pulp
import pandas as pd


def define_model():
    """
    Creates and returns a linear programming model for the asset selling problem.
    
    The model loads data from an Excel file, defines decision variables for
    selling on each date (binary variables), sets the objective to maximize
    profit based on the sale date, and includes a constraint that only one sale
    can occur.

    Returns:
        pulp.LpProblem: A PuLP linear programming problem object representing the model.
    """
    
    # Load the Excel file containing the data
    df = pd.read_excel('Linear Programming/asset_selling_problem_data_K+N.xlsx')

    # Convert the 'Date' column to datetime objects, then to date (without time)
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    # Create a list of all dates in the dataset
    dates = df['Date'].tolist()

    # Extract the 'Open' prices into a list
    price_list = df['Open'].tolist()
    
    # Create a dictionary mapping each date to its corresponding price
    price = {t: price_list[idx] for idx, t in enumerate(dates)}    

    # Define decision variables: binary variables indicating whether to sell on each date
    x = pulp.LpVariable.dict(
        name="x",
        indices=dates,
        cat="Binary"
    )

    # Initialize a maximization problem
    myModel = pulp.LpProblem(name='Asset_Selling_Problem', sense=pulp.LpMaximize)

    # Set the objective function: maximize total profit (selling price based on chosen date)
    myModel += pulp.lpSum(x[t] * price[t] for t in dates), 'Total_Profit'

    # Add a constraint: sell only once (exactly one date can be chosen)
    myModel += pulp.lpSum(x[t] for t in dates) == 1, 'Only_Sell_Once'
    
    return myModel


def print_statistics(myModel):
    """
    Prints the status, objective value, and variables' values.

    Args:
        myModel (pulp.LpProblem): The solved linear model.
    """
    print('Solution:')

    # Display each decision variable's value
    for v in myModel.variables():
        if v.varValue > 0:
            print(f' {v.name} = {v.varValue}')

    # Display solution status
    print('-- Status:', pulp.LpStatus[myModel.status])

    # Display the objective function's value
    print('-- Objective Value:', pulp.value(myModel.objective))


if __name__ == "__main__":
    myModel = define_model()
    myModel.solve()
    print_statistics(myModel)
