# Import the PuLP module for linear programming
import pulp

def get_data_us():

    # Supply nodes
    S = ['Kansas City', 'Omaha', 'Davenport']

    # Supply quantity
    s = {'Kansas City': 150, 'Omaha': 175, 'Davenport': 275}

    # Demand nodes
    D = ['Chicago', 'St. Louis', 'Cincinnati']

    # Demand quantity 
    d = {'Chicago': 200, 'St. Louis': 100, 'Cincinnati': 300}

    # Shippping costs 
    c = {
        'Kansas City': {'Chicago': 6, 'St. Louis': 8, 'Cincinnati': 10},
        'Omaha': {'Chicago': 7, 'St. Louis': 11, 'Cincinnati': 11},
        'Davenport': {'Chicago': 4, 'St. Louis': 5, 'Cincinnati': 12},
    }

    return S, s, D, d, c


def get_data_airfreight_transportation():
    raw_data = [
        {
            "City": "Shanghai",
            "Country": "China",
            "Longitude": 121.5,
            "Latitude": 31.4,
            "Demand": 428,
            "Supply": None
        },
        {
            "City": "Singapore",
            "Country": "Singapore",
            "Longitude": 103.8,
            "Latitude": 1.3,
            "Demand": 122,
            "Supply": None
        },
        {
            "City": "Ningbo-Zhoushan",
            "Country": "China",
            "Longitude": 121.8,
            "Latitude": 29.9,
            "Demand": 335,
            "Supply": None
        },
        {
            "City": "Shenzhen (Yantian)",
            "Country": "China",
            "Longitude": 114.2,
            "Latitude": 22.6,
            "Demand": 183,
            "Supply": None
        },
        {
            "City": "Guangzhou (Nansha)",
            "Country": "China",
            "Longitude": 113.4,
            "Latitude": 22.7,
            "Demand": 378,
            "Supply": None
        },
        {
            "City": "Hong Kong",
            "Country": "Hong Kong SAR",
            "Longitude": 114.2,
            "Latitude": 22.3,
            "Demand": 189,
            "Supply": None
        },
        {
            "City": "Tianjin",
            "Country": "China",
            "Longitude": 117.8,
            "Latitude": 39.2,
            "Demand": 482,
            "Supply": None
        },
        {
            "City": "Busan",
            "Country": "South Korea",
            "Longitude": 129,
            "Latitude": 35.1,
            "Demand": 204,
            "Supply": None
        },
        {
            "City": "Qingdao",
            "Country": "China",
            "Longitude": 120.3,
            "Latitude": 36.1,
            "Demand": 218,
            "Supply": None
        },
        {
            "City": "Dubai (Jebel Ali)",
            "Country": "UAE",
            "Longitude": 55.2,
            "Latitude": 25,
            "Demand": 292,
            "Supply": None
        },
        {
            "City": "Rotterdam",
            "Country": "Netherlands",
            "Longitude": 4.5,
            "Latitude": 51.9,
            "Demand": 200,
            "Supply": None
        },
        {
            "City": "Antwerp",
            "Country": "Belgium",
            "Longitude": 4.4,
            "Latitude": 51.2,
            "Demand": 345,
            "Supply": None
        },
        {
            "City": "Laem Chabang",
            "Country": "Thailand",
            "Longitude": 100.4,
            "Latitude": 13.1,
            "Demand": 483,
            "Supply": None
        },
        {
            "City": "Los Angeles",
            "Country": "USA",
            "Longitude": -118.2,
            "Latitude": 33.7,
            "Demand": 95,
            "Supply": None
        },
        {
            "City": "Long Beach",
            "Country": "USA",
            "Longitude": -118.2,
            "Latitude": 33.8,
            "Demand": 152,
            "Supply": None
        },
        {
            "City": "Hamburg",
            "Country": "Germany",
            "Longitude": 9.9,
            "Latitude": 53.5,
            "Demand": 373,
            "Supply": None
        },
        {
            "City": "Kobe",
            "Country": "Japan",
            "Longitude": 135.2,
            "Latitude": 34.7,
            "Demand": 428,
            "Supply": None
        },
        {
            "City": "Barcelona",
            "Country": "Spain",
            "Longitude": 2.2,
            "Latitude": 41.3,
            "Demand": 455,
            "Supply": None
        },
        {
            "City": "Santos",
            "Country": "Brazil",
            "Longitude": -46.3,
            "Latitude": -23.9,
            "Demand": 321,
            "Supply": None
        },
        {
            "City": "Port Klang (Klang)",
            "Country": "Malaysia",
            "Longitude": 101.4,
            "Latitude": 3,
            "Demand": None,
            "Supply": 802
        },
        {
            "City": "Dalian",
            "Country": "China",
            "Longitude": 121.4,
            "Latitude": 38.9,
            "Demand": None,
            "Supply": 744
        },
        {
            "City": "Xiamen",
            "Country": "China",
            "Longitude": 118.1,
            "Latitude": 24.5,
            "Demand": None,
            "Supply": 420
        },
        {
            "City": "Kaohsiung",
            "Country": "Taiwan",
            "Longitude": 120.3,
            "Latitude": 22.6,
            "Demand": None,
            "Supply": 902
        },
        {
            "City": "Tanjung Priok",
            "Country": "Indonesia",
            "Longitude": 106.9,
            "Latitude": -6.2,
            "Demand": None,
            "Supply": 162
        },
        {
            "City": "Constanța",
            "Country": "Romania",
            "Longitude": 28.6,
            "Latitude": 44.2,
            "Demand": None,
            "Supply": 398
        },
        {
            "City": "Gdansk",
            "Country": "Poland",
            "Longitude": 18.7,
            "Latitude": 54.4,
            "Demand": None,
            "Supply": 272
        },
        {
            "City": "Vladivostok",
            "Country": "Russia",
            "Longitude": 131.9,
            "Latitude": 43.1,
            "Demand": None,
            "Supply": 612
        },
        {
            "City": "Dar es Salaam",
            "Country": "Tanzania",
            "Longitude": 39.2,
            "Latitude": -6.8,
            "Demand": None,
            "Supply": 918
        },
        {
            "City": "Colombo",
            "Country": "Sri Lanka",
            "Longitude": 79.9,
            "Latitude": 6.9,
            "Demand": None,
            "Supply": 282
        },
        {
            "City": "Gwangyang (South Korea)",
            "Country": "South Korea",
            "Longitude": 127.7,
            "Latitude": 34.9,
            "Demand": None,
            "Supply": 171
        }
    ]

    # Supply nodes
    S = []

    # Supply quantity
    s = {}

    # Demand nodes
    D = []

    # Demand quantity 
    d = {}

    # Lon/Lat data
    city_location = {}

    # Process raw data
    for entry in raw_data:
        if entry['Supply'] is not None:
            S.append(entry['City'])
            s[entry['City']] = entry['Supply']
        else:
            D.append(entry['City'])
            d[entry['City']] = entry['Demand']
        
        city_location[entry['City']] = {'lon': entry['Longitude'], 
                                        'lat': entry['Latitude']}

    # Shippping costs
    import haversine
    c = {
        i: {j: round(haversine.haversine(point1=(city_location[i]['lat'], city_location[i]['lon']),
                                         point2=(city_location[j]['lat'], city_location[j]['lon'])
                                         ), 2) 
                                         for j in D}
        for i in S
    } 
    
    return S, s, D, d, c


def define_model_generic(S, s, D, d, c):
    """
    Creates and returns a generic linear programming model for the transportation problem.

    Returns:
        pulp.LpProblem: A PuLP linear programming problem object representing the model.
    """

    # Define decision variables
    # x_origin_destination represent the quantities of shipped between the supply and demand location
    x = pulp.LpVariable.dicts(name='x', indices=(S, D), lowBound=0)

    # Initialize the problem for minimization
    myModel = pulp.LpProblem(name='Transportation_Problem_Generic', sense=pulp.LpMinimize)

    # Objective function: Minimize costs
    myModel += pulp.lpSum(c[i][j] * x[i][j] for i in S for j in D), 'Total_Costs'

    # Constraints:
    # Supply capacity
    for i in S:
        myModel += s[i] == pulp.lpSum(x[i][j] for j in D), 'Supply_Capacity_' + str(i)

    # Demand quantity
    for j in D:
        myModel += d[j] == pulp.lpSum(x[i][j] for i in S), 'Demand_Quantity_' + str(j)
    
    return myModel


def define_model_explicit_us():
    """
    Creates and returns an explicit linear programming model for the transportation problem.
    
    Returns:
        pulp.LpProblem: A PuLP linear programming problem object representing the model.
    """

    # Define decision variables:
    # x_origin_destination represent the quantities of shipped between the two locations
    # lowerBound=0 indicates production quantities cannot be negative
    x_kansas_city_to_chicago = pulp.LpVariable(name='x_kansas_city_to_chicago', lowBound=0)
    x_kansas_city_to_st_louis = pulp.LpVariable(name='x_kansas_city_to_st_louis', lowBound=0)
    x_kansas_city_to_cincinnati = pulp.LpVariable(name='x_kansas_city_to_cincinnati', lowBound=0)

    x_omaha_to_chicago = pulp.LpVariable(name='x_omaha_to_chicago', lowBound=0)
    x_omaha_to_st_louis = pulp.LpVariable(name='x_omaha_to_st_louis', lowBound=0)
    x_omaha_to_cincinnati = pulp.LpVariable(name='x_omaha_to_cincinnati', lowBound=0)

    x_davenport_to_chicago = pulp.LpVariable(name='x_davenport_to_chicago', lowBound=0)
    x_davenport_to_st_louis = pulp.LpVariable(name='x_davenport_to_st_louis', lowBound=0)
    x_davenport_to_cincinnati = pulp.LpVariable(name='x_davenport_to_cincinnati', lowBound=0)

    # Initialize the problem for minimization
    myModel = pulp.LpProblem(name='Transportation_Problem_Explicit', sense=pulp.LpMinimize)

    # Objective function: Maximize profit
    # Profit contributions from each product
    myModel += 6*x_kansas_city_to_chicago + 8*x_kansas_city_to_st_louis + 10*x_kansas_city_to_cincinnati \
            + 7*x_omaha_to_chicago + 11*x_omaha_to_st_louis + 11*x_omaha_to_cincinnati \
            + 4*x_davenport_to_chicago + 5*x_davenport_to_st_louis + 12*x_davenport_to_cincinnati \
                , 'Total_Costs'

    # Constraints
    # Supply Capacity
    myModel += x_kansas_city_to_chicago + x_kansas_city_to_st_louis + x_kansas_city_to_cincinnati == 150, 'Supply_Capacity_Kansas_City'
    myModel += x_omaha_to_chicago + x_omaha_to_st_louis + x_omaha_to_cincinnati == 175, 'Supply_Capacity_Chicago'
    myModel += x_davenport_to_chicago + x_davenport_to_st_louis + x_davenport_to_cincinnati == 275, 'Supply_Capacity_Davenport'

    # Demand Quantity
    myModel += x_kansas_city_to_chicago + x_omaha_to_chicago + x_davenport_to_chicago == 200, 'Demand_Quantity_Chicago'
    myModel += x_kansas_city_to_st_louis + x_omaha_to_st_louis + x_davenport_to_st_louis == 100, 'Demand_Quantity_St_Lois'
    myModel += x_kansas_city_to_cincinnati + x_omaha_to_cincinnati + x_davenport_to_cincinnati == 300, 'Demand_Quantity_Cincinnati'

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
        print(f' {v.name} = {v.varValue}')

    # Display solution status
    print('-- Status:', pulp.LpStatus[myModel.status])

    # Display the objective function's value
    print('-- Objective Value:', pulp.value(myModel.objective))


if __name__ == "__main__":
    # Explicit version
    # myModel = define_model_explicit_us()
    # myModel.solve()
    # print_statistics(myModel)

    # Generic version 
    S, s, D, d, c = get_data_us()           # US data
    # S, s, D, d, c = get_data_airfreight_transportation() # Airfreight Transportation Data

    myModel = define_model_generic(S, s, D, d, c)
    myModel.solve()
    print_statistics(myModel)