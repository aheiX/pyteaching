import pulp

def create_model(
    scenario_1=False,
    scenario_2=False,
    scenario_3=False,
    scenario_4=False,
    scenario_5=False,
    scenario_6=False,
    scenario_7=False
):
    """
    Creates a PuLP LP model for scheduling Smurfs considering multiple scenarios.

    Args:
        scenario_1 (bool): If True, apply Rest Policy constraints.
        scenario_2 (bool): If True, apply Shift Loyalty constraints.
        scenario_3 (bool): If True, apply Grouchy-Clumsy divide constraints.
        scenario_4 (bool): If True, apply Buddy System constraints.
        scenario_5 (bool): If True, apply Tuesday Assembly constraints.
        scenario_6 (bool): If True, restrict PapaSmurf from Friday.
        scenario_7 (bool): If True, apply Brainy Smurf's academic workload constraints.

    Returns:
        pulp.LpProblem: The constructed scheduling LP problem.
    """

    # Define days, shifts, and smurfs involved in scheduling
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    shifts = ["Morning", "Evening"]
    smurfs = ["PapaSmurf", "Smurfette", "BrainySmurf", "GrouchySmurf", "ClumsySmurf"]
    
    # Min and Max number of smurfs per shift each day
    min_max_shift = {
        "Monday": {
            "Morning": {"Min": 2, "Max": 3},
            "Evening": {"Min": 1, "Max": 2}
        },
        "Tuesday": {
            "Morning": {"Min": 1, "Max": 2},
            "Evening": {"Min": 2, "Max": 3}
        },
        "Wednesday": {
            "Morning": {"Min": 1, "Max": 32},
            "Evening": {"Min": 2, "Max": 3}
        },
        "Thursday": {
            "Morning": {"Min": 1, "Max": 3},
            "Evening": {"Min": 3, "Max": 4}
        },
        "Friday": {
            "Morning": {"Min": 1, "Max": 3},
            "Evening": {"Min": 1, "Max": 2}
        }
    }

    # Initialize decision variables: x[day][shift][smurf], binary (assigned or not)
    x = pulp.LpVariable.dicts(
        name="x",
        indices=(days, shifts, smurfs),
        lowBound=0,
        cat="Binary"
    )

    # Create the LP problem instance
    myModel = pulp.LpProblem(
        name="Scheduling_Smurfs",
        sense=pulp.LpMinimize
    )

    # Objective: Minimize total number of scheduled smurfs (can be modified)
    myModel += (
        pulp.lpSum(x[day][shift][smurf] for day in days for shift in shifts for smurf in smurfs),
        "Total_Smurfs"
    )


    # Constraint: Each smurf works exactly 4 shifts per week, unless scenario 7 modifies this
    for smurf in smurfs:
        # Adjust maximum shifts for BrainySmurf if scenario 7 is active
        max_shifts = 3 if (scenario_7 and smurf == "BrainySmurf") else 4
        myModel += (
            pulp.lpSum(x[day][shift][smurf] for day in days for shift in shifts) == max_shifts,
            f"Weekly_Shifts_{smurf}"
        )

    # Constraints: Limits on number of smurfs per shift each day
    for day in days:
        for shift in shifts:
            myModel += (
                pulp.lpSum(x[day][shift][smurf] for smurf in smurfs) >= min_max_shift[day][shift]["Min"],
                f"Smurfs_{day}_{shift}_Min"
            )
            myModel += (
                pulp.lpSum(x[day][shift][smurf] for smurf in smurfs) <= min_max_shift[day][shift]["Max"],
                f"Smurfs_{day}_{shift}_Max"
            )

    # Constraint: No smurf can work more than one shift per day
    for day in days:
        for smurf in smurfs:
            myModel += (
                x[day]["Morning"][smurf] + x[day]["Evening"][smurf] <= 1,
                f"Daily_Shifts_{day}_{smurf}"
            )
    
    # Scenario 1: Smurf Rest Policy
    # No Smurf can work the morning shift if they worked the evening shift the day before (no back-to-back shifts).
    if scenario_1:
        for smurf in smurfs:
            myModel += x["Monday"]["Evening"][smurf] + x["Tuesday"]["Morning"][smurf] <= 1, f"Smurf_Rest_Monday_{smurf}"
            myModel += x["Tuesday"]["Evening"][smurf] + x["Wednesday"]["Morning"][smurf] <= 1, f"Smurf_Rest_Tuesday_{smurf}"
            myModel += x["Wednesday"]["Evening"][smurf] + x["Thursday"]["Morning"][smurf] <= 1, f"Smurf_Rest_Wednesday_{smurf}"
            myModel += x["Thursday"]["Evening"][smurf] + x["Friday"]["Morning"][smurf] <= 1, f"Smurf_Rest_Thursday_{smurf}"

    # Scenario 2: Smurf Shift Loyalty
    # Every Smurf wants to work either the morning shift or the evening shift for the entire week (for consistency).
    if scenario_2:
        # Binary variables for loyalty
        LoyalMorning = pulp.LpVariable.dict("LoyalMorning", smurfs, cat="Binary")
        LoyalEvening = pulp.LpVariable.dict("LoyalEvening", smurfs, cat="Binary")

        for smurf in smurfs:
            # Each Smurf is either loyal to morning or evening
            myModel += LoyalMorning[smurf] + LoyalEvening[smurf] == 1, f"ShiftLoyalty_{smurf}"

            # Consider deviating number of max. shifts in scenario 7
            max_shifts = 3 if scenario_7 and smurf == "BrainySmurf" else 4 

            # Enforce loyalty in shifts:
            # If loyal to morning, total morning shifts must be 3 or 4 (as defined via max_shifts), evening 0
            myModel += pulp.lpSum(x[day]["Morning"][smurf] for day in days) >= max_shifts * LoyalMorning[smurf], f"MorningLoyal_{smurf}"
            myModel += pulp.lpSum(x[day]["Morning"][smurf] for day in days) <= max_shifts * LoyalMorning[smurf], f"MorningLoyal_upper_{smurf}"

            myModel += pulp.lpSum(x[day]["Evening"][smurf] for day in days) >= max_shifts * LoyalEvening[smurf], f"EveningLoyal_{smurf}"
            myModel += pulp.lpSum(x[day]["Evening"][smurf] for day in days) <= max_shifts * LoyalEvening[smurf], f"EveningLoyal_upper_{smurf}"

    # Scenario 3: The Grouchy-Clumsy Divide
    # Grouchy Smurf demands a Clumsy-free shift zone for his own peace of mind. He says: ‘If Clumsy’s on the shift, I’m out!’
    if scenario_3:
        for day in days:
            for shift in shifts:
                myModel += x[day][shift]["GrouchySmurf"] + x[day][shift]["ClumsySmurf"] <= 1, f"Grouchy_Clumsy_{day}_{shift}"
    
    # Scenario 4: The Smurf Buddy System
    # Papa Smurf is determined to support Clumsy Smurf, so he requests at least three shifts together to keep things… less clumsy.
    if scenario_4:
        # Define binary variables indicating if PapaSmurf and ClumsySmurf work together on each shift in each day
        # BuddyShift[day][shift] == 1 if both Smurfs work the same shift on that day
        BuddyShift = pulp.LpVariable.dicts("BuddyShift", (days, shifts), cat="Binary")

        # Loop over each day and shift to enforce that BuddyShift[day][shift] == 1
        # if and only if both PapaSmurf and ClumsySmurf are scheduled to work that shift on that day
        for day in days:
            for shift in shifts:
                # The sum of their assignment variables must be at least 2 (both working),
                # scaled by the binary BuddyShift variable
                myModel += x[day][shift]["PapaSmurf"] + x[day][shift]["ClumsySmurf"] >= 2 * BuddyShift[day][shift], f"Buddy_{day}_{shift}"

        # Enforce that collectively, PapaSmurf and ClumsySmurf work together for at least 3 shifts during the week
        myModel += pulp.lpSum(BuddyShift[day][shift] for day in days for shift in shifts) >= 3, "Smurf_Buddy_System"

    # Scenario 5: The Great Tuesday Assembly
    # Tuesday is meeting day! All Smurfs need to be scheduled for either the morning or evening shift so they can attend the team gathering in between.
    if scenario_5:
        for k in smurfs:
            myModel += x["Tuesday"]["Morning"][k] + x["Tuesday"]["Evening"][k] == 1, f"Tues_Attendance_{k}"
        
    # Scenario 6: No Friday for Papa Smurf
    # Papa Smurf prefers not to work on Fridays to prepare for the weekend village meeting. He will still work four shifts. 
    if scenario_6:
        myModel += x["Friday"]["Morning"]["PapaSmurf"] + x["Friday"]["Evening"]["PapaSmurf"] == 0, "Papa_No_Friday_Shift"

    # Scenario 7: Academic Escape Clause
    # Brainy Smurf exploited a loophole and now only works three shifts – Monday, Tuesday, and Wednesday. He claims the rest of the week is reserved for “academic research.”
    if scenario_7:
        # Please also notice the adjustment in regular constraint "Each Smurf works four shifts a week" and constraint for scenario 2 (morning/evening consistency)
        myModel += pulp.lpSum(x[day][shift]["BrainySmurf"] for day in ["Thursday", "Friday"] for shift in shifts) == 0, "Academic_Escape_Clause"

    # Scenario 8: The Smurf Perfection Pact
    # Smurfs expect nothing less than perfection: ensure all constraints are fulfilled at once.
    # --> Activate scenarios one to seven 

    return myModel

def print_statistics(model):
    """
    Prints basic statistics and schedule plan from the LP model.
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    shifts = ["Morning", "Evening"]
    for day in days:
        print(day)
        for shift in shifts:
            assigned_smurfs = [
                v.name.split("_")[3]
                for v in model.variables()
                if day in v.name and shift in v.name and v.varValue > 0 and v.name.startswith("x")
            ]
            print(f"  {shift}: {assigned_smurfs}")

    # Display model status and objective value
    print("-- Status:", pulp.LpStatus[model.status])
    print("-- Objective Value:", pulp.value(model.objective))


if __name__ == "__main__":
    # Build model with all scenarios enabled
    myModel = create_model()

    # Solve the LP problem
    myModel.solve()
    
    # Print the scheduling result statistics
    print_statistics(myModel)