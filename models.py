import json
import pandas as pd
import numpy as np
from itertools import product
    
class Paths:
    SETTINGS = "configs/settings.json"
    SCENARIOS = "configs/scenarios.json"
    DUMMY = "data/dummy.json"
    TWINS = "data/twins.json"
    RESULTS = "results/results.json"
    
class Simulation:
    def __init__(self, Q: int, config: dict):
        self.Q = Q
        self.Table = pd.DataFrame()
        self.D = config['demand']
        self.return_levels = config['levels']
        self.c_U = config['c_U']
        self.c_O = config['c_O']
        self.scenario = config['name']

    def run(self):

        # Initialize the table
        self.Table["Gross Demand"] = self.D
        self.Table = pd.concat([self.Table, pd.DataFrame({'Gross Demand': [0] * len(self.return_levels)})], ignore_index=True)
        self.Table["Returns"] = 0.0 # to set column type to float for later returns
        self.Table["Week"] = range(len(self.Table))

        # Set the "Week" column as the index
        self.Table.set_index("Week", inplace=True)

        # Initialize lists to store starting and ending inventory
        starting_inventory = []
        ending_inventory = []
        sales = []
        lost_sales = []
        
        Q_temp = self.Q  # Starting inventory level
        
        # Calculate starting and ending inventory for each week
        for id, demand in enumerate(self.Table["Gross Demand"]):

            # write the starting inventory to the list
            starting_inventory.append(Q_temp)

            # assuming returns are available at the beginning of the week
            #Q_temp += self.Table["Returns"][id]
            Q_temp += self.Table.loc[id, "Returns"] # Week number is interpreted as label of the index

            # demand can be fully met from inventory
            if Q_temp >= demand:
                sales.append(demand)
                lost_sales.append(0)
                Q_temp -= demand
                ending_inventory.append(Q_temp)

            # demand cannot be fully met from inventory
            else:
                sales.append(Q_temp)
                lost_sales.append(demand - Q_temp)
                Q_temp = 0
                ending_inventory.append(0)

            # Add returns to inventory
            if sales[id] > 0: # otherwise runs out of index
                for i in range(len(self.return_levels)):
                    self.Table.loc[id+i+1, "Returns"] = np.random.binomial(n=sales[id], p=self.return_levels[i])

        self.Table["Starting Inventory"] = starting_inventory
        self.Table["Ending Inventory"] = ending_inventory
        self.Table["Sales"] = sales
        self.Table["Lost Sales"] = lost_sales
        self.Table["Net Demand"] = self.Table["Gross Demand"] - self.Table["Returns"]

        display(self.Table) # type: ignore

    def eval(self):
        # Calculate evaluation metrics
        self.Overage = self.Table['Ending Inventory'].iloc[-1]
        self.Underage = self.Table['Lost Sales'].sum()
        self.ASL = (1 - (len(self.Table[self.Table['Lost Sales'] > 0]) / len(self.Table))) * 100
        
        gross_demand = self.Table['Gross Demand'].sum()
        demand_served = self.Table['Sales'].sum()

        # Print results
        print("Evaluation Results: \n ")
        print(f"Order Quantity: {self.Q}")
        print(f"Overage: {self.Overage}")
        print(f"Underage: {self.Underage}")
        print(f"Underage Costs: {self.Underage * self.c_U:.2f}")
        print(f"Overage Costs: {self.Overage * self.c_O:.2f}")
        print(f"Gross Demand: {gross_demand}")
        # Check for gross demand being zero
        if gross_demand > 0:
            print(f"Demand Served [%]: {demand_served / gross_demand * 100:.2f} %")
        else:
            print("Demand Served [%]: None")

        # Handle cases where there are no lost sales
        first_week_lost_sales = (
            self.Table[self.Table['Lost Sales'] > 0].index[0]
            if not self.Table[self.Table['Lost Sales'] > 0].empty
            else "None"
        )

        print(f"First Week with Lost Sales: {first_week_lost_sales}")
        print(f"Weeks with Lost Sales: {len(self.Table[self.Table['Lost Sales'] > 0])}")
        print(f"Alpha Service Level [%]: {self.ASL:.2f} %")

def create_scenarios():
    print("Creating scenarios...")
    with open(Paths.SETTINGS, 'r') as file:
        config = json.load(file)

    returns_levels = config['returns_level']
    returns_delays = config['returns_delay']
    marketing_effects = config['marketing_effects']

    combinations = list(product(
        returns_levels.items(),
        returns_delays.items(),
        marketing_effects.items()
    ))

    scenarios = {}
    for combination in combinations:
        (level_key, level_value), (delay_key, delay_value), (effect_key, effect_value) = combination
        
        scenario_key = f"{level_key}_{delay_key}_{effect_key}"
        scenarios[scenario_key] = {
            "returns_level": level_value,
            "returns_delay": delay_value,
            "marketing_effects": effect_value
        }

    with open(Paths.SCENARIOS, 'w') as file:
        json.dump(scenarios, file, indent=4)
    print(f"Successfully created {len(scenarios)} scenarios and saved them to {Paths.SCENARIOS}.")

create_scenarios()

'''
Weniger Szenarien erlauben bessere Differenzierung der Szenarien. Wichtig für die Analyse der Ergebnisse.
Muss man noch verschiedene Nachfragelevel vergleichen?
Summe der demand patterns muss gleich sein für Vergleichbarkeit, noch testen !!!!!
Frage: Welche Szenarien solten wir rauslassen?
-> Extreme Szenarien begründen die Robuste Analyse am Besten
-> In Extremen Szenarian ist die Worst Caste Analyse aber vielleicht zu konservativ?
'''