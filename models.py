import json
import pandas as pd
from types import SimpleNamespace

def load_config(path):
    with open(path, 'r') as file:
        configs = json.load(file)
        print(f"Successfully loaded {len(configs)} scenarios.")
        return SimpleNamespace(**configs)
    
class Simulation:
    def __init__(self, Q: int, config: dict):
        self.Q = Q
        self.Table = pd.DataFrame()
        self.D = config['demand']
        self.return_levels = config['levels']
        self.c_U = config['c_U']
        self.c_O = config['c_O']

    def run(self):

        # Initialize the table
        self.Table["Gross Demand"] = self.D
        self.Table = pd.concat([self.Table, pd.DataFrame({'Gross Demand': [0] * len(self.return_levels)})], ignore_index=True)
        self.Table["Returns"] = 0.0 # to set column type to float for later returns

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
            Q_temp += self.Table["Returns"][id]

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
                    self.Table.loc[id+i+1, "Returns"] = self.return_levels[i] * sales[id]

        self.Table["Starting Inventory"] = starting_inventory
        self.Table["Ending Inventory"] = ending_inventory
        self.Table["Sales"] = sales
        self.Table["Lost Sales"] = lost_sales

        display(self.Table) # type: ignore

    def eval(self):
        self.Overage = self.Table['Ending Inventory'].iloc[-1]
        self.Underage = self.Table['Lost Sales'].sum()
        self.ASL = None

        print(f"Overage: {self.Overage}")
        print(f"Underage: {self.Underage}")
        print(f"ASL: {self.ASL}")
        print(f"Underage Costs{self.Underage * self.c_U}")
        print(f"Overage Costs{self.Overage * self.c_O}")
        print(f"Gross Demand: {self.Table['Gross Demand'].sum()}")
        print(f""