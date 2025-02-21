import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from google.cloud import bigquery as bq
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

class InputData:
    project_id = "brain-flash-dev"
    dataset_id = "dagster_common"
    data_path = "data"
    TestData = {}  # Store TestData as a class attribute
    TwinData = {}  # Store TwinData as a class attribute

    def __init__(self, file_name: str = "twins_100"):
        """Loads data from a CSV file and initializes unique communication keys."""
        self.data = pd.read_csv(f"{self.data_path}/{file_name}.csv")

        self.TestData = {key : self.get_test_item(key) for key in self.data["TEST_ITEM_COMMUNICATIONKEY"].unique()}
        self.TwinData = {key : self.get_twin_item(key, 10) for key in self.data["TEST_ITEM_COMMUNICATIONKEY"].unique()}

    @classmethod
    def load_data(cls, file_name: str = "twins_100"):
        """Loads data from a CSV file and initializes class-level TestData and TwinData."""
        cls.data = pd.read_csv(f"{cls.data_path}/{file_name}.csv")

        cls.TestData = {key: cls.get_test_item(cls, key) for key in cls.data["TEST_ITEM_COMMUNICATIONKEY"].unique()}
        cls.TwinData = {key: cls.get_twin_item(cls, key, 10) for key in cls.data["TEST_ITEM_COMMUNICATIONKEY"].unique()}
    
    @classmethod
    def download_from_bq(cls, table_id: str = "CN_data_to_fetch", filename: str = "twins_100"):
        """Downloads data from BigQuery and saves it as a CSV file."""
        client = bq.Client(project=cls.project_id)
        table_ref = f"{cls.project_id}.{cls.dataset_id}.{table_id}"
        
        query = f"SELECT * FROM `{table_ref}`"
        df = client.query(query).to_dataframe()
        df.to_csv(f"{cls.data_path}/{filename}.csv", index=False)

    def get_test_item(self, key: int) -> pd.DataFrame:
        """Retrieves test item data."""
        df = self.data.loc[
            (self.data["TEST_ITEM_COMMUNICATIONKEY"] == key) & 
            (self.data["TEST_ITEM_COMMUNICATIONKEY"] == self.data["TWIN_ITEM_COMMUNICATIONKEY"]),
            ["CALENDAR_DATE", "TWIN_ITEM_COMMUNICATIONKEY", "ANSPRACHE"]
        ].reset_index(drop=True)
        
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"There are {nan_count} NaN values in the data which are replaced with 0s.")
            df.fillna(0, inplace=True)
        
        return df.pivot(index="CALENDAR_DATE", columns="TWIN_ITEM_COMMUNICATIONKEY", values="ANSPRACHE")

    def get_twin_item(self, key: int, num_twins: int) -> pd.DataFrame:
        """Retrieves twin item data."""
        df = self.data.loc[
            (self.data["TEST_ITEM_COMMUNICATIONKEY"] == key) & 
            (self.data["TEST_ITEM_COMMUNICATIONKEY"] != self.data["TWIN_ITEM_COMMUNICATIONKEY"]),
            ["CALENDAR_DATE", "TWIN_ITEM_COMMUNICATIONKEY", "ANSPRACHE"]
        ].reset_index(drop=True)
        
        df = df.pivot(index="CALENDAR_DATE", columns="TWIN_ITEM_COMMUNICATIONKEY", values="ANSPRACHE")
        df = df.iloc[:, :num_twins]  # Reduce to the desired number of twin items
        
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            #print(f"There are {nan_count} NaN values in the data which are replaced with 0s.")
            df.fillna(0, inplace=True)
        return df

class Resampling:
    num_samples = 5000 # maintained on class level to ensure comparability between experiments

    @classmethod
    def iid_bootstrap(cls, data: pd.DataFrame) -> pd.Series:
        """
        müsste ich eigentlich auch durch den local Bootstrap mit B = 1 und b = 1 ersetzen können
        Dann ist nur die indizierung scuffed, weil wir decimal numbers as startpuntke der intervalle haben
        """
        
        N, col = data.shape
        # Precompute column choices for each bootstrap sample
        row_choices = np.random.randint(N, size=(cls.num_samples, N))  # Shape: (num_samples, N)
        col_choices = np.random.randint(col, size=(cls.num_samples, N))  # Shape: (num_samples, N)

        # Extract sampled observations from data using NumPy advanced indexing
        sampled_observations = data.values[row_choices, col_choices]  # Shape: (num_samples, N)

        return pd.Series(np.sum(sampled_observations, axis=1), name="Bootstrap_Sums")

    @classmethod
    def lb_bootstrap(cls, data: pd.DataFrame, window_size: int, block_size: int) -> pd.Series:
        """
        Performs the Local Block Bootstrap (LBB) method from Paparoditis and Politis (2002) 
        in a vectorized manner with NumPy, adapted to work on multiple sample series.
        
        Parameters:
        - data: pd.DataFrame -> Input time series data
        - B: float -> Locality parameter to determine window size
        - b: int -> Block size for bootstrapping
        
        Returns:
        - pd.Series -> Summed bootstrap samples
        """
        
        N, col = data.shape

        # Number of blocks
        M = int(np.ceil(N / block_size))

        # Precompute column choices for each bootstrap sample
        col_choices = np.random.randint(col, size=(cls.num_samples, M))  # Shape: (num_samples, M)

        # Compute Neighborhood window starting and ending indices for each block m
        # Note: The -1 is necessary to convert from 1-based to 0-based indexing
        # J_1m = np.maximum(1, np.arange(M) * b - 0.5*window_size)-1  # Shape: (M,)
        # J_2m = np.minimum(np.arange(M) * b + 0.5*window_size, N - b + 1)-1  # Shape: (M,)

        #Test: modified version des Papers -> robustes Grenzverhalten
        J_1m = np.maximum(0, np.minimum(np.arange(M) * block_size - window_size/2, N-block_size - window_size/2)) # Shape: (M,)
        J_2m = np.maximum(0, np.minimum(np.arange(M) * block_size + window_size/2, N-block_size))                 # Shape: (M,)


        # Generate block starting indices for each block m
        I_m = np.random.randint(J_1m, J_2m+1, size=(cls.num_samples, M))  # Shape: (num_samples, M), +1 to account for open interval

        # Generate row index ranges for each block (vectorized)
        row_ranges = I_m[:, :, None] + np.arange(block_size)  # Shape: (num_samples, M, b)

        # Extract sampled blocks from data using NumPy advanced indexing
        sampled_blocks = data.values[row_ranges, col_choices[:, :, None]]  # Shape: (num_samples, M, b)

        # Flatten each sample into a 1D time series and truncate to length N
        bootstrap_samples = sampled_blocks.reshape(cls.num_samples, -1)[:, :N]  # Shape: (num_samples, N)
        #Christian: soll lieber Zeitreihe vorher auf Vielfaches von b kürzen, als unvollständige Blöcke zu verwenden

        return pd.Series(np.sum(bootstrap_samples, axis=1), name="Bootstrap_Sums")

class Metrics:

    @staticmethod
    def rmse(test_item_series: pd.Series, bootstrap_samples: pd.Series) -> float:
        """
        Wie kann ich denn den RMSE normieren für die Zeitreihen?
        -> nehme ich da die Testreihe oder die Twins zum normieren?
        """
        season_demand = np.sum(test_item_series, axis=0).values
        bias = (np.mean(bootstrap_samples) - season_demand) ** 2
        
        variance = np.var(bootstrap_samples, ddof=1)  # Using sample variance (ddof=1 for unbiased estimator)
        
        mse = bias + variance
        return np.sqrt(mse).item()
    
    @staticmethod
    def mape(test_item_series: pd.Series, bootstrap_samples: pd.Series) -> float:
        """
        Computes the Mean Absolute Percentage Error (MAPE) as a percentage value.
        """
        season_demand = np.sum(test_item_series, axis=0).values
        return np.mean(np.abs(bootstrap_samples - season_demand) / season_demand) * 100

    @staticmethod
    def mae(test_item_series: pd.Series, bootstrap_samples: pd.Series) -> float:
        """
        Computes the Mean Absolute Error (MAE).
        """
        season_demand = np.sum(test_item_series, axis=0).values
        return np.mean(np.abs(bootstrap_samples - season_demand))

    @staticmethod
    def discrete_wasserstein(dist1: pd.Series, dist2:pd.Series, p: int= 2) -> float:
        """
        Compute the p-Wasserstein distance between two discrete one dimensional distributions.
        own implementation bc wasserstein distance in stats package is only defined for the first order


        Parameters:
        - dist1: np.array, first distribution samples
        - dist2: np.array, second distribution samples
        - p: int, order of Wasserstein distance
        
        Returns:
        - Wasserstein-p distance (float)
        """
        dist1_sorted = np.sort(dist1)
        dist2_sorted = np.sort(dist2)

        return np.power(np.sum(np.abs(dist1_sorted - dist2_sorted) ** p) / len(dist1), 1 / p)

### will be run on overy import    
InputData.load_data()
print("succes")