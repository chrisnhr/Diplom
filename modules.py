import numpy as np
import pandas as pd
from google.cloud import bigquery as bq
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import os
import warnings
from scipy.ndimage import gaussian_filter1d

class InputData:
    project_id = "brain-flash-dev"
    dataset_id = "dagster_common"
    data_path = "data"

    def __init__(self, demand_column = "ANSPRACHE", max_twin_num = 10, kernel_size = 1, file_name: str = "twins_100"):
        """Loads data from a CSV file and initializes unique communication keys."""

        self.demand_column = demand_column
        self.max_twin_num = max_twin_num
        self.kernel_size = kernel_size #actually its the kernel size
        self.data = pd.read_csv(f"{self.data_path}/{file_name}.csv")
        self.TestData = {key : self.get_test_item(key) for key in self.data["TEST_ITEM_COMMUNICATIONKEY"].unique()}
        self.TwinData = {key : self.get_twin_item(key, self.max_twin_num) for key in self.data["TEST_ITEM_COMMUNICATIONKEY"].unique()}

    @staticmethod
    def apply_kernel_smoothing(data: pd.DataFrame, kernel_size: int, sigma = None):
        if sigma is None:
            sigma = kernel_size / 3  # Rule of thumb: sigma ~ kernel_size / 3

        for column in data.columns:
            smoothed = gaussian_filter1d(data[column], sigma, mode='nearest')
            data[column] = smoothed * (data[column].sum() / smoothed.sum())
        return data

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
            ["CALENDAR_DATE", "TWIN_ITEM_COMMUNICATIONKEY", self.demand_column]
        ].reset_index(drop=True)
        
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"There are {nan_count} NaN values in the data which are replaced with 0s.")
            df.fillna(0, inplace=True)

        pivoted_df = df.pivot(index="CALENDAR_DATE", columns="TWIN_ITEM_COMMUNICATIONKEY", values=self.demand_column)
        
        return self.apply_kernel_smoothing(pivoted_df, kernel_size = self.kernel_size)
    
    def get_twin_item(self, key: int, num_twins: int) -> pd.DataFrame:
        """Retrieves twin item data."""
        df = self.data.loc[
            (self.data["TEST_ITEM_COMMUNICATIONKEY"] == key) & 
            (self.data["TEST_ITEM_COMMUNICATIONKEY"] != self.data["TWIN_ITEM_COMMUNICATIONKEY"]),
            ["CALENDAR_DATE", "TWIN_ITEM_COMMUNICATIONKEY", self.demand_column]
        ].reset_index(drop=True)
        
        df = df.pivot(index="CALENDAR_DATE", columns="TWIN_ITEM_COMMUNICATIONKEY", values=self.demand_column)
        df = df.iloc[:, :num_twins]  # Reduce to the desired number of twin items
        
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            #print(f"There are {nan_count} NaN values in the data which are replaced with 0s.")
            df.fillna(0, inplace=True)

        return self.apply_kernel_smoothing(df, kernel_size = self.kernel_size)

class Resampling:
    num_samples: int = 5000 # maintained on class level to ensure comparability between experiments

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
    def mpe(test_item_series: pd.Series, bootstrap_samples: pd.Series) -> float:
        """
        Computes the Mean Absolute Percentage Error (MAPE) as a percentage value.
        """
        season_demand = np.sum(test_item_series, axis=0).values
        return np.mean((bootstrap_samples - season_demand) / season_demand) * 100

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
    
class GridEvaluation:

    max_window_size = 60
    max_block_size = 30
    max_twin_number = 10
    batch_size = 5

    @staticmethod
    def evaluate_lbb(test_data: pd.DataFrame, twin_data: pd.DataFrame, test_item_key: int, b: int, w: int):

        twin_lbb = Resampling.lb_bootstrap(twin_data, window_size = w, block_size = b)
        test_lbb = Resampling.lb_bootstrap(test_data, window_size = w, block_size = b)

        summary = {
            "TEST_ITEM_COMMUNICATIONKEY": test_item_key,
            "BLOCK_SIZE": b,
            "WINDOW_SIZE": w,
            "TWIN_NUMBER": twin_data.shape[1],
            "MEAN_SAMPLE": np.mean(twin_lbb),
            "MEAN_TEST": np.mean(test_data.sum(axis=0)),
            "BIAS": np.mean(twin_lbb)-np.mean(test_data.sum(axis=0)),
            "VARIANCE": np.var(twin_lbb, ddof=1),
            "CV": np.std(twin_lbb, ddof=1)/np.mean(twin_lbb) * 100,
            "RMSE": np.sqrt(Metrics.rmse(test_data, twin_lbb)),
            "MAPE": Metrics.mape(test_data, twin_lbb),
            "MPE": Metrics.mpe(test_data, twin_lbb),
            "MAE": Metrics.mae(test_data, twin_lbb),
            "WASSERSTEIN": Metrics.discrete_wasserstein(test_lbb, twin_lbb),
        }
        return summary
    
    @staticmethod
    def evaluate_idd(test_data: pd.DataFrame, twin_data: pd.DataFrame, test_item_key: int):
        
        twin_idd = Resampling.iid_bootstrap(twin_data)
        test_idd = Resampling.iid_bootstrap(test_data)

        summary = {
            "TEST_ITEM_COMMUNICATIONKEY": test_item_key,
            "BLOCK_SIZE": 1,
            "WINDOW_SIZE": 0,
            "TWIN_NUMBER": twin_data.shape[1],
            "MEAN_SAMPLE": np.mean(twin_idd),
            "MEAN_TEST": np.mean(test_data.sum(axis=0)),
            "BIAS": np.mean(twin_idd)-np.mean(test_data.sum(axis=0)),
            "VARIANCE": np.var(twin_idd, ddof=1),
            "CV": np.std(twin_idd, ddof=1)/np.mean(twin_idd) * 100,
            "RMSE": Metrics.rmse(test_data, twin_idd),
            "MAPE": Metrics.mape(test_data, twin_idd),
            "MPE": Metrics.mpe(test_data, twin_idd),
            "MAE": Metrics.mae(test_data, twin_idd),
            "WASSERSTEIN": Metrics.discrete_wasserstein(test_idd, twin_idd),
        }
        return summary
    
    ä#hier vor ne loop über alle keys, die auch die daten sliced
    def run_lbb(twin_raw: pd.DataFrame, test_raw: pd.DataFrame, test_item_key: int, params):

            twin_lbb = Resampling.lb_bootstrap(twin_raw, *params)
            test_lbb = Resampling.lb_bootstrap(test_raw, *params)

            return GridEvaluation.evaluate_lbb(twin_raw, test_raw, twin_lbb, test_lbb, test_item_key)
    
    def run_idd(twin_raw: pd.DataFrame, test_raw: pd.DataFrame, test_item_key: int):
        
            twin_idd = Resampling.iid_bootstrap(twin_raw)
            test_idd = Resampling.iid_bootstrap(test_raw)

            return GridEvaluation.evaluate_idd(twin_raw, test_raw, twin_idd, test_idd, test_item_key)

        
        #führe lbb aus für anzahl an twins im input
        #noch aus der evaluation function das sampling auslagern, unten in der parallelisierung muss ich dann run_lbb statt der evaluation einfügen
    
    @classmethod
    def run(cls, input_data: InputData, output_file: str = "results/grid_results.csv"):
        keys = list(input_data.TestData.keys())
        batches = [keys[i:i + cls.batch_size] for i in range(0, len(keys), cls.batch_size)]
        grid = [(w, b, cls.max_twin_number) for b in range(1, cls.max_block_size + 1, 3) for w in range(1, cls.max_window_size + 1, 4)]
        
        all_results = []
        for batch in tqdm(batches, desc="Processing the parameter grid in batches"):
            
            lbb_results = Parallel(n_jobs=-1)(
                delayed(cls.evaluate_lbb)(input_data.TestData[test_item_key], input_data.TwinData[test_item_key], test_item_key, b, w)
                for w, b, _ in grid 
                for test_item_key in batch
            )
            
            idd_results = Parallel(n_jobs=-1)(
                delayed(cls.evaluate_idd)(input_data.TestData[test_item_key], input_data.TwinData[test_item_key], test_item_key)
                for test_item_key in batch
            )
            
            all_results.extend(lbb_results + idd_results)
        
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(output_file, index=False)