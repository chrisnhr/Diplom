class ParameterTuning:

    #input_data = InputData()
    max_block_size = 30
    max_window_size = 60
    max_twin_num = 10
    max_item_keys = 10

    @classmethod
    def generate_grid(cls):
        """
        Generate a grid of parameters for the Local Block Bootstrap method.
        """
        block_sizes = np.arange(1, cls.max_block_size + 1, 3 )
        window_size = np.arange(2, cls.max_window_size + 1 , 4)

        return [(w, b, cls.max_twin_num) for w in window_size for b in block_sizes]

    @classmethod
    def iid_grid_samples(cls):

        results = []
        for test_item_key in tqdm(cls.input_data.TwinData.keys(), desc= "Test Items", total = cls.max_item_keys):
            for params in tqdm(cls.generate_grid(), desc = "Parameter Grid", leave = False):
                results.append(
                    {
                        "TEST_ITEM_COMMUNICATIONKEY": test_item_key,
                        "window_size": params[0],
                        "block_size": params[1],
                        "num_twins": params[2],
                        "Sample_Series": Resampling.iid_bootstrap(cls.input_data.TwinData[test_item_key])
                    }
                )

        return pd.DataFrame(results)

    @classmethod
    def lbb_grid_samples(cls):
        """
        Perform parallelized sample generation with the parameter grid as the outer loop.
        cache testdata before parallelization -> cache decorator makes it impossible to pickle the function
        test with inner and outer loops the grid ws memor funcito
        """

        # ✅ Extract parameter grid first (outer loop)
        param_grid = cls.generate_grid()

        # ✅ Define a function that only uses picklable objects
        def sampling_process(params):
            w, b, num_twins = params

            # Process all test items for a given parameter set
            return [
                {
                    "TEST_ITEM_COMMUNICATIONKEY": test_item_key,
                    "window_size": w,
                    "block_size": b,
                    "num_twins": num_twins,
                    #"Samples": Resampling.lb_bootstrap(cls.input_data.get_twin_item(test_item_key, num_twins), w, b)
                    "Samples": Resampling.lb_bootstrap(cls.input_data.TwinData[test_item_key], w, b)
                }
                for test_item_key in cls.input_data.TwinData.keys()
            ]

        # ✅ Run parallel execution with only picklable arguments
        results_list = Parallel(n_jobs=3, backend="loky")(
            delayed(sampling_process)(params) for params in tqdm(param_grid, desc="Parallel Processing", total = cls.max_item_keys)
        )

        # ✅ Flatten results efficiently
        results_flat = list(chain.from_iterable(results_list))

        return pd.DataFrame(results_flat)
    
    @classmethod
    def sampling_process(cls):

        param_grid = cls.generate_grid()
        batch_results = []

        for i, test_item_key in tqdm(enumerate(cls.input_data.TwinData.keys()), total = cls.max_item_keys):
            batch_results.append(
                {
                    "TEST_ITEM_COMMUNICATIONKEY": test_item_key,
                    "window_size": w,
                    "block_size": b,
                    "num_twins": num_twins,
                    "Samples": Resampling.lb_bootstrap(cls.input_data.TwinData[test_item_key], w, b).tolist()
                }
                for w, b , num_twins in tqdm(param_grid, desc = "Parameter Grid", leave = False)
            )

            if i % 5 == 0:
                pd.DataFrame(batch_results).to_csv("lbb_samples.csv", mode="a", header=False, index=False, quoting=1)
                batch_results = []  # Clear memory

        return batch_results