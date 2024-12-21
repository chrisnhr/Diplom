# diplom

setup:
conda env create -f environment.yml
conda activate diplom
conda install ...
conda env update -f environment.yml
conda deactivate
conda remove --name thesis_env --all