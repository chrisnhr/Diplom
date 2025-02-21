import os
import subprocess

def create_folders():
    folders = ['archive', 'plots', 'data', 'queries','results']

    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"'{folder}' directory created or already exists.")
        except Exception as e:
            print(f"An error occurred while creating '{folder}': {e}")

def setup_conda_environment():
    if os.path.isfile('environment.yml'):
        try:
            subprocess.run(['conda', 'env', 'create', '-f', 'environment.yml'], check=True)
            print("Conda environment set up successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while setting up the conda environment: {e}")
    else:
        print("No 'environment.yml' file found. Skipping conda environment setup.")

if __name__ == "__main__":
    create_folders()
    setup_conda_environment()
    print("\nTo activate the Conda environment, run the following command:")
    print("conda activate diplom")