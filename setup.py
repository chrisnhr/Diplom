import os
import subprocess
import sys

def create_folders():
    # List of folders to create
    folders = ['archive', 'configs', 'plots', 'data']

    # Create each folder if it does not exist
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"'{folder}' directory created or already exists.")
        except Exception as e:
            print(f"An error occurred while creating '{folder}': {e}")

def setup_conda_environment():
    # Check if environment.yml file exists
    if os.path.isfile('environment.yml'):
        try:
            # Create the conda environment using the environment.yml file
            subprocess.run(['conda', 'env', 'create', '-f', 'environment.yml'], check=True)
            print("Conda environment set up successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while setting up the conda environment: {e}")
    else:
        print("No 'environment.yml' file found. Skipping conda environment setup.")

def print_activation_instructions():
    print("\nTo activate the Conda environment, run the following command:")
    print("conda activate diplom")

if __name__ == "__main__":
    create_folders()
    setup_conda_environment()
    print_activation_instructions()