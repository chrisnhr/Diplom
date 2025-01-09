import os

def create_folders():
    folders = ['archive', 'configs', 'plots', 'data', 'results']

    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"'{folder}' directory created or already exists.")
        except Exception as e:
            print(f"An error occurred while creating '{folder}': {e}")

if __name__ == "__main__":
    create_folders()