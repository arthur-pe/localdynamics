import os
import importlib.util

def load_data(max_time_dim=2000, ignore=()):
    data_root = "data"
    datasets = []

    # Iterate through each dataset directory
    for dataset_dir in os.listdir(data_root):
        dataset_path = os.path.join(data_root, dataset_dir)
        if os.path.isdir(dataset_path):  # Ensure it's a directory

            if dataset_dir.split('/')[-1] in ignore or dataset_dir.split('/')[-1] == '__pycache__':
                continue
            import_file = os.path.join(dataset_path, "import_data.py")

            if os.path.isfile(import_file):  # Ensure import_data.py exists
                # Dynamically import the import_data.py module
                spec = importlib.util.spec_from_file_location("import_data", import_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Call the import_data() function
                if hasattr(module, "import_data"):
                    datasets = datasets + module.import_data(max_time_dim=max_time_dim)
                else:
                    print(f"No function 'import_data' found in {import_file}")
            else:
                print(f"No file 'import_data.py' found in {dataset_path}")

    return datasets