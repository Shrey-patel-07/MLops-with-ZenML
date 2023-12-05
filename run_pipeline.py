from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__  == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path = "data/cleaned_data.csv")
    
    # mlflow ui --backend-store-uri "file:/home/shrey/.config/zenml/local_stores/c40890c1-396e-421c-b3ab-7cdbd3205f79/mlruns"