import mlrun
from kfp import dsl

@dsl.pipeline(name="breast-cancer-demo")
def pipeline(model_name="classifier"):
    
    ingest = mlrun.run_function(
        "load-iris-data",
        name="load-iris-data",
        params={"format": "pq", "model_name": model_name},
        outputs=["dataset"],
    )
    
    train = mlrun.run_function(
        "trainer",
        inputs={"dataset": ingest.outputs["dataset"]},
        hyperparams={
            "n_estimators": [10, 100],
            "max_depth": [2, 8],
            "min_samples_split": [2, 5]
        },
        selector="max.accuracy",
        outputs=["model"],
    )
    
    deploy = mlrun.deploy_function(
        "serving",
        models=[{"key": model_name, "model_path": train.outputs["model"], "class_name": "ClassifierModel"}],
        mock=True
    )