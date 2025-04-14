# First cell - Update function and dataset naming
import mlrun
from sklearn.datasets import load_breast_cancer
import pandas as pd

@mlrun.handler(outputs=["dataset", "label_column"])
def dataset_loader(context, format="csv"):

    breast_cancer = load_breast_cancer(as_frame=True)
    breast_cancer_dataset = breast_cancer.frame
    breast_cancer_dataset['target'] = breast_cancer.target
    
    context.logger.info('saving breast cancer dataset to {}'.format(context.artifact_path))
    context.log_dataset('breast_cancer_dataset', df=breast_cancer_dataset, format=format, index=False)
    
    return breast_cancer_dataset, "target"

if __name__ == "__main__":
    with mlrun.get_or_create_ctx("breast_cancer_generator", upload_artifacts=True) as context:
        dataset_loader(context, context.get_param("format", "csv"))