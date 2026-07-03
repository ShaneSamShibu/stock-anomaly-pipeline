"""
SageMaker Deployment Script
============================
This script deploys the trained Isolation Forest model to AWS SageMaker
as a real-time inference endpoint.

To run this for real:
1. Complete `aws configure` with valid credentials
2. Ensure your IAM user has SageMaker and S3 permissions
3. Run: python deploy_sagemaker.py

Estimated cost: ~$0.10 for a short test (ml.t2.medium @ $0.065/hr)
Remember to delete the endpoint after testing to avoid ongoing charges.
"""

import boto3
import sagemaker
import joblib
import tarfile
import os

# ── Configuration ─────────────────────────────────────────────────────────────
REGION = "us-east-1"
BUCKET_PREFIX = "stock-anomaly-pipeline"
MODEL_DIR = "model"

# ── Step 1: Package the model for SageMaker ───────────────────────────────────
# SageMaker expects model artifacts in a tar.gz archive called model.tar.gz
print("Packaging model artifacts...")
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add(f"{MODEL_DIR}/isolation_forest.pkl", arcname="isolation_forest.pkl")
    tar.add(f"{MODEL_DIR}/scaler.pkl", arcname="scaler.pkl")
print("model.tar.gz created.")

# ── Step 2: Upload to S3 ──────────────────────────────────────────────────────
# SageMaker pulls the model from S3, not your local machine
print("Uploading model to S3...")
session = sagemaker.Session(boto3.Session(region_name=REGION))
bucket = session.default_bucket()
s3_model_path = session.upload_data(
    path="model.tar.gz",
    bucket=bucket,
    key_prefix=f"{BUCKET_PREFIX}/model"
)
print(f"Model uploaded to: {s3_model_path}")

# ── Step 3: Define the SageMaker model ───────────────────────────────────────
# We use SKLearnModel since our model was built with scikit-learn
from sagemaker.sklearn.model import SKLearnModel

role = sagemaker.get_execution_role()

sklearn_model = SKLearnModel(
    model_data=s3_model_path,
    role=role,
    framework_version="1.2-1",
    py_version="py3",
    entry_point="serve_model.py",
)

# ── Step 4: Deploy the endpoint ───────────────────────────────────────────────
print("Deploying endpoint (this takes 5-10 minutes)...")
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name="stock-anomaly-detector"
)
print(f"Endpoint deployed: stock-anomaly-detector")

# ── Step 5: Test the endpoint ─────────────────────────────────────────────────
import numpy as np
test_data = np.array([
    [281.2, 281.2, 0.0, 0.001, 0.5],   # normal
    [150.0, 280.0, 25.0, -0.15, 4.2],  # anomaly
])
response = predictor.predict(test_data)
print(f"Test predictions: {response}")

# ── Step 6: Cleanup (IMPORTANT — avoids ongoing charges) ─────────────────────
# Uncomment this after verifying the endpoint works:
# predictor.delete_endpoint()
# print("Endpoint deleted.")

print("\nDeployment complete.")
print("⚠️  Remember to delete the endpoint when done to avoid charges:")
print("   predictor.delete_endpoint()")