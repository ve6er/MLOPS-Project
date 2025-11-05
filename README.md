# Automated MLOps Pipeline for Renewable Energy Share Prediction

This project provides a comprehensive, end-to-end MLOps framework to predict a country's renewable energy share. It uses environmental, demographic, and economic factors from the sustainable energy (2020-2022) dataset.

The core of this project is a **fully automated CI/CD pipeline** built with GitHub Actions. This pipeline handles everything from data ingestion and versioning (with DVC) to model retraining, containerization (with Docker), and automatic deployment to AWS.

The deployed solution provides a robust, data-driven tool for policymakers to understand the impact of policy changes on renewable energy goals.

## Problem Statement

The goal is to predict the renewable energy share of a country using various economic and demographic factors. Understanding this relationship is crucial for developing nations to shape effective policies, increase energy production, and align with the UN Sustainable Development Goals (UN SGDs).

## Key Features

  * **End-to-End Automation:** The entire pipeline from data ingestion to deployment is 100% automated using GitHub Actions.
  * **Data & Model Versioning:** **DVC** is used to version large data files and models, keeping the Git repository lightweight.
  * **Continuous Integration (CI):** On every `git push`, the pipeline automatically retrains and evaluates three regression models (Ridge, Lasso, and Random Forest).
  * **Automatic Model Selection:** The pipeline automatically compares the new models against the old benchmark and selects the best-performing one for deployment.
  * **Continuous Deployment (CD):** The best model and application are containerized with **Docker** and automatically deployed to an **AWS EC2** instance.
  * **Cloud Storage:** All data, models, and artifacts are versioned and stored in an **AWS S3** bucket.

## Tech Stack

  * **Data Science:** Pandas, Scikit-learn
  * **ML Models:** Ridge Regression, Lasso Regression, Random Forest Regressor
  * **MLOps Tools:** DVC (Data Version Control)
  * **CI/CD:** GitHub Actions 
  * **Containerization:** Docker 
  * **Cloud Platform:** Amazon Web Services (AWS)
      * **Storage:** AWS S3
      * **Compute:** AWS EC2

## MLOps Pipeline Architecture

The pipeline is broken into two main workflows: Continuous Integration (CI) and Continuous Delivery (CD).

### 1\. Continuous Integration (CI)

This workflow is triggered on every `git push` to the main branch.

1.  **Checkout & Setup:** A GitHub Actions runner checks out the code and sets up the Python environment.
2.  **Configure AWS:** Securely configures AWS credentials using GitHub Secrets.
3.  **Run Pipeline (`dvc repro`):** This command executes the entire ML pipeline:
      * **Data Ingestion:** Pulls the latest data from the S3 bucket.
      * **Preprocessing:** Handles null values, drops columns, and One-Hot-Encodes categorical features.
      * **Feature Engineering:** Applies a Box-Cox Transformation to normalize skewed data and splits it into 80-20 train/test sets.
      * **Model Training:** Trains Ridge, Lasso, and Random Forest models on the new data.
      * **Model Evaluation:** Logs metrics (R2, MSE, MAE, RMSE) for all models using DVC Live.
      * **Model Selection:** Compares the new models' R2 scores as well as carbon emission and selects the best one with optimal trade-off between sustainability and performance.
4.  **Push Artifacts (`dvc push`):** The new versioned data and the best model are pushed to the AWS S3 remote storage.
5.  **Commit Lock File (`git push`):** The `dvc.lock` file, which tracks the new data and model versions, is committed back to the Git repository.

### 2\. Continuous Delivery (CD)

This workflow is triggered *only* after the CI workflow completes successfully.

1.  **Pull Artifacts (`dvc pull`):** The runner pulls the production-ready model, scaler, and OHE artifacts from S3.
2.  **Build Docker Image:** A new Docker image is built, packaging the application and the new model artifacts.
3.  **Push Image to S3:** The Docker image is saved as a `.tar` file and pushed to the S3 bucket.
4.  **Deploy to EC2:**
      * The GitHub runner connects to the AWS EC2 instance via SSH using a private key stored in Secrets.
      * A script is executed on the EC2 instance to:
          * Download the new `image.tar` from S3.
          * Stop the old, currently running Docker container.
          * Load and run the new Docker container, exposing it on port 80.

## Model Results

Three models were trained and evaluated. The Ridge Regressor provided the best trade-off by a significant margin.

| Model | MSE | RMSE | MAE | R2 Score | Adjusted R2 Score | Carbon Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Lasso | 2.464 | 1.570 | 1.181 | 0.397 | 0.184 |  |
| Ridge | 0.168 | 0.410 | 0.283 | 0.959 | 0.944 |  |
| Random Forest | 0.078 | 0.279 | 0.123 | 0.981 | 0.974 |  |

*Table 1: Evaluation of all models*

The final deployed model is the **Ridge Regressor**, which achieved an **$R^{2}$ score of 0.959**.

## Getting Started

To run this project, you will need to set up the necessary cloud infrastructure and local tools.

### Prerequisites

  * Python 3.11+
  * Git
  * DVC
  * Docker
  * An AWS Account

### Configuration

1.  **Clone Repository:**

    ```bash
    git clone https://github.com/ve6er/MLOPS-Project.git
    cd MLOPS-Project
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure AWS:**

      * Create an **S3 bucket** for DVC remote storage and for storing the Docker image.
      * Create an **EC2 instance** (e.g., t2.micro) and configure its security group to allow SSH (port 22) and HTTP (port 80).
      * Configure your local AWS CLI with `aws configure`.

4.  **Configure DVC:**

      * Set up your S3 bucket as the DVC remote:
        ```bash
        dvc remote add -d my-remote s3://your-bucket-name/dvc-storage
        git commit .dvc/config -m "Configure DVC remote"
        ```

5.  **Configure GitHub Secrets:**

      * In your GitHub repository, go to `Settings > Secrets and variables > Actions` and add the following secrets:
          * `AWS_ACCESS_KEY_ID`: Your AWS access key.
          * `AWS_SECRET_ACCESS_KEY`: Your AWS secret key.
          * `AWS_REGION`: Your AWS region (e.g., `us-east-1`).
          * `AWS_S3_BUCKET`: The name of your S3 bucket.
          * `EC2_HOST`: The public IP address or DNS of your EC2 instance.
          * `EC2_USER`: The user for your EC2 instance (e.g., `ec2-user`).
          * `SSH_PRIVATE_KEY`: The private key (.pem) used to SSH into your EC2 instance.

### Running the Pipeline

  * **Locally:**

    ```bash
    # Pull the latest data from S3
    dvc pull

    # Run the full pipeline
    dvc repro
    ```

  * **Automatically (via CI/CD):**

      * Simply commit and push your changes to GitHub:
        ```bash
        git add .
        git commit -m "Updated data processing logic"
        git push
        ```
      * This will trigger the full CI/CD pipeline, and your changes will be live on the EC2 instance in a few minutes.

## Future Work

Future work can focus on enhancing the pipeline's capabilities:

  * **Real-time Data:** Support data ingestion in real-time as changes are reflected in World Bank and IEA repositories.
  * **Multimodal Inputs:** Integrate real-time analysis of economic factors and multimodal inputs like satellite imagery.
  * **Explainability:** Integrate SHAP and LIME to provide better interpretability and transparency for policy decisions.

## Authors & Acknowledgments

This project was submitted by:

  * Sejal Dubey
  * Samriddhi Kumari
  * Veer Kukreja

Under the guidance of:

  * Dr. Mayur Gaikwad
  * Dr. Aniket Shahade
