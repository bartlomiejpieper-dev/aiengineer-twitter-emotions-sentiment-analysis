
# Twitter Emotions Sentiment Analysis

A machine learning project designed to analyze emotions in Twitter posts [Twitter emotion classification dataset](https://www.kaggle.com/datasets/aadyasingh55/twitter-emotion-classification-dataset). using various neural network models. The model classifies tweets into different emotion categories, helping to understand the sentiment of user posts.

```python
{
    0: 'sadness', 
    1: 'joy', 
    2: 'love', 
    3: 'anger', 
    4: 'fear', 
    5: 'surprise'
}
```

Example:
| Text                         | Label            |
|------------------------------|------------------|
| "I am so happy today!"       | Joy              |
| "This is so frustrating!"    | Anger            |
| "Feeling a bit down..."      | Sadness          |

#### !!! WARNING !!!
Training models takes a lot of time. See charts of already trained models (see `src/images/mlruns`) for analysis or Docker container for predictions.

## Table of Contents

- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [API Deployment](#api-deployment)

## Features
- **Sentiment Analysis**: Classifies tweets into multiple emotions such as happiness, sadness, anger, etc.
- **Neural Network Models**: Uses an LSTM, BiLSTM, GRU neural networks for sequence modeling.
- **Model Logging and Tracking**: Tracks experiment runs and metrics using MLflow.
- **Dokerized API**: Returns models info and predictions in dockerized api.

## File Structure

```
twitter_emotions_sentiment_analysis/
├── .github/                                              # Github workflows
├── api/                                                  # Api files
├── data/                                                 # Data files and datasets
├── packages/                                             # Internal packages: mypreporcessing & myvisualization
├── src/                                                  # Source code
│   ├── images/                                           # Metrics of trained models
│   ├── models/                                           # Saved models
│   ├── mlruns/                                           # MLflow tracking files
│   └── twitter_emotions_sentiment_analysis.ipynb         # Python workbook with experiments
├── Dockerfile                                            # Image definition
├── pyproject.toml                                        # Dependencies info for poetry
└── README.md                                             # Project documentation
```


## Installation

1. **Install**:
   - [Python 3.10.0](https://www.python.org/downloads/)
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Poetry package:

   ```bash
   pip install poetry
   ```

2. **Set up a virtual environment using Poetry**:

   ```bash
   rm -rf .venv  # Remove existing environment if needed
   poetry config virtualenvs.in-project true  # Create the environment within the project
   poetry shell  # Activate the environment
   poetry install  # Install all dependencies
   poetry env info  # Display environment information
   ```

3. **MLflow**:

   - To view experiment runs, first execute `Run All` in the **python workbook**, which will create the `src/mlruns` directory. Then, start the MLflow UI with:

   ```bash
   mlflow ui --backend-store-uri ./src/mlruns
   ```

   - Remember, training models takes a lot of time. It is adviced to track already trained models or use Docker images. See `src/images/mlruns` catalog for details.

4. **GitHub Workflow & Docker Image**:

   - Go to the [GitHub Actions page](https://github.com/CodecoolGlobal/aiengineer-summary-4-python-bartlomiejpieper-dev/actions/workflows/docker_build.yml) to trigger the Docker build workflow on-demand using [docker_build.yml](https://github.com/CodecoolGlobal/aiengineer-summary-4-python-bartlomiejpieper-dev/blob/development/.github/workflows/docker_build.yml).
   - Once workflow succeeds download the built Docker image from [GitHub Packages](https://github.com/orgs/CodecoolGlobal/packages?repo_name=aiengineer-summary-4-python-bartlomiejpieper-dev) 
   - Check if you are authorized using **Peronal Access Token**:
   ```bash
   echo YOUR_GITHUB_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
   ```   
   - Download image from github packages
   ```bash
   docker pull ghcr.io/codecoolglobal/model:latest
   ```
   - Run container
   ```bash
   docker run -p 8000:8000 ghcr.io/codecoolglobal/model:latest
   ```

## API Deployment

### Running the API Using Docker

1. **Open Docker Desktop**
1. **Build and Run the Docker Image**:

   ```bash
   docker build -t myapi .
   docker run -p 8000:8000 myapi
   ```

3. **Test the API**:

    ```browser
    http://localhost:8000/docs
    ```

   Access the [Swagger Documentation](http://localhost:8000/docs) for interactive API documentation.

### Running the API Using Uvicorn

1. Run the API locally:

    ```bash
    uvicorn api.src.main:app --reload --port 8000
    ```

2. Test the API:

    ```browser
    http://localhost:8000/docs
    ```

    Access [Swagger Documentation](http://localhost:8000/docs) for further details.

