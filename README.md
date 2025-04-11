# Capital Markets Agents Service

This repository hosts the backend for the **Capital Markets Agents** service.

## Where Does MongoDB Shine?

MongoDB is a powerful database solution that excels in managing financial data, particularly in the capital markets domain. Here are some key reasons why MongoDB is an ideal choice for financial services:

## The 4 Pillars of the Document Model

1. **Easy**: [MongoDB's document model](https://www.mongodb.com/resources/basics/databases/document-databases) naturally fits with object-oriented programming, utilizing BSON documents that closely resemble JSON. This design simplifies the management of complex data structures such as user accounts, allowing developers to build features like account creation, retrieval, and updates with greater ease.

2. **Fast**: Following the principle of "Data that is accessed together should be stored together," MongoDB enhances query performance. This approach ensures that related data—like user and account information—can be quickly retrieved, optimizing the speed of operations such as account look-ups or status checks, which is crucial in services demanding real-time access to operational data.

3. **Flexible**: MongoDB's schema flexibility allows account models to evolve with changing business requirements. This adaptability lets financial services update account structures or add features without expensive and disruptive schema migrations, thus avoiding costly downtime often associated with structural changes.

4. **Versatile**: The document model in MongoDB effectively handles a wide variety of data types, such as strings, numbers, booleans, arrays, objects, and even vectors. This versatility empowers applications to manage diverse account-related data, facilitating comprehensive solutions that integrate user, account, and transactional data seamlessly.

## MongoDB Key Features

- **Time Series** - ([More info](https://www.mongodb.com/products/capabilities/time-series)): For storing market data in a time series format.
- **Atlas Vector Search**  ([More info](https://www.mongodb.com/products/platform/atlas-vector-search)): For enabling vector search on financial news data.

## Tech Stack

- [MongoDB Atlas](https://www.mongodb.com/atlas/database) for the database.
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework.
- [Poetry](https://python-poetry.org/) for dependency management.
- [Uvicorn](https://www.uvicorn.org/) for ASGI server.
- [Docker](https://www.docker.com/) for containerization.

## Relevant Python Packages

- [yfinance](https://pypi.org/project/yfinance/) for extracting market data from Yahoo Finance.
- [pyfredapi](https://pypi.org/project/pyfredapi/) for extracting macroeconomic data from the FRED API.
- [pandas](https://pandas.pydata.org/) for data manipulation.
- [scheduler](https://pypi.org/project/scheduler/) for job scheduling.
- [transformers](https://huggingface.co/transformers/) for natural language processing.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **MongoDB Atlas** account - [Register Here](https://account.mongodb.com/account/register)
- **Python 3.10 or higher**
- **Poetry** (install via [Poetry's official documentation](https://python-poetry.org/docs/#installation))

## Setup Instructions

### Step 1a: Set Up MongoDB Database and Collections

1. Log in to **MongoDB Atlas** and create a database named `agentic_capital_markets`. Ensure the name is reflected in the environment variables.
2. Create the following collections:
   - `financial_news` (for storing financial news data) - You can export some sample data to this collection using `backend/loaders/db/collections/agentic_capital_markets.financial_news.json` file.
   - `pyfredapiMacroeconomicIndicators` (for storing macroeconomic data) - You can export some sample data to this collection using `backend/loaders/db/collections/agentic_capital_markets.pyfredapiMacroeconomicIndicators.json` file.
   - `yfinanceMarketData` (for storing market data) - You can export some sample data to this collection using `backend/loaders/db/collections/agentic_capital_markets.yfinanceMarketData.json` file. Additionally, there are some more backup files in this directory that you can use to populate the collection:  `backend/loaders/backup/*`

> **_Note:_** For creating the time series collection, you can run the following python script located in the `backend/loaders/db/` directory: `create_time_series_collection.py`. Make sure to parametrize the script accordingly.

### Step 1b: Set Up Vector Search Index

1. Create the vector search index for the `financial_news` collection.

> **_Note:_** For creating the vector search index, you can run the following python script located in the `backend/loaders/db/` directory: `vector_search_idx_creator.py`. Make sure to parametrize the script accordingly.


### Step 2: Add MongoDB User

Follow [MongoDB's guide](https://www.mongodb.com/docs/atlas/security-add-mongodb-users/) to create a user with **readWrite** access to the `agentic_capital_markets` database.

## Configure Environment Variables

Create a `.env` file in the `/backend` directory with the following content:

```bash
MONGODB_URI=
DATABASE_NAME="agentic_capital_markets"
APP_NAME="your_app_name"
AWS_REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
FRED_API_KEY="your_fred_api_key"
YFINANCE_TIMESERIES_COLLECTION = "yfinanceMarketData"
PYFREDAPI_COLLECTION = "pyfredapiMacroeconomicIndicators"
NEWS_COLLECTION = "financial_news"
SCRAPE_NUM_ARTICLES = 1
```

## Running the Backend

### Virtual Environment Setup with Poetry

1. Open a terminal in the project root directory.
2. Run the following commands:
   ```bash
   make poetry_start
   make poetry_install
   ```
3. Verify that the `.venv` folder has been generated within the `/backend` directory.

### Start the Backend

To start the backend service, run:

```bash
poetry run uvicorn main:app --host 0.0.0.0 --port 8004
```

> Default port is `8004`, modify the `--port` flag if needed.

## Running with Docker

Run the following command in the root directory:

```bash
make build
```

To remove the container and image:

```bash
make clean
```

## API Documentation

You can access the API documentation by visiting the following URL:

```
http://localhost:<PORT_NUMBER>/docs
```
E.g. `http://localhost:8004/docs`

> **_Note:_** Make sure to replace `<PORT_NUMBER>` with the port number you are using and ensure the backend is running.

## Common errors

- Check that you've created an `.env` file that contains the required environment variables.

## Future tasks

- [ ] Add tests
- [ ] Evaluate SonarQube for code quality
- [ ] Automate the deployment process using GitHub Actions or CodePipeline
