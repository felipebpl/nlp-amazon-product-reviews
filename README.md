# Sentiment Analysis of Amazon Product Reviews

This project performs sentiment analysis on Amazon product reviews to classify customer feedback as positive or negative. By leveraging natural language processing (NLP) and machine learning techniques, businesses can gain insights into customer satisfaction, identify areas for improvement, and enhance their products and services.

## Dataset

We use the [Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products) dataset from Kaggle, which contains a large collection of Amazon product reviews including ratings, review texts, and other metadata.

### Downloading the Dataset

1. **Access the Dataset:**
   - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products).
   - You may need to create a Kaggle account if you don't have one.

2. **Download the Dataset:**
   - Click on the **Download** button to download the dataset files.
   - Save the files to a directory named `data` within the project folder.

## Business Case

Understanding customer sentiment is crucial for businesses to:

- **Improve Products and Services:** Identify common issues and areas for enhancement.
- **Customer Satisfaction:** Monitor satisfaction levels to retain customers.
- **Competitive Advantage:** Stay ahead by responding to customer feedback promptly.
- **Market Trends:** Detect trends and preferences in the market.

By automating sentiment analysis using machine learning, companies can process vast amounts of data efficiently and make data-driven decisions.

## Project Structure

The project is divided into several sections, each focusing on different aspects of the analysis:

### 1. Data Loading and Preprocessing

- **Objective:** Load the dataset and perform initial preprocessing.
- **Actions:**
  - Read the data files and combine them into a single DataFrame.
  - Handle missing values and correct data types.
  - Merge relevant text fields to create a combined text column for analysis.

### 2. Exploratory Data Analysis (EDA)

- **Objective:** Explore and visualize the data to understand its distribution and characteristics.
- **Actions:**
  - Analyze the distribution of ratings.
  - Visualize the frequency of reviews over time.
  - Identify common words and phrases in the reviews.
  - Examine the balance of positive vs. negative reviews.

### 3. Classification Pipeline

- **Objective:** Build and evaluate machine learning models to classify reviews as positive or negative.
- **Actions:**
  - Convert ratings into binary sentiment labels.
  - Implement various classifiers, including Logistic Regression and XGBoost.
  - Use cross-validation to assess model performance.
  - Generate confusion matrices to analyze misclassifications.
  - Identify the most important words contributing to the classification.

### 4. Dataset Size and Learning Curves

- **Objective:** Assess the impact of dataset size on model performance.
- **Actions:**
  - Plot learning curves to visualize training and validation scores.
  - Determine if increasing the dataset size could improve accuracy.
  - Analyze whether the model suffers from overfitting or underfitting.

### 5. Topic Analysis

- **Objective:** Analyze topics within the reviews and evaluate classification performance across different topics.
- **Actions:**
  - Apply topic modeling using Latent Dirichlet Allocation (LDA).
  - Assign topics to each review and interpret the main themes.
  - Assess classification accuracy within each topic.
  - Implement a two-layer classifier that first identifies the topic and then classifies sentiment, improving overall performance.

## How to Run the Project

### Prerequisites

- Python 3.7 or higher
- Recommended: Create a virtual environment to manage dependencies.

### Setting Up the Environment

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/felipebpl/nlp-amazon-product-reviews.git
   cd amazon-product-reviews-sentiment-analysis
   ```

2. **Create a Virtual Environment:**

  ```bash
  python3 -m venv venv
  ```

3. **Activate the Virtual Environment:**
  
- On Unix or MacOS:

  ```bash
  source venv/bin/activate
  ```

- On Windows:
  ```bash
  venv\Scripts\activate
  ```

4. **Install the Requirements::**
 
  ```bash
  pip install -r requirements.txt
  ```

5. **Running the Analysis:**

    1. Place the Dataset:

      - Ensure the dataset files are located in the data directory within the project folder.

    2. Run the Main Script:

      ```bash
      python src/main.py
      ```
      - This script will execute all sections of the project sequentially.

    3. View the Outputs:

      - Results, plots, and analysis summaries will be displayed in the console or saved to the output directory as configured.
