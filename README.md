# OmniFeedback AI

## Enterprise Customer Feedback Analytics & Intelligence Copilot

OmniFeedback AI is an end-to-end customer feedback analytics project that combines **Data Engineering, Machine Learning, Deep Learning, NLP, and Generative AI concepts** to analyze customer feedback and support escalation decisions.

The project takes customer feedback from multiple sources, processes and analyzes the text, predicts feedback classification and urgency, detects named entities, and presents the results through an interactive **Streamlit dashboard**.

---

## Project Objectives

The main objectives of OmniFeedback AI are:

- Analyze customer feedback from different channels
- Clean and preprocess unstructured text data
- Store structured feedback data using SQLite
- Perform exploratory data analysis (EDA)
- Convert text into numerical features using TF-IDF
- Classify customer feedback using Random Forest
- Group similar feedback using K-Means clustering
- Predict urgency using a BiLSTM deep learning model
- Detect entities from customer feedback using BERT NER
- Assign feedback priority based on urgency score
- Provide recommended actions for support teams
- Build an interactive Streamlit application for feedback analysis

---

## Technologies Used

### Programming
- Python

### Data Processing
- Pandas
- NumPy
- Regular Expressions (Regex)

### Database
- SQLite3
- SQL

### Machine Learning
- Scikit-learn
- TF-IDF
- Random Forest
- K-Means Clustering

### Deep Learning
- PyTorch
- BiLSTM

### Natural Language Processing
- NLTK
- Tokenization
- Stopword removal
- Text preprocessing
- BERT Named Entity Recognition (NER)

### Application
- Streamlit

### Model & Artifact Storage
- Pickle
- PyTorch `.pth` model

---

## Project Workflow

```text
Customer Feedback
        |
        v
Data Collection
        |
        v
Data Cleaning & Preprocessing
        |
        v
SQLite Database
        |
        v
Exploratory Data Analysis
        |
        +----------------------+
        |                      |
        v                      v
     TF-IDF                Text Processing
        |                      |
        v                      v
 Random Forest              BiLSTM
        |                      |
        v                      v
Classification            Urgency Score
        |                      |
        |                      v
        |                   Priority
        |
        +----------------------+
        |
        v
K-Means Clustering
        |
        v
BERT NER
        |
        v
Streamlit Feedback Copilot
```

---

## Dataset

The project uses customer feedback data containing fields such as:

- `feedback_id`
- `user_id`
- `raw_text`
- `channel`
- `aspect_category`
- `urgency_score`
- `label`
- `timestamp`

The project also incorporates sample feedback datasets for NLP and machine learning experimentation.

---

## Data Engineering

The project uses **SQLite3** as the relational database.

The database workflow includes:

- Creating the SQLite database
- Creating tables
- Loading customer feedback data
- Using primary keys and foreign keys
- Storing structured customer feedback
- Querying data using SQL

The project demonstrates basic relational database concepts and data organization for an analytics application.

---

## Text Preprocessing

Customer feedback is cleaned before applying machine learning and NLP techniques.

The preprocessing workflow includes:

- Converting text to lowercase
- Removing unwanted characters
- Tokenization
- Stopword handling
- Removing punctuation
- Lemmatization
- Preparing text for machine learning and deep learning models

Regular expressions are also used for text cleaning.

---

## TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert customer feedback text into numerical feature vectors.

These numerical features are then used by the machine learning classification model.

The trained TF-IDF vectorizer is saved as:

```text
tfidf.pkl
```

---

## Random Forest Classification

A **Random Forest classifier** is used to classify customer feedback.

The trained model is saved as:

```text
random_forest.pkl
```

In the Streamlit application, the saved TF-IDF vectorizer and Random Forest model are loaded and used to classify newly entered customer feedback.

The application displays the classification result as:

```text
Classification: 0 or 1
```

---

## K-Means Clustering

**K-Means clustering** is used to group customer feedback into clusters based on their text features.

Unlike supervised classification, clustering does not require predefined class labels.

The K-Means model is trained separately and retained as part of the project artifacts.

---

## BiLSTM Urgency Prediction

A **Bidirectional Long Short-Term Memory (BiLSTM)** neural network is used to predict the urgency of customer feedback.

The model processes the sequence of words in the feedback and produces an urgency score between `0` and `1`.

The trained model is saved as:

```text
bilstm_urgency.pth
```

The Streamlit application loads this saved model and uses it to calculate the urgency score for newly entered feedback.

---

## Priority Assignment

The urgency score is converted into a support priority using predefined thresholds.

```text
Urgency >= 0.70
        → High Priority

Urgency >= 0.40
        → Medium Priority

Urgency < 0.40
        → Low Priority
```

The application also provides a recommended action based on the priority.

Example:

```text
High   → Escalate to support team immediately
Medium → Review and assign to support team
Low    → Monitor feedback
```

---

## BERT Named Entity Recognition

BERT-based Named Entity Recognition is used to identify entities present in customer feedback.

The project uses:

```text
dslim/bert-base-NER
```

The Streamlit application displays detected entities and their entity types.

---

## Streamlit Application

The final application is built using **Streamlit**.

The application contains:

### Home

Provides an overview of the OmniFeedback AI system.

### EDA

Displays feedback analytics such as:

- Urgency Score
- Feedback by Channel
- Feedback by Label
- Aspect Category

### Feedback Copilot

Allows the user to enter new customer feedback and receive:

- Classification
- Urgency Score
- Priority
- Recommended Action
- Detected Entities
- Original Feedback

---

## Saved Model Artifacts

The project contains the following important trained artifacts:

```text
tfidf.pkl
random_forest.pkl
kmeans.pkl
bilstm_urgency.pth
```

These saved artifacts allow the Streamlit application to reuse trained models instead of retraining them every time the application starts.

---

## Example

### Input

```text
The product has been a complete disappointment.
It keeps crashing multiple times a day, important
features do not work, and I have lost a lot of time
trying to fix these issues.
```

### Example Output

```text
Classification: 0
Urgency Score: 0.46
Priority: Medium
Recommended Action: Review and assign to support team
```

---

## Project Structure

```text
OmniFeedback-AI/
│
├── app.py
├── omnifeedback_raw_dataset.csv
│
├── tfidf.pkl
├── random_forest.pkl
├── kmeans.pkl
├── bilstm_urgency.pth
│
└── README.md
```

---

## How to Run the Application

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project folder

```bash
cd OmniFeedback-AI
```

### 3. Install the required Python libraries

Install the libraries used by the project, such as:

```bash
pip install streamlit pandas numpy scikit-learn torch transformers nltk
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## Key Learning Outcomes

Through this project, I worked with:

- Python programming
- Data cleaning and preprocessing
- Regex
- Pandas and NumPy
- SQL and SQLite
- Exploratory Data Analysis
- NLP preprocessing
- TF-IDF
- Supervised Machine Learning
- Random Forest
- Unsupervised Machine Learning
- K-Means clustering
- PyTorch
- BiLSTM
- BERT Named Entity Recognition
- Model serialization and reuse
- Streamlit application development
- End-to-end ML/NLP project workflow

---

## Conclusion

OmniFeedback AI demonstrates how multiple data science and AI techniques can be combined into a single customer feedback intelligence application.

The system connects **data processing, database management, machine learning, deep learning, NLP, and interactive visualization** to transform unstructured customer feedback into useful analytical insights and support-priority information.
