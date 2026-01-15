# Fake/Real News Classifier with Natural Language Processing - Accenture

---

### 👥 **Team Members**

| Name               | GitHub Handle     | Contribution                                                                 |
|--------------------|-------------------|------------------------------------------------------------------------------|
| Alexandru Soroiu   | @alexandrusoroiu  | Feature engineering, SVM model design and evaluation, project documentation  |
| Grace Madison Wu   | @gracemadisonwu   | Data collection, exploratory data analysis (EDA), dataset documentation      |
| Aniekan Inyang     | @aniekanai        | Data preprocessing, Exploratory Data Analysis (EDA) Naive Bayes, and LSTM model development            |
| Karina Hernandez   | @khern2005        | Model selection, hyperparameter tuning, training classical ML baselines      |
| Arif Manawer       | @arifmanawer      | LSTM model development, deep learning experiments, performance analysis      |
| Sumaiya Chowdhury  | @sumaiyachow3     | BERT model experimentation, Random forest development, evaluation, results interpretation               |

---

## 🎯 **Project Highlights**

- Built a fake vs real news classifier using `[multiple NLP approaches]`, including `[TF‑IDF]` + `[linear SVM]`, `[LSTM]`, and a fine‑tuned `[BERT]` sequence classification model on news titles and article bodies.
- After removing data leakage from the original “subject” feature, the best TF‑IDF + linear SVM model on title and text reached about `[99.6%]` accuracy with near‑perfect precision, recall, and ROC‑AUC on a held‑out test set.
- A deep LSTM model on combined title and text achieved roughly `[99.3%]` test accuracy, correctly classifying `[8,919]` out of `[8,980]` articles and showing stable learning across epochs. 
- Engineered interpretable features such as `[punctuation density]`, `[text length]`, `[seasonality]` (year, month, weekday), and `[contraction usage]`, and used these insights to diagnose and reduce data leakage from tokens like `[“Reuters”]` and `[“said”]`, making the models more robust for an enterprise context at `[Accenture]`.

---

## 👩🏽‍💻 **Setup and Installation**

### 1. Clone the repository

git clone https://github.com/AlexandruSoroiu/accenture-ai-studio-fall2025.git
cd accenture-ai-studio-fall2025

### 2. Create and activate a virtual environment (optional but recommended)

python -m venv .venv
source .venv/bin/activate # on macOS/Linux

.venv\Scripts\activate # on Windows

### 3. Install dependencies

pip install -r requirements.txt

Key libraries include `[pandas]`, `[numpy]`, `[scikit-learn]`, `[matplotlib]`, `[seaborn]`, `[nltk]`, `[tensorflow/keras]`, `[transformers]`, `[datasets]`, and `[dateparser]`. 

### 4. Download and place the dataset

This project uses the “Fake” and “True” news CSV files commonly distributed as `Fake.csv` and `True.csv`, each containing article `[title]`, `[text]`, `[subject]`, and `[date]` columns.

1. Download `Fake.csv` and `True.csv` from the original source (`[https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data]`).  
2. Place them in a `[data/]` folder at the root of this repo:

data/
Fake.csv
True.csv

3. In the notebook or config, ensure the paths point to this folder rather than Google Drive (for example: `[data/Fake.csv]`, `[data/True.csv]`). 

### 5. Run the notebook(s) or scripts

- Open the main Jupyter notebook (`notebooks/Accenture1e.ipynb`) and run all cells in order.

For BERT training, a GPU is strongly recommended (e.g., Google Colab with GPU runtime).

---

## 🏗️ **Project Overview**

This project was completed as part of the Break Through Tech @ Cornell Tech Fall AI Studio program in collaboration with Accenture. The goal was to design and evaluate machine learning models that can automatically classify online news articles as fake or real, focusing on scalable, explainable, and production‑oriented NLP solutions.  

Our team worked within Accenture’s problem context where misinformation impacts brand trust, client decision‑making, and risk management. The objective was to explore classical and deep learning approaches, compare their performance, and identify modeling choices that reduce leakage and overfitting so that results are more trustworthy in real‑world deployment. 

Detecting fake news at scale is increasingly important for technology and consulting companies that support media, finance, and public‑sector clients. Robust classifiers can serve as building blocks for content moderation tools, early‑warning systems, and human‑in‑the‑loop review workflows, helping stakeholders prioritize high‑risk articles and maintain information quality.

---

## 📊 **Data Exploration**

The dataset combines two CSV files, `Fake.csv` and `True.csv`, which were merged after labeling fake articles as 0 and true articles as 1. Each file includes columns for article title, full text, subject, and a date string; the combined, deduplicated dataset contains around `[35,763]` articles, with roughly `[59%]` true and `[41%]` fake.

During EDA, our team inspected dataset shapes, unique value counts, missing values, and subject distributions, and parsed the free‑form date strings into usable `[datetime]` objects using a custom normalization function plus `[dateparser]` as a fallback. From the parsed dates, additional features such as year, monthly period, and day‑of‑week were created to analyze seasonal patterns in fake vs real articles. 

EDA revealed:
- Most true articles cluster in `[2016–2017]`, reflecting the original collection period, so date patterns mainly reflect dataset design rather than a generalizable signal.
- There are typically more fake than true articles in many months, with spikes that likely reflect collection bias.
- Fake news tends to appear more frequently on weekends compared with true news, while weekdays show a more balanced mix.

Additional exploratory work examined distributions of title and text lengths, top frequent tokens for fake vs true sets, the heavy use of `[“Reuters”]` in true articles, and the presence of contractions and punctuation. Word frequency analyses and histograms of punctuation counts (such as exclamation points and question marks per title or per word) suggested stylistic signals that differ between fake and real content.

---

## 🧠 **Model Development**

The project followed a structured pipeline:

1. **Preprocessing and cleaning**  
   - Combined the Fake and True datasets, added binary labels, and dropped the `[subject]` column after discovering it caused serious data leakage (certain subjects almost perfectly indicated the label).
   - Parsed and cleaned the `[date]` field, created seasonality features, and removed boilerplate like “(Reuters)” and wire‑style lead‑ins from the text.
   - Removed English stopwords, lemmatized tokens using NLTK’s `[WordNetLemmatizer]`, and built cleaned title and text columns; additional numeric features captured text and title length, punctuation counts, and contractions per article.

2. **Classical frequency‑based models**  
   - Bag‑of‑words and TF‑IDF representations were built for titles, texts, and combined text, with vocabulary caps and `[min_df`/`max_df]` thresholds to drop extremely rare or overly common terms. 
   - Trained a series of baseline models, including Multinomial Naive Bayes, logistic regression, random forests, and an SVM using both count and TF‑IDF features.
   - For the primary SVM, a `[ColumnTransformer]` applied separate TF‑IDF vectorizers to the title and text, feeding a `[LinearSVC]` with class balancing and stratified train/validation/test splits to avoid leakage.

3. **Embedding‑based and deep models**  
   - Constructed averaged Word2Vec document embeddings using pre‑trained Google News word vectors and used them as features for an SVM classifier.
   - Built an LSTM model over tokenized combined title + text sequences using Keras: an embedding layer, an LSTM layer with dropout, and dense layers for binary classification, trained with early stopping on validation loss.
   - Fine‑tuned a BERT‑base model (`[bert-base-uncased]`) on constructed inputs such as “Title: … [SEP] Body: …”, using Hugging Face’s `[Trainer]` API, class‑balanced loss weights, and early stopping, with train/validation/test splits stored as a `[datasets.DatasetDict]`.

Hyperparameters such as n‑gram ranges, vocabulary sizes, regularization strength, number of estimators (for random forests), LSTM hidden units, sequence length, and BERT learning rate were iteratively tuned based on validation metrics. 

---

## 📈 **Results & Key Findings**

Across models, performance was evaluated using accuracy, precision, recall, F1 score, confusion matrices, and ROC‑AUC on held‑out validation and test sets. The best TF‑IDF + linear SVM configuration, trained only on title and text, obtained about 99.6% test accuracy with a very strong ROC‑AUC, demonstrating that a well‑regularized linear model with n‑gram representations can nearly match deep models for this task.

The LSTM model on combined title and text sequences reached approximately 99.3% test accuracy, misclassifying only 61 out of 8,980 articles (41 false positives and 20 false negatives), with 99-100% precision and recall for both classes. This indicates that the model is not just memorizing the training data but has learned stable patterns in news language.

Logistic regression with TF‑IDF also produced very high accuracy, initially raising concerns about data leakage from terms highly associated with specific sources (e.g., “Reuters”, “said”). Experiments that explicitly removed such tokens from the text led to a small drop in accuracy (for example from about 0.987 to about 0.984) and slightly more fake articles misclassified, which supports the conclusion that those tokens were partially leaking dataset‑specific source information rather than reflecting intrinsic truthfulness.

Random forest models trained on TF‑IDF features reached strong accuracy but did not significantly outperform simpler linear models when controlling for leakage. BERT provided competitive performance with high accuracy and F1, while also supporting rich token‑level representations; however, given the structured leakage risk and dataset bias, simpler models remained attractive for interpretability and deployment cost.

Fairness in the sense of user‑ or group‑level bias could not be fully assessed because the dataset does not contain demographic attributes. Instead, the team focused on fairness‑adjacent issues like class balance, data collection bias over time, and topic‑level leakage from `[subject]`, “Reuters”, and similar tokens, and then adjusted the pipeline to reduce these artifacts.

---

## 🚀 **Next Steps**

With more time and resources, several directions could strengthen the project:

- **Broader, fresher data**: Incorporate additional news sources, languages, and time periods beyond 2016-2017 to reduce dataset bias and evaluate robustness under distribution shift.  
- **Robustness and adversarial evaluation**: Test models against paraphrased, slightly edited, or adversarially perturbed articles to understand how easily fake content could evade detection.  
- **Fairness and topic analysis**: If enriched metadata becomes available (e.g., outlet, geography, topic), analyze performance across subgroups to ensure the classifier does not systematically over‑ or under‑flag specific topics or publishers.  
- **Deployment‑oriented work**: Package the best model (e.g., SVM, LSTM, or BERT) behind a simple API or web interface, add calibration for predicted probabilities, and integrate logging/monitoring hooks suitable for a production scenario at `[Accenture]`. 

---

## 📝 **License**

This project is licensed under the MIT License.

---

## 🙏 **Acknowledgements**

The team would like to thank `[Break Through Tech AI]`, `[Accenture]`, Challenge Advisor **Isabel Heard**, Coach **Jenna Hunte**, the `[BTT instructional team]`, and our `[mentors]` for guidance, technical feedback, and continuous support throughout this project.
