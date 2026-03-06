# NLP-Tract-Task
NLP Task Portfolio — News &amp; Text Intelligence A collection of end-to-end Natural Language Processing projects built with Python, covering the full ML pipeline from raw data to evaluation. Each task uses real-world datasets loaded directly from Hugging Face and is implemented in a clean and well structured manner


Here's a comprehensive README for your repo:



---

##  Table of Contents
- [Task 2 — News Category Classification](#task-2)
- [Task 3 — Fake News Detection](#task-3)
- [Task 4 — Named Entity Recognition](#task-4)
- [Task 6 — Question Answering with Transformers](#task-6)
- [Task 7 — Text Summarization](#task-7)
- [Task 8 — Resume Screening](#task-8)
- [Tech Stack](#tech-stack)
- [Setup](#setup)

---

<a name="task-2"></a>
##  Task 2 — News Category Classification

### Goal
Automatically classify news articles into one of four categories: **World, Sports, Business, or Sci/Tech**.

### Dataset
**AG News** — 120,000 training articles and 7,600 test articles loaded via Hugging Face (`datasets` library). Each article has a title and description.

### Approach
- **Preprocessing** — Custom `TextPreprocessor` class handles lowercasing, URL/HTML removal, tokenization, stopword removal, and lemmatization using NLTK
- **Feature Engineering** - TF-IDF vectorization with unigrams + bigrams (50,000 features, sublinear TF scaling)
- **Models Trained** - Logistic Regression, Linear SVM, Random Forest, XGBoost, LightGBM
- **Evaluation** - Accuracy, classification report, confusion matrix

### Key Results
- All 5 models trained and compared on the same test set
- Best models (Logistic Regression / Linear SVM) typically achieve **90%+ accuracy** on AG News
- Full model leaderboard ranked by accuracy and training time

### Bonus
- Top 20 most frequent words per category (bar charts)
- Word clouds for each category
- `predict_category()` inference function for classifying any new article

### Libraries
`scikit-learn` · `pandas` · `nltk` · `xgboost` · `lightgbm` · `wordcloud` · `datasets`

---

<a name="task-3"></a>
##  Task 3 — Fake News Detection

### Goal
Build a binary classifier to detect whether a news article is **Real or Fake** based on its text content.

### Dataset
**Fake and Real News Dataset** — loaded from Hugging Face (`ErfanMoosaviMonazzah/fake-news-detection-dataset-English`). Contains ~44,000 articles with labels: `0 = Fake`, `1 = Real`. Pre-split into train/validation/test.

### Approach
- **Preprocessing** — Combined title + text, then applied: lowercase → URL removal → punctuation stripping → stopword removal → lemmatization
- **Feature Engineering** — TF-IDF vectorization with unigrams + bigrams (30,000 features)
- **Models Trained** — Logistic Regression and Linear SVM
- **Evaluation** — Accuracy and F1-Score (both reported and visualized)

### Key Results
- Both models achieve very high accuracy (~98%+) on this dataset
- Confusion matrix shows where the model makes mistakes
- Side-by-side Accuracy and F1 bar chart for both models

### Bonus
- Side-by-side **Word Clouds** — Fake news (red palette) vs Real news (green palette)
- Simple `predict()` function to classify any new article in one line

### Libraries
`scikit-learn` · `pandas` · `nltk` · `wordcloud` · `datasets`

---

<a name="task-4"></a>
##  Task 4 — Named Entity Recognition (NER) from News Articles

### Goal
Identify and categorize named entities — **People (PER), Organizations (ORG), Locations (LOC), and Miscellaneous (MISC)** — from real news article text.

### Dataset
**CoNLL-2003** — The standard benchmark dataset for NER, in BIO (Beginning-Inside-Outside) sequence tagging format. Loaded from local files (`train.txt`, `valid.txt`, `test.txt`) — 219,554 training tokens across 14,041 sentences.

### Approach
- **CoNLL Parser** — Custom `parse_conll()` function reads BIO-tagged files and converts them to `(word, tag)` pairs
- **Entity Extraction** — `extract_entities()` converts BIO tags into clean `(entity_text, entity_type)` pairs
- **Rule-Based NER** — spaCy `en_core_web_sm` (statistical model with lookup tables)
- **Model-Based NER** — spaCy `en_core_web_lg` (300-dimensional GloVe word vectors)

### Key Results
- Both models run on 500 test sentences and results compared side by side
- Entity type distribution charts (PERSON vs ORG vs GPE vs LOC)
- Top most-mentioned entities per type (bar charts)

### Bonus
- **displaCy visualization** — coloured inline entity highlighting rendered directly in the notebook
- Direct sm vs lg model comparison on the same sentence
- Entities exported to a clean pandas DataFrame

### Libraries
`spacy` · `pandas` · `matplotlib` · `seaborn`

---

<a name="task-6"></a>
##  Task 6 — Question Answering with Transformers

### Goal
Build an **extractive QA system** that reads a passage of text and extracts the exact answer span to a given question — no generation, just finding where the answer is in the context.

### Dataset
**SQuAD v1.1** (Stanford Question Answering Dataset) — loaded from Hugging Face (`squad`). Contains 87,599 training and 10,570 validation question-context-answer triples from Wikipedia passages.

### Approach
- **Model Architecture** — Encoder-only transformers fine-tuned for span extraction. The model predicts `start` and `end` token positions of the answer within the context
- **Models Used:**
  - `distilbert-base-cased-distilled-squad` — Lightweight, fast
  - `deepset/roberta-base-squad2` — More accurate
  - `deepset/bert-base-cased-squad2` — Classic BERT baseline
- **Implementation** — Used `AutoTokenizer` + `AutoModelForQuestionAnswering` directly (compatible with all transformers versions)
- **Evaluation** — Exact Match (EM) and token-level F1 score on 200 validation samples

### Key Results
- All 3 models evaluated and ranked by EM and F1
- Inference speed (seconds/sample) compared
- Per-example prediction DataFrame showing question, ground truth, prediction, and scores
- F1 score distribution histogram

### Bonus
- `ask()` function — paste any context and question to get answers from all 3 models instantly
- Streamlit web app (`app.py`) with model selector and confidence display

### Libraries
`transformers` · `datasets` · `torch` · `pandas` · `matplotlib`

---

<a name="task-7"></a>
##  Task 7 — Text Summarization Using Pre-trained Models

### Goal
Automatically generate **concise, readable summaries** of long news articles using pre-trained encoder-decoder transformer models, and evaluate summary quality with ROUGE scores.

### Dataset
**CNN-DailyMail 3.0.0** — loaded from Hugging Face (`cnn_dailymail`). Contains 287,113 news articles paired with human-written highlights (reference summaries). Average article length ~800 words, average summary ~55 words.

### Approach

**Abstractive Summarization** (generates new sentences):
- `t5-small` — Text-to-Text Transfer Transformer, requires `summarize:` prefix
- `facebook/bart-large-cnn` — BART large, fine-tuned specifically on CNN-DailyMail
- `facebook/bart-base` — Lighter BART variant for faster inference

**Extractive Summarization** (selects existing sentences):
- **TextRank** via `sumy` — Graph-based ranking algorithm, no model needed

- **Implementation** — Used `AutoTokenizer` + `AutoModelForSeq2SeqLM` directly (avoids broken `summarization` pipeline in newer transformers versions)
- **Evaluation** — ROUGE-1, ROUGE-2, ROUGE-L on 50 test articles

### Key Results
- Full ROUGE leaderboard comparing all 4 approaches
- Individual ROUGE metric bar charts + grouped comparison chart
- Abstractive vs Extractive final comparison chart
- Side-by-side summary output for 3 articles across all models

### What ROUGE Measures
| Metric | What it checks |
|---|---|
| ROUGE-1 | Unigram (single word) overlap |
| ROUGE-2 | Bigram (two word) overlap |
| ROUGE-L | Longest common subsequence |

### Libraries
`transformers` · `datasets` · `rouge-score` · `sumy` · `torch` · `pandas` · `matplotlib`

---

<a name="task-8"></a>
##  Task 8 — Resume Screening Using NLP

### Goal
Build an intelligent resume screening system that **ranks resumes against a job description** using semantic similarity — going beyond keyword matching to understand meaning.

### Datasets
- **Resumes** — `InferencePrince555/Resume-Dataset` (Hugging Face) — 32,500 resumes across 52 job categories
- **Job Descriptions** — `jacob-hugging-face/job-descriptions` (Hugging Face) — real job postings with titles and full descriptions

### Approach
- **Embeddings** - `all-MiniLM-L6-v2` Sentence Transformer converts both resumes and job descriptions into 384-dimensional semantic vectors
- **Matching** - Cosine similarity between the job embedding and all resume embeddings
- **Ranking** - Resumes sorted from highest to lowest similarity score with match labels:
  - 🟢 `>= 0.70` Strong Match
  - 🟡 `>= 0.50` Good Match
  - 🟠 `>= 0.35` Partial Match
  - 🔴 `< 0.35` Weak Match

### Why Semantic Embeddings?
Traditional keyword matching fails when a resume says *"Python developer with ML experience"* and a job asks for a *"Machine Learning Engineer"*. Sentence Transformers understand that these mean the same thing by encoding context and meaning — not just exact words.

### Key Features
- Resumes encoded **once** and reused across all job descriptions (efficient)
- `screen_resumes(job_description, top_n=10)` - returns ranked DataFrame with scores and justifications
- Batch mode — screen multiple jobs at once
- **Similarity heatmap** - jobs × resume categories matrix showing which resume types match which jobs best

### Bonus
- **Named Entity Extraction** — spaCy NER pulls out organisations and locations from top resumes; keyword matching identifies technical skills (Python, SQL, TensorFlow, etc.)
- **`match_single_resume()`** - paste your own resume and a job description to get an instant match score and skill analysis

### Libraries
`sentence-transformers` · `scikit-learn` · `spacy` · `datasets` · `pandas` · `matplotlib` · `seaborn`

---

<a name="tech-stack"></a>
##  Tech Stack

| Category | Libraries |
|---|---|
| **Data** | `pandas`, `numpy`, `datasets` (Hugging Face) |
| **NLP / Text** | `nltk`, `spacy`, `sentence-transformers` |
| **ML Models** | `scikit-learn`, `xgboost`, `lightgbm` |
| **Deep Learning** | `transformers` (Hugging Face), `torch` |
| **Evaluation** | `rouge-score`, `sklearn.metrics` |
| **Visualization** | `matplotlib`, `seaborn`, `wordcloud`, `spacy displacy` |
| **Extractive NLP** | `sumy` (TextRank) |

---

<a name="setup"></a>
## ⚙️ Setup

### Requirements
- Python 3.9+
- Anaconda (recommended) or any virtual environment
- Jupyter Notebook or JupyterLab
<img width="2685" height="1508" alt="wordclouds" src="https://github.com/user-attachments/assets/dc78d046-5ed9-491c-802a-6e962c305bb3" />

### Install all dependencies
```bash
pip install transformers datasets sentence-transformers torch
pip install scikit-learn xgboost lightgbm pandas numpy
pip install nltk spacy rouge-score wordcloud sumy matplotlib seaborn
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

### Known Issues & Fixes

**`Unknown task summarization` or `Unknown task question-answering`**
> Newer versions of `transformers` (v4.41+) removed these pipeline tasks.
> All notebooks in this repo use `AutoTokenizer` + task-specific model classes directly - no pipeline needed.

**`SentencePiece` error with Pegasus**
> Pegasus tokenizer requires `sentencepiece` + `protobuf`. If you encounter this error, either install both packages and restart your kernel, or use `facebook/bart-base` as a drop-in replacement (already done in this repo).

**Rust / tokenizers build error on Windows**
> Do not pin `transformers` to older versions - this triggers a Rust compilation of `tokenizers==0.19`. Use the latest transformers version with the direct model loading approach used in this repo.

---

## 📁 Project Structure
```
 nlp-task-portfolio/
 ┣ 📓 task2_news_category_classification.ipynb
 ┣ 📓 task3_fake_news_detection.ipynb
 ┣ 📓 task4_ner_news_articles.ipynb
 ┣ 📓 task6_question_answering_transformers.ipynb
 ┣ 📓 task7_text_summarization.ipynb
 ┣ 📓 task8_resume_screening_nlp.ipynb
 ┣ 📄 README.md
 ┗ 📄 train.txt / valid.txt / test.txt  <- CoNLL-2003 (Task 4)
```

---

> **Note:** All datasets except CoNLL-2003 are loaded automatically from Hugging Face . no manual downloads required. For Task 4, place the CoNLL-2003 files in the same directory as the notebook before running.
<img width="1891" height="735" alt="tag_distribution" src="https://github.com/user-attachments/assets/25f7c7cc-9c50-4cbd-bdd8-60261bd42cdb" />

<img width="1785" height="737" alt="model_comparison" src="https://github.com/user-attachments/assets/f72311f8-e07f-4b8a-a78e-c60e4238d28b" />
<img width="1065" height="887" alt="confusion_matrix" src="https://github.com/user-attachments/assets/a42625bb-8119-4559-8d85-19edee9cc7c0" />

