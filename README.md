<div align="center">

# 📺 YouTube Viral Video Prediction

### Can engagement data predict the next viral video?

A machine learning project that explores YouTube trending data and predicts whether a video will surpass **1 million views**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1R-mDFXPxyUFlbQrPH-vkE_xvEpGbgFvk?usp=sharing)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)

</div>

---

## 🎬 The Question

Millions of videos are uploaded to YouTube, but only a small percentage gain massive attention.

This project asks:

> **Can early engagement signals help identify which YouTube videos are likely to go viral?**

Using data from trending YouTube videos, I built a complete machine learning pipeline that analyzes engagement behavior and predicts whether a video will reach at least **1,000,000 views**.

---

## 🧠 Project Overview

The project covers the complete data science workflow:

```text
Raw YouTube Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Performance Evaluation
        ↓
Viral Video Predictions
```

The final model uses engagement metrics, publishing behavior, and video characteristics to classify videos as either:

* 🚀 **Viral:** At least 1 million views
* 📼 **Non-viral:** Fewer than 1 million views

---

## 🔍 What I Explored

The exploratory analysis examines questions such as:

* How balanced are viral and non-viral videos?
* What relationship exists between likes and views?
* Which video categories receive the most views?
* Does title length affect video performance?
* How strongly are likes, dislikes, and comments correlated?
* How quickly do videos begin trending after publication?
* Which engagement signals are most useful for predicting virality?

Visualizations include:

* Viral versus non-viral class distribution
* View distribution by viral status
* Likes versus views
* Likes versus dislikes
* Likes versus comments
* Views by video category
* Title length versus views
* Feature correlation heatmap

---

## 🛠️ Feature Engineering

Several additional features were created from the original YouTube data.

| Feature              | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `engagement_rate`    | Combined likes and comments relative to total views    |
| `like_ratio`         | Likes relative to total recorded reactions             |
| `comment_rate`       | Comments relative to total views                       |
| `time_to_trend_days` | Number of days between publishing and trending         |
| `title_length`       | Number of characters in the video title                |
| `is_viral`           | Binary target based on whether views reached 1 million |

The final model was trained using:

```python
features = [
    "likes",
    "dislikes",
    "comment_count",
    "title_length",
    "engagement_rate",
    "like_ratio",
    "comment_rate",
    "time_to_trend_days"
]
```

---

## 🤖 Machine Learning Model

The project uses a **Random Forest Classifier**.

### Model configuration

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
```

Why Random Forest?

* Handles nonlinear relationships
* Works well with numerical features
* Is resistant to overfitting compared with a single decision tree
* Provides feature-importance scores
* Can model interactions between multiple engagement signals
* Supports class weighting for imbalanced datasets

The dataset was divided into:

* **80% training data**
* **20% testing data**

---

## 📊 Model Performance

| Metric                            |    Result |
| --------------------------------- | --------: |
| Overall accuracy                  | **87.0%** |
| Viral-video recall                |   **84%** |
| Viral-video precision             |   **11%** |
| Viral videos correctly identified |   **126** |
| Viral videos missed               |    **24** |

### Confusion Matrix

```text
                         Predicted
                     Non-Viral   Viral

Actual Non-Viral       6,234      930
Actual Viral              24      126
```

### What These Results Mean

The model successfully identified approximately **84% of the videos that were actually viral**.

However, viral-video precision was considerably lower. This means the model also labeled many non-viral videos as viral.

That tradeoff is important because the dataset contains far more non-viral videos than viral videos. Overall accuracy alone would therefore provide an incomplete picture of model performance.

For a real-world recommendation or marketing system, the next priority would be improving precision while maintaining strong recall.

---

## 🏆 Most Important Predictors

The Random Forest model found that the strongest predictors included:

1. 👍 **Likes**
2. 📈 **Engagement rate**
3. 💬 **Comment activity**
4. 👎 **Dislikes and reaction behavior**
5. ⏱️ **Time required to begin trending**

One of the project's biggest findings is that virality is not explained by a single metric.

Instead, it emerges from a combination of audience interaction, engagement intensity, and how quickly a video gains attention.

---

## 💡 Key Takeaways

* High engagement is one of the clearest indicators of viral potential.
* Likes were more influential than basic video characteristics such as title length.
* Comment activity provides useful information about audience involvement.
* Viral-video prediction is a heavily imbalanced classification problem.
* Accuracy must be evaluated alongside precision, recall, and the confusion matrix.
* More advanced text features could improve predictions beyond numerical engagement data.

---

## 📁 Repository Structure

```text
Youtube-Trending-Data-Analysis/
│
├── Nas_P_YouTube_Viral_Video_Prediction_Project.ipynb
└── README.md
```

---

## 🚀 Running the Project

### Option 1: Google Colab

Click the badge below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1R-mDFXPxyUFlbQrPH-vkE_xvEpGbgFvk?usp=sharing)

Then:

1. Upload `USvideos.csv` when required.
2. Select **Runtime → Run all**.
3. Review the visualizations, model results, and predictions.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/PullenN9163/Youtube-Trending-Data-Analysis.git
cd Youtube-Trending-Data-Analysis
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Nas_P_YouTube_Viral_Video_Prediction_Project.ipynb
```

Place `USvideos.csv` in the expected location or update the notebook's dataset path.

---

## 🔮 Future Improvements

Potential next steps include:

* [ ] Use precision-recall curves to select a better classification threshold
* [ ] Compare Random Forest with XGBoost, LightGBM, and logistic regression
* [ ] Apply SMOTE or alternative resampling techniques
* [ ] Perform hyperparameter tuning with GridSearchCV
* [ ] Add cross-validation
* [ ] Analyze tags, titles, and descriptions with natural language processing
* [ ] Include channel popularity and subscriber information
* [ ] Analyze thumbnail characteristics using computer vision
* [ ] Predict expected view counts using regression
* [ ] Deploy the model as an interactive web application

---

## ⚠️ Limitations

* The dataset contains videos that already appeared on YouTube's trending list.
* Viral videos are much less common than non-viral videos.
* Engagement metrics may already reflect some level of popularity.
* The project does not currently analyze thumbnails, tags, descriptions, or channel history.
* YouTube's recommendation algorithm and user behavior change over time.
* Model performance on newer or non-trending videos may differ.

---

## 📚 Dataset

This project uses the `USvideos.csv` dataset containing information about trending YouTube videos in the United States.

Before publishing the repository, add the original dataset source here:

```text
Dataset source: YOUR_DATASET_SOURCE_LINK
```

The dataset itself may be excluded from the repository depending on its size and licensing requirements.

---

## 👨🏽‍💻 Author

**Nas P.**

Data analyst and developer interested in machine learning, data platforms, automation, and building practical systems from real-world data.

[![GitHub](https://img.shields.io/badge/GitHub-PullenN9163-181717?style=for-the-badge\&logo=github)](https://github.com/PullenN9163)

---

<div align="center">

### From clicks and comments to machine learning predictions. 📺📊🤖

⭐ Star the repository if you found the project interesting.

</div>
