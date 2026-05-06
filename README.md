🚢 Titanic Survival Prediction
This project is a simple machine learning model built to predict whether a passenger survived the Titanic disaster or not. It’s based on the classic Kaggle dataset and focuses on understanding how different factors like age, gender, ticket class, etc. influenced survival.
📌 What this project does
Cleans and preprocesses the Titanic dataset
Handles missing values and encodes categorical data
Trains a classification model (Logistic Regression)
Evaluates performance using accuracy, confusion matrix, and classification report
🧠 Features used
Some of the main features considered:
Passenger class (Pclass)
Sex
Age
Fare
Number of siblings/spouses (SibSp)
Number of parents/children (Parch)
⚙️ Tech stack
Python
Pandas, NumPy
Scikit-learn
Matplotlib / Seaborn (for basic visualization)
🚀 How to run
Clone the repository
Install dependencies:
pip install -r requirements.txt
Run the script:
python main.py
📊 Model performance
The model achieves a decent accuracy (around ~70–80%, depending on preprocessing).
This isn’t meant to be perfect — the goal was to understand the workflow end-to-end.
📁 Project structure
Titanic-Survival-prediction/
│
├── data/     
├── main.py main script
├── titanic-dataset-prediction.pynb
├── requirements.txt
├── titanic.csv
└── README.md
📝 Notes
This project is more about learning than optimization
There’s definitely room for improvement (feature engineering, tuning, trying other models like Random Forest, etc.)
📎 Final thoughts
I built this to get comfortable with the basics of machine learning — loading data, cleaning it, training a model, and evaluating results.
If you have suggestions or improvements, feel free to fork or open an issue.
