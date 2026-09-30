# Customer Churn Prediction

Predicts which telecom customers are likely to cancel, explains **why** for each customer with SHAP, and chooses the decision threshold from business cost instead of accuracy. Includes a Streamlit app that scores a single customer and shows the reasons behind the score.

**Stack:** Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit

## The problem

Keeping a customer is much cheaper than winning a new one, so the goal is to flag likely churners early enough for the retention team to act. The two possible mistakes don't cost the same:

- **Missed churner** (false negative): the customer leaves. Expensive.
- **False alarm** (false positive): a loyal customer gets an offer they didn't need. Cheap.

So the main metric is **recall on the churn class**, not accuracy. With 26.5% of customers churning, a model that predicts "nobody churns" would already be 73.5% accurate while catching no one.

## Results

Test set of 1,409 customers (374 churners), stratified 80/20 split.

| Model | Threshold | Accuracy | Precision (churn) | Recall (churn) | Missed churners |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.5 | 0.807 | 0.658 | 0.567 | 162 |
| XGBoost, default | 0.5 | 0.776 | 0.589 | 0.521 | 179 |
| XGBoost + `scale_pos_weight` | 0.5 | 0.757 | 0.534 | 0.671 | 123 |
| **XGBoost + `scale_pos_weight`** | **0.30** | **0.734** | **0.499** | **0.805** | **73** |

**ROC-AUC: 0.818**

- Default XGBoost did *worse* than the baseline on recall: a more powerful model doesn't fix class imbalance on its own. Weighting the churn class (`scale_pos_weight`) did.
- The threshold was chosen by assigning costs (₹5,000 per lost customer, ₹500 per wasted offer, both assumptions). The cost-minimizing threshold (0.03) would flag about 71% of all customers, which isn't actionable, so 0.30 was chosen as a balance: it catches about 80% of churners and flags about 43% of customers.

## Key findings

- **Contract type is the strongest driver.** Month-to-month customers churn at 42.7%, two-year customers at 2.8%.
- **Risk factors compound.** Month-to-month customers paying by electronic check churn at 53.7%, higher than either factor alone.
- **Fiber optic customers churn at 41.9%** (vs 19.0% for DSL). SHAP flagged this before it was checked in the raw data.
- **SHAP's top drivers** (contract, tenure, monthly charges) match what the exploratory analysis suggested.

## The app

Enter a customer's details to get a churn risk score, a HIGH/LOW decision at the 0.30 threshold, and a SHAP breakdown of what's pushing the score up or down.

The app also:
- Rebuilds exactly the 30 columns the model was trained on (`get_dummies` + `reindex` to the saved column list) and was tested against known customers from the notebook to rule out training-serving skew.
- Auto-corrects dependent fields (e.g. no phone service → "No phone service" for multiple lines) and tells the user what it changed.
- Refuses to score combinations that never appear in the training data (e.g. no phone service with fiber optic internet).

## Project structure

```
├── churn_analysis.ipynb   # full analysis: cleaning, EDA, models, threshold, SHAP
├── app.py                 # Streamlit app
├── churn_model.joblib     # trained model + column list + threshold + dropdown options
├── requirements.txt
└── data/                  # dataset goes here (not included in the repo)
```

## How to run

Built with Python 3.14.

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app (uses the saved `churn_model.joblib`, so the dataset isn't needed):
   ```bash
   python -m streamlit run app.py
   ```
3. To re-run the analysis, download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), put `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/`, install Jupyter (`pip install notebook`), and run `churn_analysis.ipynb` from top to bottom. The last section saves a fresh `churn_model.joblib`.

## Limitations

- **Scores aren't calibrated probabilities.** `scale_pos_weight` inflates them; they rank customers well but overstate the chance of churn.
- **SHAP explains the model, not cause and effect.** Whether an offer actually prevents churn would need an A/B test.
- **Single train/test split** and **default hyperparameters** (apart from `scale_pos_weight`). Cross-validation and tuning are natural next steps.
- **The costs are assumptions.** The right threshold depends on real costs and the retention team's capacity.

## Data

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): IBM sample dataset, 7,043 customers, 21 columns.
