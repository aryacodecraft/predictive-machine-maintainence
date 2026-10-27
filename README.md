# AI-Based Predictive Maintenance

A reproducible college mini-project that predicts the AI4I 2020 dataset's `Machine failure` label using operating measurements and product type. It includes a detailed Jupyter notebook and a Streamlit dashboard. The notebook trains an overall failure classifier and a separate multi-label model for the five failure modes. The model is intended for education and demonstration, not as a certified industrial safety assessment.

## Dataset

The included `data/ai4i2020.csv` is the UCI AI4I 2020 Predictive Maintenance Dataset (10,000 records, 14 columns). The dataset is distributed under CC BY 4.0. Cite and attribute the dataset to its authors using the [UCI dataset record](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). To get a fresh copy, download the archive from UCI, extract `ai4i2020.csv`, and place it in `data/`.

## Setup and run

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The app runs immediately using the included trained models. To retrain them, install the notebook tools and run every cell in `notebooks/Predictive_Maintenance_ML.ipynb`:

```bash
python -m pip install -r requirements-dev.txt
jupyter lab
```

The notebook generates the fitted preprocessing/classification pipeline at `models/best_model.pkl`, the multi-label failure-mode pipeline at `models/failure_mode_model.pkl`, model metadata, test comparison, feature importance, and failure-mode metrics. The checked-in artifacts let the app run immediately without retraining.

Launch the app:

```bash
streamlit run app.py
```

The app uses the same ordered features as the notebook: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`. It does not train at startup. The failure-mode indicators `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded from the overall model to avoid leakage. A separate multi-label model uses those columns as targets, never as inputs, to estimate each mode from the sensor readings. The prediction page shows both the overall failure risk and the five independent mode probabilities.

## Method and evaluation

The notebook compares class-weighted Logistic Regression, Decision Tree, and Random Forest models. It uses stratified train/test splitting, fitted preprocessing pipelines, and five-fold stratified CV for a compact Random Forest grid search. Recall is the search objective because missed failures can be costly; precision, F1, ROC-AUC, confusion matrices, classification reports, and held-out test performance are also reported. A second multi-label model wraps a class-weighted Random Forest in `MultiOutputClassifier`, trains one binary classifier per failure mode, and reports test positives, precision, recall, F1, ROC-AUC, and PR-AUC for each mode. All numerical results are computed when the notebook runs; no metrics are hard-coded.

## GitHub and Streamlit Community Cloud deployment

1. Create an empty GitHub repository, then initialize and push this project from its root:

   ```bash
   git init
   git add .
   git commit -m "Add predictive maintenance app"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   git push -u origin main
   ```

2. In [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub and choose **Create app**.
3. Select the repository and branch, set the main file path to `app.py`, choose Python 3.11 or newer, and deploy. Community Cloud installs packages from `requirements.txt`.
4. Open the assigned `*.streamlit.app` URL in a private/incognito browser session. Check the dashboard, dataset charts, at least one overall prediction, and the five failure-mode estimates.
5. Community Cloud's free apps may sleep after inactivity and can have resource/quota limits; availability for a fixed 24-hour period is not guaranteed. Reopen the URL to wake a sleeping app.

No GitHub repository or hosted app URL is claimed by this project until a real repository is connected and the deployment URL has been verified.

## Viva preparation

- **Logistic Regression:** linear decision boundary over scaled sensor values and one-hot product type; class weighting helps account for rarity.
- **Decision Tree:** rule-based feature splits; interpretable but prone to overfitting.
- **Random Forest:** bagged decision trees with randomized feature selection; averages trees to improve generalization.
- **Failure-mode model:** `MultiOutputClassifier` fits one binary Random Forest per mode, so the five probabilities are independent and need not add to 100%; TWF and RNF have very few positives, and RNF is close to random.
- **Preprocessing:** one-hot encode the product category; standardize numeric inputs in a `ColumnTransformer` fitted only within training folds; `Pipeline` keeps training/inference transforms consistent.
- **Metrics:** recall is the share of true failures detected; precision is the share of failure alerts that are correct; F1 balances these; ROC-AUC measures ranking across thresholds; accuracy can hide missed minority failures.
- **Validation:** stratified holdout gives a final test estimate; stratified CV and grid search tune using training data only.
- **Limitations:** synthetic dataset, rare failures, default probability threshold, possible sensor drift, impurity importance is associative, failure-mode probabilities are less reliable for rare modes, and deployment needs validation/calibration on the specific equipment.

## Syllabus mapping

| ML lab concept | Demonstrated |
| --- | --- |
| Data loading, inspection, missing/duplicate checks | Yes |
| Exploratory plots and correlation analysis | Yes |
| Categorical encoding and numerical scaling | Yes |
| Stratified train/test split and reproducibility | Yes |
| Supervised binary classification | Yes |
| Logistic Regression, Decision Tree, Random Forest | Yes |
| Imbalanced classification and class weighting | Yes |
| Accuracy, precision, recall, F1, ROC-AUC | Yes |
| Confusion matrix, ROC curve, classification report | Yes |
| Stratified cross-validation and grid search | Yes |
| Feature importance and limitations | Yes |
| Saving/loading a complete model pipeline | Yes |
| Deep learning, clustering, time-series forecasting, causal inference | No |

## Repository layout

```text
data/ai4i2020.csv
notebooks/Predictive_Maintenance_ML.ipynb
src/maintenance.py
models/                 # generated fitted model and result tables
app.py
requirements.txt
requirements-dev.txt    # optional notebook and training dependencies
README.md
```
