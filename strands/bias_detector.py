import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference

def detect_bias(file_path, sensitive_feature="sex", target_column="income"):
    try:
        df = pd.read_csv(file_path)

        # Filter dataset to only valid income values
        df = df[df[target_column].isin(['<=50K', '>50K'])]

        # Drop rows with missing data (if any)
        df = df.dropna()

        # Encode target as binary: 1 for >50K, 0 for <=50K
        y = df[target_column].apply(lambda x: 1 if x == '>50K' else 0)

        # Save sensitive feature column
        sensitive = df[sensitive_feature]

        # Drop target and sensitive feature from features (we don't train on them)
        X = df.drop([target_column, sensitive_feature], axis=1)

        # Encode categorical variables using LabelEncoder (simplest option)
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col])

        # Train/test split
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
            X, y, sensitive, test_size=0.3, random_state=42
        )

        # Train a simple logistic regression model
        model = LogisticRegression(max_iter=10000)
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Fairness evaluation
        group_metrics = MetricFrame(
            metrics={"selection_rate": selection_rate},
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_test
        )

        dp_diff = demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_test
        )

        eo_diff = equalized_odds_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_test
        )

        return {
            "selection_rate_by_group": group_metrics.by_group.to_dict(),
            "overall_selection_rate": group_metrics.overall.item(),
            "demographic_parity_difference": float(dp_diff),
            "equalized_odds_difference": float(eo_diff)
        }

    except Exception as e:
        return {"error": str(e)}