import pandas as pd

def analyze_dataset(file_path):
    try:
        df = pd.read_csv(file_path)

        analysis = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
        }

        # Try to detect target column (optional)
        target_col = df.columns[-1]
        if df[target_col].nunique() < 20:  # Likely a classification label
            class_dist = df[target_col].value_counts().to_dict()
            analysis["target_column"] = target_col
            analysis["class_distribution"] = class_dist

        return analysis

    except Exception as e:
        return {"error": str(e)}