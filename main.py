from strands.dataset_analyzer import analyze_dataset
from strands.bias_detector import detect_bias
from strands.report_generator import generate_bias_report

print("🔍 Dataset Analysis Result:")
result = analyze_dataset("sample_data.csv")
print(result)

print("\n⚖️ Bias Detection Result:")
bias_result = detect_bias("sample_data.csv", sensitive_feature="sex", target_column="income")
print(bias_result)

print("\n📝 Natural Language Report:")
report = generate_bias_report(bias_result, sensitive_feature="sex", target_label=">50K")
print(report)