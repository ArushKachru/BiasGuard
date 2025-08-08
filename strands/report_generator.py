def generate_bias_report(metrics: dict, sensitive_feature="sex", target_label=">50K"):
    try:
        rates = metrics["selection_rate_by_group"]["selection_rate"]
        overall_rate = metrics["overall_selection_rate"]
        dp = metrics["demographic_parity_difference"]
        eo = metrics["equalized_odds_difference"]

        group_names = list(rates.keys())
        rate_descriptions = [
            f"{group}: {rate * 100:.2f}%"
            for group, rate in rates.items()
        ]
        rate_summary = " | ".join(rate_descriptions)

        report = (
            f"📊 The model predicts '{target_label}' at the following rates by {sensitive_feature}:\n"
            f"    {rate_summary}\n\n"
            f"⚖️ Overall selection rate: {overall_rate * 100:.2f}%\n"
            f"📉 Demographic parity difference: {dp:.2f} "
            f"({'⚠️ Biased' if abs(dp) > 0.1 else '✅ Acceptable'})\n"
            f"📉 Equalized odds difference: {eo:.2f} "
            f"({'⚠️ Biased' if abs(eo) > 0.1 else '✅ Acceptable'})\n\n"
            f"🧠 Interpretation: The model may be favoring certain groups when predicting '{target_label}'. "
            f"A demographic parity gap of >10% and an equalized odds difference near or above 0.2 suggest measurable bias."
        )

        return report

    except Exception as e:
        return f"❌ Error generating report: {str(e)}"