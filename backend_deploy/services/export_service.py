import io
import json
import pandas as pd


def generate_json_export(audit_data: dict):
    return json.dumps(audit_data, indent=2)


def generate_csv_export(audit_data: dict):

    rows = []

    metrics = audit_data.get("metrics", {})

    for key, value in metrics.items():

        if isinstance(value, dict):

            for sub_key, sub_val in value.items():
                rows.append({
                    "metric": f"{key}.{sub_key}",
                    "value": sub_val
                })

        else:
            rows.append({
                "metric": key,
                "value": value
            })

    df = pd.DataFrame(rows)

    output = io.StringIO()
    df.to_csv(output, index=False)

    output.seek(0)

    return output