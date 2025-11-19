    """
    Linear Probe LSTM용 Dataset.
    JSON 구조:
    {
        "data": [
            {
                "sequence_id": "tidy_s14",
                "windows": [
                    {"sensor_path": "trim_2s_sensor/tidy/...csv", "class_name": "tidy", ...},
                    ...
                ]
            }
        ]
    }
    """