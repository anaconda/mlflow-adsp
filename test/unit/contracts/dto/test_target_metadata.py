from mlflow_adsp import TargetMetadata


def test_non_reload_able_uri():
    test_cases: list[dict[str, str]] = [
        {"uri": "models:/registry/Production", "reloadable": True},
        {"uri": "models:/registry/Staging", "reloadable": True},
        {"uri": "models:/registry@winner", "reloadable": True},
        {"uri": "runs:/some_run/model", "reloadable": False},
        {"uri": "./model", "reloadable": False},
        {"uri": "s3://bucket/model", "reloadable": False},
        {"uri": "models:/registry/1", "reloadable": False},
        {"uri": "models://", "reloadable": False},
    ]

    for test_case in test_cases:
        metadata: TargetMetadata = TargetMetadata(model_uri=test_case["uri"])
        print(f"{test_case['uri']}: {test_case['reloadable']}")
        assert metadata.reloadable == test_case["reloadable"]
