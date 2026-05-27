"""Tests for data/registry.py — pure data, no external dependencies."""

import pytest

from molgate.data.registry import DATASET_REGISTRY, DatasetInfo, get_dataset_info, list_datasets


class TestDatasetInfo:
    """Tests for the DatasetInfo dataclass."""

    def test_frozen(self):
        """DatasetInfo instances should be immutable."""
        info = get_dataset_info("solubility")
        with pytest.raises(AttributeError):
            info.tdc_name = "something_else"

    def test_fields(self):
        """DatasetInfo should have all required fields."""
        info = get_dataset_info("solubility")
        assert isinstance(info.tdc_name, str)
        assert isinstance(info.tdc_group, str)
        assert isinstance(info.task_type, str)
        assert isinstance(info.target_column, str)
        assert isinstance(info.metric, str)
        assert isinstance(info.description, str)


class TestRegistry:
    """Tests for the registry dictionary and lookup functions."""

    def test_registry_has_expected_datasets(self):
        """We should have all TDC ADMET datasets registered."""
        assert len(DATASET_REGISTRY) == 23

    def test_list_datasets_sorted(self):
        """list_datasets should return sorted names."""
        names = list_datasets()
        assert names == sorted(names)
        assert len(names) == 23

    def test_get_dataset_info_valid(self):
        """Looking up a valid name should return a DatasetInfo."""
        info = get_dataset_info("solubility")
        assert isinstance(info, DatasetInfo)
        assert info.tdc_name == "Solubility_AqSolDB"
        assert info.task_type == "regression"
        assert info.metric == "rmse"

    def test_get_dataset_info_invalid(self):
        """Looking up an invalid name should raise KeyError with helpful message."""
        with pytest.raises(KeyError, match="Unknown dataset 'nonexistent'"):
            get_dataset_info("nonexistent")

    def test_task_types_valid(self):
        """All datasets should have either 'regression' or 'classification' task type."""
        for name, info in DATASET_REGISTRY.items():
            assert info.task_type in ("regression", "classification"), (
                f"{name} has invalid task_type: {info.task_type}"
            )

    def test_metrics_valid(self):
        """All datasets should have a recognized metric."""
        valid_metrics = {"rmse", "mae", "auroc"}
        for name, info in DATASET_REGISTRY.items():
            assert info.metric in valid_metrics, (
                f"{name} has invalid metric: {info.metric}"
            )

    def test_tdc_groups_valid(self):
        """All datasets should have 'ADME' or 'Tox' as tdc_group."""
        for name, info in DATASET_REGISTRY.items():
            assert info.tdc_group in ("ADME", "Tox"), (
                f"{name} has invalid tdc_group: {info.tdc_group}"
            )

    def test_solubility_details(self):
        """Spot-check our primary dataset."""
        info = get_dataset_info("solubility")
        assert info.tdc_group == "ADME"
        assert info.target_column == "Y"

    def test_ames_is_classification(self):
        """Spot-check a classification dataset."""
        info = get_dataset_info("ames")
        assert info.task_type == "classification"
        assert info.metric == "auroc"
        assert info.tdc_group == "Tox"
