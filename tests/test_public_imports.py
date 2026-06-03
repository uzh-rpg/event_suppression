import pytest


def test_public_train_and_validation_modules_import():
    import evsup.training  # noqa: F401
    import evsup.validation  # noqa: F401
    import evsup  # noqa: F401
    import evsup.training  # noqa: F401
    import evsup.validation  # noqa: F401


def test_eed_loader_gap_is_explicit():
    from evsup.data import build_eed_validation_sequences

    config = {"data": {"dataset": "eed", "path": "/tmp/does-not-matter"}}
    with pytest.raises(ImportError, match="EED validation is configured"):
        build_eed_validation_sequences(config)
