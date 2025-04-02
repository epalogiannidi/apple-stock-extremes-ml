from apple_stock_extremes_ml import about


def test_version(version):
    assert about.__version__ == version
