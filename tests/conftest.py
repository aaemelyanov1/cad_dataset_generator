import sys
from pathlib import Path

# Корень репозитория (D:/Studing/Diploma), чтобы пакет cad_dataset_generator импортировался
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _strip_node_ids(d):
    if isinstance(d, dict):
        return {k: _strip_node_ids(v) for k, v in d.items() if k != "node_id"}
    if isinstance(d, list):
        return [_strip_node_ids(x) for x in d]
    return d


import pytest


@pytest.fixture
def strip_node_ids():
    return _strip_node_ids
