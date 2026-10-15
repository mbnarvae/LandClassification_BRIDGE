import importlib.util

from bridge_lc.config import REPO_ROOT


def _load_script():
    spec = importlib.util.spec_from_file_location("b", REPO_ROOT / "scripts/exp001_build_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_season_dates_wrap_year():
    mod = _load_script()
    assert mod.season_dates(["06-01", "09-30"], 2023) == ("2023-06-01", "2023-09-30")
    assert mod.season_dates(["12-01", "03-31"], 2023) == ("2022-12-01", "2023-03-31")
