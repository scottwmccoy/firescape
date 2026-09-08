"""The CLI launches the as-run pipeline by name; --dry-run shows the command."""
import pytest

from firescape import cli
from firescape.config import CURRENT_CALIBRATION


def run(argv, capsys):
    rc = cli.main(argv)
    return rc, capsys.readouterr().out


def test_scripts_dir_is_the_checkout():
    assert (cli.scripts_dir() / "products" / "p8_fire_forecast.py").is_file()


@pytest.mark.parametrize("argv, script, tail", [
    (["prefire", "Bug"], "products/p8_fire_forecast.py", ["Bug", CURRENT_CALIBRATION]),
    (["storm", "Bug", "statewide_v1_3"], "products/p8_fire_storm_response.py", ["Bug", "statewide_v1_3"]),
    (["assess", "Hawk"], "products/p9_fire_observed.py", ["Hawk", CURRENT_CALIBRATION]),
    (["hindcast", "NV3930511982820240907"], "products/p16_mtbs_fire_hindcast.py",
     ["NV3930511982820240907", CURRENT_CALIBRATION]),
    (["annualprob"], "surface/p7_annual_probability.py", ["statewide_v1_2"]),
    (["calibrate", "--summary"], "calibrate/p8_calib_summary.py", []),
    (["severity"], "analysis/p2_severity_mosaic.py", []),
])
def test_dry_run_builds_the_right_command(argv, script, tail, capsys):
    rc, out = run(argv + ["--dry-run"], capsys)
    assert rc == 0
    line = out.strip().splitlines()[-1]
    assert line.endswith(" ".join([str(cli.scripts_dir() / script), *tail]).rstrip())


def test_map_all_runs_three_sheets(capsys):
    rc, out = run(["map", "Bug", "--sheet", "all", "--dry-run"], capsys)
    assert rc == 0
    lines = [l for l in out.splitlines() if l.startswith("$ ")]
    assert len(lines) == 3
    assert "p15_fire_postfire_hazard.py" in lines[0] and "--prefire" not in lines[0]
    assert "--prefire" in lines[1]
    assert "p8_fire_storm_map.py" in lines[2]


def test_fire_commands_require_a_fire():
    with pytest.raises(SystemExit):
        cli.main(["assess"])


def test_every_launch_target_exists():
    d = cli.scripts_dir()
    for rel in ("products/p8_fire_forecast.py", "products/p8_fire_storm_response.py",
                "products/p9_fire_observed.py", "products/p16_mtbs_fire_hindcast.py",
                "figures/p15_fire_postfire_hazard.py", "figures/p8_fire_storm_map.py",
                "figures/p8_fire_forecast_map.py", "analysis/p2_severity_mosaic.py",
                "calibrate/p8_calib_driver.py", "calibrate/p8_calib_summary.py",
                "surface/p7_annual_probability.py"):
        assert (d / rel).is_file(), rel
