"""
The stack definition and the environment template must stay in step.

`docker compose up` substitutes every `${VAR}` in docker-compose.yaml from the
shell environment. A variable the compose file substitutes but `.env.example`
never mentions is invisible until someone clones the repository and the stack
comes up half-configured, so it is checked here instead.

Variables delivered to a container through `env_file` are not covered: compose
passes those through without substituting them, and each service validates its
own with pydantic-settings at startup.
"""

import re
import subprocess

import pytest

from tests.conftest import ROOT

COMPOSE = ROOT / "docker-compose.yaml"
ENV_EXAMPLE = ROOT / ".env.example"

# ${VAR}, ${VAR:-default} and ${VAR-default}. $${VAR} is escaped for the
# container shell and is never substituted by compose.
_REFERENCE = re.compile(r"(?<!\$)\$\{([A-Za-z_][A-Za-z0-9_]*)(:?-[^}]*)?\}")


def _variables_without_default() -> list[str]:
    text = COMPOSE.read_text(encoding="utf-8")
    return sorted(
        {match.group(1) for match in _REFERENCE.finditer(text) if match.group(2) is None}
    )


def _documented_variables() -> set[str]:
    names: set[str] = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        names.add(line.split("=", 1)[0].strip())
    return names


def test_the_compose_file_and_the_env_template_are_present() -> None:
    assert COMPOSE.is_file()
    assert ENV_EXAMPLE.is_file()


def test_the_compose_file_substitutes_something() -> None:
    """Guards the regex above: an empty match list would make the next test vacuous."""
    assert len(_variables_without_default()) > 10


@pytest.mark.parametrize("variable", _variables_without_default())
def test_every_required_variable_is_documented(variable: str) -> None:
    assert variable in _documented_variables(), (
        f"docker-compose.yaml substitutes ${{{variable}}} with no default, "
        f"but .env.example never mentions it."
    )


def test_the_real_env_file_is_not_tracked() -> None:
    """`.env` holds the database and Grafana credentials; it must stay untracked."""
    tracked = subprocess.run(
        ["git", "ls-files", "--", ".env"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    assert tracked == [], f"committed to the repository: {tracked}"
