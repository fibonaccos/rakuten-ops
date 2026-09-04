"""
Every page must render without raising, for each role that can reach it.

Streamlit pages are scripts: a typo, a bad import or a `st.session_state` key
that is only set after login shows up when a human clicks through the app.
`AppTest` runs the real entry point headless — navigation included, so
`st.page_link` resolves the way it does in the browser — and turns "the sell
page crashes for a logged-out visitor" into a test failure.
"""

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT

ENTRY_POINT = ROOT / "services" / "streamlit" / "app.py"

PAGES = [
    "views/connexion.py",
    "views/accueil.py",
    "views/vendre.py",
    "views/administration.py",
]

VISITOR: dict = {"token": None, "username": None, "role": None}
USER: dict = {"token": "a-token", "username": "strincal", "role": "user"}
ADMIN: dict = {"token": "a-token", "username": "rmazoyer", "role": "admin"}


def open_app(session_state: dict, page: str | None = None) -> AppTest:
    """Start the app with a given session, optionally on a specific page."""
    app = AppTest.from_file(str(ENTRY_POINT), default_timeout=60)
    for key, value in session_state.items():
        app.session_state[key] = value
    app.run()
    if page is not None:
        app.switch_page(page)
        app.run()
    return app


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize(
    ("who", "session"), [("visiteur", VISITOR), ("membre", USER), ("admin", ADMIN)]
)
def test_a_page_renders(page: str, who: str, session: dict) -> None:
    app = open_app(session, page)

    assert not app.exception, f"{page} a levé une exception pour un {who}"


def test_the_app_opens_on_the_home_page() -> None:
    """Accueil is the default page (st.Page(..., default=True) in app.py) --
    a visitor lands on the catalogue, not on the login screen."""
    app = open_app(VISITOR)

    assert not app.exception
    assert any("Top des produits" in subheader.value for subheader in app.subheader)


def test_the_sell_page_asks_a_visitor_to_log_in_first() -> None:
    app = open_app(VISITOR, "views/vendre.py")

    assert any("Connecte-toi" in warning.value for warning in app.warning)


def _admin_page_was_rendered(app: AppTest) -> bool:
    return any("Administration" in title.value for title in app.title)


def test_a_plain_user_cannot_reach_the_admin_page() -> None:
    """app.py leaves the page out of st.navigation, so the router refuses it."""
    app = open_app(USER, "views/administration.py")

    assert not app.exception
    assert not _admin_page_was_rendered(app)


def test_an_admin_reaches_the_admin_page() -> None:
    app = open_app(ADMIN, "views/administration.py")

    assert not app.exception
    assert _admin_page_was_rendered(app)


def test_the_home_page_shows_the_catalogue() -> None:
    app = open_app(VISITOR, "views/accueil.py")

    assert not app.exception
    assert any("Top des produits" in subheader.value for subheader in app.subheader)
