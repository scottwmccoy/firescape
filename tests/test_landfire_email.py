import pytest

from firescape import landfire


def test_delivery_email_precedence(monkeypatch):
    monkeypatch.setenv("FIRESCAPE_EMAIL", "general@example.edu")
    monkeypatch.setenv("FIRESCAPE_LFPS_EMAIL", "lfps@example.edu")
    assert landfire.delivery_email("arg@example.edu") == "arg@example.edu"
    assert landfire.delivery_email() == "lfps@example.edu"        # older name wins
    monkeypatch.delenv("FIRESCAPE_LFPS_EMAIL")
    assert landfire.delivery_email() == "general@example.edu"


def test_delivery_email_has_no_default(monkeypatch):
    monkeypatch.delenv("FIRESCAPE_EMAIL", raising=False)
    monkeypatch.delenv("FIRESCAPE_LFPS_EMAIL", raising=False)
    with pytest.raises(ValueError, match="FIRESCAPE_EMAIL"):
        landfire.delivery_email()
    with pytest.raises(ValueError):
        landfire._email(None)
