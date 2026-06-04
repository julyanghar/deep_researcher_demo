import asyncio

from deep_researcher_demo import cli
from fakes import StubChatClient, StubSearchProvider


def test_quiet_cli_suppresses_progress(capsys, monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda **kwargs: StubChatClient())
    monkeypatch.setattr(cli, "create_search_provider", lambda *args, **kwargs: StubSearchProvider())

    asyncio.run(
        cli.async_main(
            [
                "--env-file",
                str(env_file),
                "--quiet",
                "What are local LLM inference tradeoffs?",
            ]
        )
    )

    output = capsys.readouterr().out
    assert "[starting_research]" not in output
    assert "Stub Deep Research Report" in output
