from deep_researcher_demo.config import AppConfig
from deep_researcher_demo.cli import default_env_path


def test_config_reads_runtime_env(monkeypatch):
    monkeypatch.setenv("MAX_ITERATIONS", "7")
    monkeypatch.setenv("MAX_FOLLOWUPS", "4")
    monkeypatch.setenv("MAX_QUERIES_PER_RESEARCHER", "3")
    monkeypatch.setenv("MAX_CONCURRENCY", "2")
    monkeypatch.setenv("MAX_RESULTS", "6")
    monkeypatch.setenv("FETCH_WEBPAGES", "false")
    monkeypatch.setenv("MAX_CONTENT_CHARS", "99")
    monkeypatch.setenv("FETCH_TIMEOUT", "2.5")
    monkeypatch.setenv("FETCH_CONCURRENCY", "4")
    monkeypatch.setenv("OUTPUT", "outputs/test.md")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")

    config = AppConfig.from_env()

    assert config.max_iterations == 7
    assert config.max_followups == 4
    assert config.max_queries_per_researcher == 3
    assert config.max_concurrency == 2
    assert config.max_results == 6
    assert config.fetch_webpages is False
    assert config.max_content_chars == 99
    assert config.fetch_timeout == 2.5
    assert config.fetch_concurrency == 4
    assert config.output == "outputs/test.md"
    assert config.judge_base_url == "https://judge.example/v1"
    assert config.judge_api_key == "judge-key"
    assert config.judge_model == "judge-model"


def test_default_env_path_points_to_repo_root():
    assert default_env_path().name == ".env"
    assert default_env_path().parent.name == "deep_researcher_demo"
