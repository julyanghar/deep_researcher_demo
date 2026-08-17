# conda env 备份:gpt-deep

deep_researcher_demo 的运行环境(agent 主链路 + 三 benchmark 评测)。

| 文件 | 说明 |
|---|---|
| `environment-gpt-deep.yml` | `conda env export --no-builds`(conda 层) |
| `requirements-gpt-deep.txt` | `pip list --format=freeze`(真实 pip 版本,大量包是 pip 装的,以这份为准) |

- Python 3.11.15,torch 2.9.1+cu128。
- `deep-researcher-demo==0.1.0` 是本仓库的 editable 安装(`pip install -e .`),重建时在仓库根执行,别从 freeze 里装。

## 重建

```bash
conda create -n gpt-deep python=3.11 -y
conda activate gpt-deep
pip install -r requirements-gpt-deep.txt   # 先手动删掉 deep-researcher-demo 那行
pip install -e /path/to/deep_researcher_demo
```
