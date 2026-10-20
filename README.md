# Lec Notes

[![Deploy GitHub Pages](https://github.com/Lodpixel/Lec-Notes/actions/workflows/pages.yml/badge.svg)](https://github.com/Lodpixel/Lec-Notes/actions/workflows/pages.yml)

课程学习笔记与讲座转录，使用 Material for MkDocs 构建。

- 在线阅读：<https://lodpixel.github.io/Lec-Notes/>
- 内容目录：[`docs/`](docs/)

## 当前结构

- **课内**
  - 1844
  - 大学物理
  - 概率统计
  - ICS
  - 离散数学
  - 数据结构
- **课外自学**（[专题总览](docs/课外自学/index.md)）
  - [JetBrains](docs/课外自学/jetbrains/index.md)
  - [AI Infra](docs/课外自学/ai-infra/index.md)

课内课堂转录按实际上课日期组织；同一周不同日期的课程分别保存为独立页面。课外资料按系列归档：每个系列用 `index.md` 维护目录，每场讲座使用稳定的主题文件名；同场讲座分段录音合并整理并保留衔接说明。

新增专题时同步更新专题首页、专题总览、站点首页与 `mkdocs.yml` 导航。原始录音、识别缓存及凭据不放入 `docs/` 或提交到仓库；`docs/` 中未列入导航的文件也可能被构建和搜索。

## 本地预览

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

浏览器访问 <http://127.0.0.1:8000/>。提交到 `main` 分支后，GitHub Actions 会自动构建并发布网站。
