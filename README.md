# Lec Notes

[![Deploy GitHub Pages](https://github.com/Lodpixel/Lec-Notes/actions/workflows/pages.yml/badge.svg)](https://github.com/Lodpixel/Lec-Notes/actions/workflows/pages.yml)

课程学习笔记与讲座转录，使用 Material for MkDocs 构建。

- 在线阅读：<https://lodpixel.github.io/Lec-Notes/>
- 内容目录：[`docs/`](docs/)

## 本地预览

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

浏览器访问 <http://127.0.0.1:8000/>。提交到 `main` 分支后，GitHub Actions 会自动构建并发布网站。
