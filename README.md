# advaiapp1-final

このリポジトリは, 人工知能応用特論Ⅰ (Advanced Artificial Intelligence Applications I) の最終レポート課題に際して作成した成果物を公開するためのものです.

この課題は, 既存の論文のコードやPyTorchのデモコード等に対して, 型アノテーションの付与やテストコードの記述など保守性の向上に繋がる変更を加え, GitHub上で公開するものです.
今回, 私は[PyTorch LightningのTutorial](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/01-introduction-to-pytorch.html)で用いられているJupyter Notebook形式のコードを基に本課題に取り組みました.
変更元のソースコードは[GitHub上](https://github.com/Lightning-AI/tutorials/blob/main/course_UvA-DL/01-introduction-to-pytorch/notebook.py)でも公開されています.

## Execute

[uv](https://docs.astral.sh/uv/)を使用しています.

```sh
uv sync
uv run advaiapp1-final
```

### pytest

```sh
uv run pytest
```

## License

変更元のソースコードはCC BY-SAライセンスによって公開されているため ([参照](https://github.com/Lightning-AI/tutorials/blob/main/course_UvA-DL/01-introduction-to-pytorch/.meta.yml)), 本リポジトリについてもCC BY-SA 4.0で公開します.
ライセンスの全文については[LICENSEファイル](LICENSE)を参照してください.
