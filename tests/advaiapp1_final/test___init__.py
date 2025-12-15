from pytest import fixture, MonkeyPatch

import matplotlib
from matplotlib.figure import Figure
from pathlib import Path

from advaiapp1_final import main


class TestMain():
    figure_show_count = 0

    @fixture(autouse=True)
    def patch_for_main(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ):
        def increment_figure_show_count(_: Figure):
            self.figure_show_count += 1

        # Tkinterが使えないんだが？？という不具合の対処
        matplotlib.use("Agg")
        monkeypatch.setattr("matplotlib.figure.Figure.show", increment_figure_show_count)
        # main()でwaitforbuttonpressしているのでそのままだとテストが進まない
        monkeypatch.setattr("matplotlib.pyplot.waitforbuttonpress", lambda: None)

        monkeypatch.chdir(tmp_path)

        yield

        # 後処理
        self.figure_show_count = 0

    def test_file_output(
        self,
        tmp_path: Path,
    ):
        main()
        assert tmp_path.joinpath("out/our_model.tar").exists()

    def test_figure_show_count(
        self,
    ):
        main()
        assert self.figure_show_count == 2
