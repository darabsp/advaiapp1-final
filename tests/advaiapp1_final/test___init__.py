from pytest import fixture, MonkeyPatch

import matplotlib
from pathlib import Path

from advaiapp1_final import main


# NOTE: Tkinterが使えないんだが？というエラーの抑制
@fixture(autouse=True)
def make_matplotlib_not_use_gui(
    monkeypatch: MonkeyPatch,
):
    matplotlib.use("Agg")
    monkeypatch.setattr("matplotlib.pyplot.Figure.show", lambda _: None) # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]


class TestMain():
    @fixture(autouse=True)
    def patch_for_main(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ):
        # NOTE: main()でwaitforbuttonpressしているのでそのままだとテストが進まない
        monkeypatch.setattr("matplotlib.pyplot.waitforbuttonpress", lambda: None)
        monkeypatch.chdir(tmp_path)

    def test_file_output(
        self,
        tmp_path: Path,
    ):
        main()

        assert tmp_path.joinpath("our_model").with_suffix(".tar").exists()
