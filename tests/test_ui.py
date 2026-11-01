from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_panel_loads_and_invalid_input_has_useful_error():
    ui = Path(__file__).resolve().parents[1] / "src/panorama/ui.py"
    app = AppTest.from_file(str(ui)).run(timeout=30)
    assert not app.exception
    app.text_input[0].set_value("/this-folder-does-not-exist")
    next(button for button in app.button if button.label == "开始拼接").click()
    app.run(timeout=30)
    assert not app.exception
    assert any("目录不存在" in error.value for error in app.error)
