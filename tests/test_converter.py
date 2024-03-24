from src.retsim_pytorch.converter import convert_retsim
from src.retsim_pytorch.retsim import RETSim
from pathlib import Path
import pytest


def test_convert_retsim():
    weigth_path = Path(__file__).parent.parent / "weights" / "original" / "v1"
    data = convert_retsim(weigth_path)

    correct_state_dict = RETSim().state_dict()

    for key in correct_state_dict:
        if data[key].shape != correct_state_dict[key].shape:
            raise Exception(
                f"Shape mismatch for key {key} expected {correct_state_dict[key].shape} got {data[key].shape}"
            )
        if data[key].dtype != correct_state_dict[key].dtype:
            raise Exception(
                f"Dtype mismatch for key {key} expected {correct_state_dict[key].dtype} got {data[key].dtype}"
            )


pytest.mark.skip(reason="Only for manual testing")


def test_export_as_safetensor():
    from safetensors.torch import save_file

    weigth_path = Path(__file__).parent.parent / "weights" / "original" / "v1"
    data = convert_retsim(weigth_path)

    model = RETSim()
    model.load_state_dict(data)

    save_file(data, Path(__file__).parent.parent / "weights" / "model.safetensors")
