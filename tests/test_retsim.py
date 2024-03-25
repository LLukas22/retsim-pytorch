import sys
from unittest.mock import Mock

sys.modules["nmslib"] = Mock()

from src.retsim_pytorch.retsim import RETSim
from src.retsim_pytorch.preprocessing import binarize
from src.retsim_pytorch.retsim import RETSim
import torch
from safetensors import safe_open
from pathlib import Path
import onnxruntime as ort
from torch.testing import assert_close


def test_is_constructable():
    model = RETSim()
    assert model is not None


def test_forward():
    model = RETSim()
    x = torch.rand(1, 512, 24)
    embedding,_ = model(x)
    assert embedding is not None
    assert embedding.shape == (1, 256)

def test_half():
    model = RETSim()
    model = model.half()
    x = torch.rand(1, 512, 24).half()
    embedding,_ = model(x)
    assert embedding is not None
    assert embedding.shape == (1, 256)

def test_with_encoder():
    model = RETSim()
    input, chunks = binarize(["hello world"])
    embedding,_ = model(torch.tensor(input))
    assert embedding is not None
    assert embedding.shape == (1, 256)

def test_against_onnx_model():
    input, chunks = binarize(["hello world", "foo bar"])

    model = RETSim()
    weigth_path = Path(__file__).parent.parent / "weights" / "model.safetensors"
    tensors = {}
    with safe_open(weigth_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    model.load_state_dict(tensors)

    embedding, unpooled = model(torch.tensor(input, dtype=torch.float32))
    assert embedding is not None

    session = ort.InferenceSession(
        str(Path(__file__).parent.parent / "weights" / "original" / "v1.onnx")
    )

    onnx_result = torch.tensor(
        session.run(None, {session.get_inputs()[0].name: input})[0]
    )

    assert_close(embedding, onnx_result, atol=1e-3, rtol=1e-3)

    torch_similarity = embedding[0] @ embedding[1]
    onnx_similarity = onnx_result[0] @ onnx_result[1]

    assert round(torch_similarity.item(), 4) == round(onnx_similarity.item(), 4)


def test_can_be_exported_as_onnx():
    input, chunks = binarize(["hello world", "foo bar"])

    model = RETSim()
    weigth_path = Path(__file__).parent.parent / "weights" / "model.safetensors"
    tensors = {}
    with safe_open(weigth_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    model.load_state_dict(tensors)
    model.eval()
    
    export_dir = Path(__file__).parent / "test_out"
    export_dir.mkdir(exist_ok=True)
    script_model = torch.jit.script(model, example_inputs=[(torch.tensor(input, dtype=torch.float32),)])
    torch.onnx.export(
        script_model,
        torch.tensor(input, dtype=torch.float32),
        export_dir / "model.onnx",
        dynamic_axes={
            "input": {0: "batch_size"},
        },
        input_names=["input"],
        output_names=["embedded","unpooled"],
    )
    
    original_session = ort.InferenceSession(
        str(Path(__file__).parent.parent / "weights" / "original" / "v1.onnx")
    )

    original_onnx_result = torch.tensor(
        original_session.run(None, {original_session.get_inputs()[0].name: input})[0]
    )
    
    new_session = ort.InferenceSession(str(export_dir / "model.onnx"))
    
    new_onnx_result = torch.tensor(
        new_session.run(None, {new_session.get_inputs()[0].name: input})[0]
    )
    
    assert_close(new_onnx_result, original_onnx_result, atol=1e-3, rtol=1e-3)
    
    