import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from bmwvision.classifier import ModelMeta, OnnxTypeClassifier


def tiny_model(path, K, F):
    """image -> global avg pool (3) ++ geom (F) -> linear -> logits; embedding = concat."""
    rng = np.random.default_rng(0)
    W = rng.normal(size=(3 + F, K)).astype(np.float32)
    nodes = [
        helper.make_node("GlobalAveragePool", ["image"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"]),
        helper.make_node("Concat", ["flat", "geom"], ["embedding"], axis=1),
        helper.make_node("MatMul", ["embedding", "W"], ["logits"]),
    ]
    g = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, None, None]),
         helper.make_tensor_value_info("geom", TensorProto.FLOAT, [1, F])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, K]),
         helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 3 + F])],
        [numpy_helper.from_array(W, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def test_onnx_classifier_contract(tmp_path, type_names):
    K, F = len(type_names), 10
    tiny_model(str(tmp_path / "m.onnx"), K, F)
    meta = ModelMeta(type_names=type_names, geom_mean=[0.0] * F, geom_std=[1.0] * F)
    meta.save(tmp_path / "m.json")
    clf = OnnxTypeClassifier(tmp_path / "m.onnx", ModelMeta.load(tmp_path / "m.json"),
                             providers=["CPUExecutionProvider"])
    out = clf(np.full((600, 600, 3), 128, np.uint8), np.arange(F, dtype=np.float32))
    assert out.logits.shape == (K,) and out.embedding.shape == (3 + F,)
    # resize path when the ROI does not match the model input
    out2 = clf(np.full((300, 300, 3), 128, np.uint8), np.zeros(F, np.float32))
    assert np.isfinite(out2.logits).all()
