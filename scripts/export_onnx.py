#!/usr/bin/env python3
"""
Export NeMo Parakeet or Canary model to ONNX for Go inference.

Usage:
    # From a local .nemo file:
    uv run python3 scripts/export_onnx.py \\
        --model parakeet \\
        --checkpoint /path/to/parakeet-ctc-0.6b-en.nemo \\
        --output-dir /path/to/output

    # Auto-download from HuggingFace / NGC by model ID:
    uv run python3 scripts/export_onnx.py \\
        --model parakeet \\
        --checkpoint nvidia/parakeet-ctc-0.6b-en \\
        --output-dir /path/to/output

    uv run python3 scripts/export_onnx.py \\
        --model parakeet \\
        --checkpoint nvidia/parakeet-tdt-0.6b-v2 \\
        --output-dir /path/to/output

    uv run python3 scripts/export_onnx.py \\
        --model canary \\
        --checkpoint nvidia/canary-180m \\
        --output-dir /path/to/output

Supported Parakeet model classes:
  EncDecCTCModelBPE          — pure CTC (parakeet-ctc-*)
  EncDecHybridRNNTCTCModel   — TDT+CTC hybrid (parakeet-tdt-*); exports CTC head

For Parakeet, produces: {output-dir}/{stem}_encoder_ctc.onnx
For Canary, produces:   {output-dir}/{stem}_encoder.onnx
                        {output-dir}/{stem}_decoder.onnx

Input/output names used by the Go code:
  Parakeet encoder+CTC:
    inputs:  audio_signal [B, n_mels, T], length [B]
    outputs: logprobs [B, T', V]

  Canary encoder:
    inputs:  audio_signal [B, n_mels, T], length [B]
    outputs: encoder_output [B, T', D], encoded_lengths [B]

  Canary decoder:
    inputs:  encoder_output [B, T', D], encoder_lengths [B],
             targets [B, U] (int64), target_lengths [B]
    outputs: log_probs [B, U, V]
"""

import argparse
import os
from pathlib import Path


def _stem(checkpoint_path: str) -> str:
    if os.path.isfile(checkpoint_path):
        return Path(checkpoint_path).stem
    return checkpoint_path.replace("/", "-")


def _prepare_for_inference(model) -> None:
    """Strip all training-specific state so the model is safe to trace/export."""
    model.eval()
    # RNNT joint: disable fused loss+WER computation (training-only).
    if hasattr(model, 'joint'):
        joint = model.joint
        joint._fuse_loss_wer = False
        joint._loss = None
        joint._wer = None
    # Freeze all parameters (no-op for export, but makes intent clear).
    for p in model.parameters():
        p.requires_grad_(False)


def _load_parakeet(checkpoint_path: str):
    """
    Load any Parakeet model, auto-detecting the model class.
    Returns the loaded model and its class name.
    """
    import nemo.collections.asr as nemo_asr

    if os.path.isfile(checkpoint_path):
        model = nemo_asr.models.ASRModel.restore_from(checkpoint_path, map_location="cpu")
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(checkpoint_path, map_location="cpu")

    cls_name = type(model).__name__
    print(f"  Model class: {cls_name}")
    return model, cls_name


def export_parakeet(checkpoint_path: str, output_dir: str) -> None:
    import nemo.collections.asr as nemo_asr

    print(f"Loading Parakeet checkpoint: {checkpoint_path}")
    model, cls_name = _load_parakeet(checkpoint_path)
    _prepare_for_inference(model)

    stem = _stem(checkpoint_path)

    # Pure RNNT/TDT models — delegate to RNNT export path automatically.
    if hasattr(model, "joint") and not hasattr(model, "ctc_decoder") and \
            not isinstance(model, nemo_asr.models.EncDecCTCModelBPE):
        print(f"  Detected pure RNNT/TDT model ({cls_name}); switching to RNNT export.")
        _export_rnnt_from_model(model, output_dir, stem)
        return

    # Hybrid TDT+CTC models (EncDecHybridRNNTCTCModel) have both an RNNT and a
    # CTC head.  Switch to the CTC decoder so model.export() produces logprobs.
    if hasattr(model, "cur_decoder"):
        print("  Switching to CTC decoder for export (hybrid model).")
        model.cur_decoder = "ctc"
    out_path = os.path.join(output_dir, f"{stem}_encoder_ctc.onnx")

    print(f"Exporting to: {out_path}")
    model.export(
        out_path,
        check_trace=False,
        verbose=False,
        onnx_opset_version=17,
    )

    if not os.path.isfile(out_path):
        raise SystemExit(
            f"Export completed but output file not found: {out_path}\n"
            "NeMo may have written the file to a different location. "
            "Check for *.onnx files in the current directory."
        )

    print(f"Done: {out_path}")

    import onnx
    m = onnx.load(out_path)
    print("Input names: ", [i.name for i in m.graph.input])
    print("Output names:", [o.name for o in m.graph.output])


def export_canary(checkpoint_path: str, output_dir: str) -> None:
    import nemo.collections.asr as nemo_asr
    import torch
    import onnx

    print(f"Loading Canary checkpoint: {checkpoint_path}")
    if os.path.isfile(checkpoint_path):
        model = nemo_asr.models.EncDecMultiTaskModel.restore_from(
            checkpoint_path, map_location="cpu"
        )
    else:
        model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(
            checkpoint_path, map_location="cpu"
        )
    _prepare_for_inference(model)

    stem = _stem(checkpoint_path)

    # --- Export encoder ---
    enc_path = os.path.join(output_dir, f"{stem}_encoder.onnx")
    print(f"Exporting encoder to: {enc_path}")

    dummy_audio = torch.zeros(1, model.cfg.preprocessor.n_mels, 100)
    dummy_len = torch.tensor([100], dtype=torch.long)

    torch.onnx.export(
        model.encoder,
        (dummy_audio, dummy_len),
        enc_path,
        input_names=["audio_signal", "length"],
        output_names=["encoder_output", "encoded_lengths"],
        dynamic_axes={
            "audio_signal": {0: "batch", 2: "time"},
            "length": {0: "batch"},
            "encoder_output": {0: "batch", 1: "time"},
            "encoded_lengths": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Done: {enc_path}")

    # --- Export decoder ---
    dec_path = os.path.join(output_dir, f"{stem}_decoder.onnx")
    print(f"Exporting decoder to: {dec_path}")

    enc_hidden = model.cfg.model_defaults.asr_enc_hidden
    dummy_enc_out = torch.zeros(1, 50, enc_hidden)
    dummy_enc_len = torch.tensor([50], dtype=torch.long)
    dummy_targets = torch.zeros(1, 4, dtype=torch.long)
    dummy_tgt_len = torch.tensor([4], dtype=torch.long)

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.transf_decoder = m.transf_decoder
            self.log_softmax = m.log_softmax

        def forward(self, encoder_output, encoder_lengths, targets, target_lengths):
            dec_out = self.transf_decoder(
                encoder_states=encoder_output,
                encoder_mems_list=None,
                decoder_mems_list=None,
                labels=targets,
            )
            log_probs = self.log_softmax(log_probs=dec_out[0])
            return log_probs

    wrapper = DecoderWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy_enc_out, dummy_enc_len, dummy_targets, dummy_tgt_len),
        dec_path,
        input_names=["encoder_output", "encoder_lengths", "targets", "target_lengths"],
        output_names=["log_probs"],
        dynamic_axes={
            "encoder_output": {0: "batch", 1: "enc_time"},
            "encoder_lengths": {0: "batch"},
            "targets": {0: "batch", 1: "dec_time"},
            "target_lengths": {0: "batch"},
            "log_probs": {0: "batch", 1: "dec_time"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Done: {dec_path}")

    for p in [enc_path, dec_path]:
        m = onnx.load(p)
        print(f"\n{p}")
        print("  Input names: ", [i.name for i in m.graph.input])
        print("  Output names:", [o.name for o in m.graph.output])


def _export_rnnt_from_model(model, output_dir: str, stem: str) -> None:
    """Export encoder + decoder+joint for an already-loaded RNNT/TDT model."""
    import json
    import torch
    import onnx
    from omegaconf import OmegaConf
    from nemo.core.classes.common import typecheck
    from nemo.collections.asr.modules.rnnt import RNNTDecoderJoint

    pp = OmegaConf.to_container(model.cfg.preprocessor, resolve=True)
    n_mels = pp.get("n_mels") or pp.get("features") or 80
    dummy_audio = torch.zeros(1, n_mels, 100)
    dummy_len = torch.tensor([100], dtype=torch.long)

    # --- Export encoder ---
    enc_path = os.path.join(output_dir, f"{stem}_encoder.onnx")
    print(f"Exporting encoder to: {enc_path}")

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, audio_signal, length):
            out = self.enc(audio_signal=audio_signal, length=length)
            return out[0], out[1]

    enc_wrapper = EncoderWrapper(model.encoder)
    enc_wrapper.eval()

    with typecheck.disable_checks():
        torch.onnx.export(
            enc_wrapper,
            (dummy_audio, dummy_len),
            enc_path,
            input_names=["audio_signal", "length"],
            output_names=["outputs", "encoded_lengths"],
            dynamic_axes={
                "audio_signal": {0: "batch", 2: "time"},
                "length": {0: "batch"},
                "outputs": {0: "batch", 2: "time"},
                "encoded_lengths": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"Done: {enc_path}")

    # --- Export decoder+joint single-step ---
    dec_joint = RNNTDecoderJoint(model.decoder, model.joint)
    dec_joint.eval()

    example = dec_joint.input_example(max_batch=1, max_dim=1)
    enc_out_ex, targets_ex, tgt_len_ex, state1_ex, state2_ex = example
    print(f"  Decoder state1 shape: {list(state1_ex.shape)}")
    print(f"  Decoder state2 shape: {list(state2_ex.shape)}")

    dj_path = os.path.join(output_dir, f"{stem}_decoder_joint.onnx")
    print(f"Exporting decoder+joint to: {dj_path}")

    with typecheck.disable_checks():
        torch.onnx.export(
            dec_joint,
            example,
            dj_path,
            input_names=["encoder_outputs", "targets", "target_length",
                         "input_states_1", "input_states_2"],
            output_names=["outputs", "prednet_lengths",
                          "output_states_1", "output_states_2"],
            dynamic_axes={
                "encoder_outputs": {0: "batch", 2: "enc_time"},
                "targets": {0: "batch", 1: "tgt_time"},
                "target_length": {0: "batch"},
                "outputs": {0: "batch", 1: "enc_time", 2: "tgt_time"},
                "prednet_lengths": {0: "batch"},
                "input_states_1": {1: "batch"},
                "input_states_2": {1: "batch"},
                "output_states_1": {1: "batch"},
                "output_states_2": {1: "batch"},
            },
            opset_version=17,
        )
    print(f"Done: {dj_path}")

    # Collect TDT durations — try decoding config then loss config.
    durations = None
    for keys in [("decoding", "durations"), ("loss", "durations")]:
        try:
            cfg = model.cfg
            for k in keys:
                cfg = getattr(cfg, k)
            durations = list(cfg)
            break
        except Exception:
            pass
    if durations is None:
        durations = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        print(f"  WARNING: could not read durations from config; using default {durations}")

    # joint.num_classes = vocab size excluding blank; blank_idx = vocab_size.
    vocab_size = int(model.joint._num_classes)

    meta = {
        "durations": durations,
        "vocab_size": vocab_size,
        "blank_id": vocab_size,
        "state1_shape": list(state1_ex.shape),
        "state2_shape": list(state2_ex.shape),
        "n_mels": n_mels,
    }
    meta_path = os.path.join(output_dir, f"{stem}_rnnt_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote RNNT metadata: {meta_path}")
    print(f"  durations: {durations}, vocab_size: {vocab_size}, blank_id: {vocab_size}")

    # Save tokenizer for the Go checkpoint loader.
    tok_path = os.path.join(output_dir, "tokenizer.model")
    if not os.path.exists(tok_path):
        try:
            tok_bytes = model.tokenizer.tokenizer.serialized_model_proto()
            with open(tok_path, "wb") as f:
                f.write(tok_bytes)
            print(f"Wrote tokenizer: {tok_path}")
        except Exception as e:
            print(f"  WARNING: could not save tokenizer automatically: {e}")
            print("  Copy tokenizer.model from the .nemo archive manually.")

    for p in [enc_path, dj_path]:
        m = onnx.load(p)
        print(f"\n{p}")
        print("  Input names: ", [i.name for i in m.graph.input])
        print("  Output names:", [o.name for o in m.graph.output])


def export_rnnt(checkpoint_path: str, output_dir: str) -> None:
    """Export encoder + decoder+joint for a pure RNNT/TDT model."""
    print(f"Loading RNNT checkpoint: {checkpoint_path}")
    model, _ = _load_parakeet(checkpoint_path)
    _prepare_for_inference(model)
    _export_rnnt_from_model(model, output_dir, _stem(checkpoint_path))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=["parakeet", "canary", "rnnt"], required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .nemo file or HuggingFace model ID (e.g. nvidia/parakeet-tdt-0.6b-v2)")
    parser.add_argument("--output-dir", default=".", help="Directory for .onnx output files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == "parakeet":
        export_parakeet(args.checkpoint, args.output_dir)
    elif args.model == "rnnt":
        export_rnnt(args.checkpoint, args.output_dir)
    else:
        export_canary(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
