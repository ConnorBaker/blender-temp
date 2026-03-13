from pathlib import Path

import torch

from blender_temp.cmd import main as main_module
from blender_temp.cmd.main import (
    assess_run_safety,
    configure_torch_compile_runtime,
    configure_tf32_backends,
    fit_with_optional_pytorch_profiler,
    format_scale_tag,
    pytorch_profiler_schedule_kwargs,
    reset_run_logs,
    setup_argparse,
)


def test_parser_accepts_fractional_scale() -> None:
    args = setup_argparse().parse_args(["--input-dir", str(Path("/tmp")), "--scale", "1.5"])

    assert args.scale == 1.5


def test_parser_accepts_profile_pytorch_flag() -> None:
    args = setup_argparse().parse_args(["--input-dir", str(Path("/tmp")), "--profile-pytorch"])

    assert args.profile_pytorch is True


def test_parser_accepts_disable_torch_compile_flag() -> None:
    args = setup_argparse().parse_args(["--input-dir", str(Path("/tmp")), "--disable-torch-compile"])

    assert args.disable_torch_compile is True


def test_format_scale_tag_strips_trailing_zeroes() -> None:
    assert format_scale_tag(2.0) == "2"
    assert format_scale_tag(1.5) == "1.5"
    assert format_scale_tag(0.75) == "0.75"


def test_pytorch_profiler_schedule_kwargs_stays_small_for_short_runs() -> None:
    assert pytorch_profiler_schedule_kwargs(1) == {"wait": 0, "warmup": 0, "active": 1, "repeat": 1}
    assert pytorch_profiler_schedule_kwargs(5) == {"wait": 1, "warmup": 1, "active": 3, "repeat": 1}


def test_configure_tf32_backends_uses_compile_friendly_api(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(main_module.torch, "set_float32_matmul_precision", lambda mode: calls.append(mode))
    monkeypatch.setattr(main_module.torch.backends.cudnn, "allow_tf32", False)

    configure_tf32_backends()

    assert calls == ["high"]
    assert main_module.torch.backends.cudnn.allow_tf32 is True


def test_configure_torch_compile_runtime_sets_recompile_limit(monkeypatch) -> None:
    compiler_config = type("CompilerConfig", (), {"recompile_limit": 0})()

    monkeypatch.setattr(main_module.torch, "compiler", type("Compiler", (), {"config": compiler_config})())

    configure_torch_compile_runtime()

    assert compiler_config.recompile_limit == 32


def test_assess_run_safety_flags_high_risk_black_output_combo() -> None:
    issues = assess_run_safety(
        num_views=3,
        scale=2,
        anchor_stride=4,
        view_batch_size=1,
        radius_clip_px=256.0,
        disable_density_control_final_stage=True,
    )

    issue_codes = {(issue.severity, issue.code) for issue in issues}
    assert ("error", "high_risk_black_output") in issue_codes
    assert ("warning", "partial_view_batch") in issue_codes
    assert ("warning", "single_view_steps") in issue_codes
    assert ("warning", "scaled_radius_clip") in issue_codes
    assert ("warning", "no_final_density_recovery") in issue_codes


def test_assess_run_safety_accepts_full_batch_no_clip_run() -> None:
    issues = assess_run_safety(
        num_views=3,
        scale=1,
        anchor_stride=4,
        view_batch_size=0,
        radius_clip_px=0.0,
        disable_density_control_final_stage=False,
    )

    assert issues == []


def test_reset_run_logs_removes_only_append_only_logs(tmp_path) -> None:
    progress_log_path = tmp_path / "progress.jsonl"
    density_event_log_path = tmp_path / "density_events.jsonl"
    history_path = tmp_path / "history.json"
    progress_log_path.write_text("old-progress\n", encoding="utf-8")
    density_event_log_path.write_text("old-density\n", encoding="utf-8")
    history_path.write_text("{}", encoding="utf-8")

    returned_progress, returned_density = reset_run_logs(tmp_path)

    assert returned_progress == progress_log_path
    assert returned_density == density_event_log_path
    assert not progress_log_path.exists()
    assert not density_event_log_path.exists()
    assert history_path.exists()


def test_fit_with_optional_pytorch_profiler_wraps_fit_and_writes_summary(tmp_path, monkeypatch) -> None:
    profile_calls: dict[str, object] = {"profiles": []}

    class DummyKeyAverages:
        def table(self, *, sort_by: str, row_limit: int) -> str:
            return f"{sort_by}:{row_limit}"

    class DummyProfiler:
        def __init__(self, **kwargs):
            self.index = len(profile_calls["profiles"])
            profile_calls["profiles"].append(kwargs)
            self.step_count = 0
            self.started = False
            self.stopped = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def step(self) -> None:
            self.step_count += 1

        def key_averages(self) -> DummyKeyAverages:
            profile_calls[f"step_count_{self.index}"] = self.step_count
            return DummyKeyAverages()

    monkeypatch.setattr(main_module.torch.profiler, "schedule", lambda **kwargs: ("schedule", kwargs))
    monkeypatch.setattr(
        main_module.torch.profiler,
        "tensorboard_trace_handler",
        lambda path: ("trace_handler", path),
    )
    monkeypatch.setattr(main_module.torch.profiler, "profile", lambda **kwargs: DummyProfiler(**kwargs))

    class DummyPipeline:
        def fit(self, images, **kwargs):
            del images
            step_callback = kwargs["step_callback"]
            progress_event_callback = kwargs["progress_event_callback"]
            profile_calls["fit_kwargs_keys"] = sorted(kwargs.keys())
            progress_event_callback({"event": "stage_start", "stage_index": 0, "steps": 2})
            step_callback(0, 0, 0)
            step_callback(0, 1, 1)
            progress_event_callback({"event": "stage_end", "stage_index": 0, "steps": 2})
            progress_event_callback({"event": "stage_start", "stage_index": 1, "steps": 1})
            step_callback(1, 0, 2)
            progress_event_callback({"event": "stage_end", "stage_index": 1, "steps": 1})
            return {"loss": [1.0]}

    history = fit_with_optional_pytorch_profiler(
        DummyPipeline(),
        torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        output_dir=tmp_path,
        profile_pytorch=True,
        total_steps=5,
        fit_kwargs={"verbose_progress": False},
    )

    assert history == {"loss": [1.0]}
    assert profile_calls["step_count_0"] == 2
    assert profile_calls["step_count_1"] == 1
    assert profile_calls["fit_kwargs_keys"] == ["progress_event_callback", "step_callback", "verbose_progress"]
    assert (tmp_path / "pytorch-profile").is_dir()
    assert (tmp_path / "pytorch-profile" / "stage1-summary.txt").read_text(encoding="utf-8") == "self_cpu_time_total:50"
    assert (tmp_path / "pytorch-profile" / "stage2-summary.txt").read_text(encoding="utf-8") == "self_cpu_time_total:50"
    summary_text = (tmp_path / "pytorch-profile-summary.txt").read_text(encoding="utf-8")
    assert "[stage 1]" in summary_text
    assert "[stage 2]" in summary_text
