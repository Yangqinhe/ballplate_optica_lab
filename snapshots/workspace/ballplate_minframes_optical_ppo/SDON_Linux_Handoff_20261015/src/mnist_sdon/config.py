"""Typed configuration for the phase-one differentiable SDON experiment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

from sdon.config import _load_toml


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("data/mnist")
    download: bool = False
    fit_samples: int = 1000
    validation_samples: int = 200
    overfit_samples: int = 20
    # Historical experiments drew both subsets from MNIST train.  New formal
    # classifiers can use the official test split so validation stays fixed
    # when fit_samples changes.
    validation_source: str = "train"


@dataclass(frozen=True)
class EncodingConfig:
    time_steps: int = 4
    minimum_probability: float = 1.0e-6
    input_mode: str = "rate"
    binary_threshold: float = 0.5


@dataclass(frozen=True)
class OpticalConfig:
    logical_size: int = 32
    readout_size: int = 16
    propagation_mode: str = "diffractive"
    simulation_size: int = 128
    # 0 keeps the historical behavior: one trainable phase value per
    # simulation pixel.  A smaller divisor constrains the trainable SLM phase
    # to a low-spatial-bandwidth control grid before differentiable upsampling.
    phase_control_size: int = 0
    # ``physical_size`` is the historical simulator aperture used to derive
    # the effective sampling pitch.  The three optional fields below describe
    # independent hardware planes.  A value of 0 preserves legacy behavior.
    physical_size: int = 512
    dmd_active_size: int = 0
    slm_active_size: int = 0
    ccd_active_size: int = 0
    physical_pixel_pitch_m: float = 8.0e-6
    wavelength_m: float = 532.0e-9
    focal_length_1_m: float = 0.15
    focal_length_2_m: float = 0.15
    phase_focal_length_m: float = 0.40
    padding_factor: int = 3
    phase_initialization: str = "random"
    phase_residual_std: float = 0.01
    phase_parameterization: str = "bounded"
    # Fraction of the calibrated device phase range available to the
    # trainable residual. A smaller value helps preserve a fixed blaze.
    phase_residual_scale: float = 1.0
    phase_modulation_depth_rad: float = 6.283185307179586
    # Fixed hardware-calibrated mapping from the designed wrapped phase to the
    # phase actually commanded to the SLM.  Keep this at 1 for legacy runs.
    phase_command_scale: float = 1.0
    # Fraction of detected intensity that follows the phase-modulated branch.
    # The remainder is an incoherent zero-order image measured on hardware.
    phase_modulated_fraction: float = 1.0
    # Optional fixed off-axis blaze in simulation-grid pixels. Zero disables it.
    phase_carrier_period_px: int = 0
    phase_carrier_axis: str = "x"
    phase_carrier_direction: int = 1
    # Optional label-free hardware calibration terms.  The input map is
    # indexed in the logical 32x32 DMD plane.  Readout blur and log-gain slopes
    # are applied on the final readout grid without introducing trainable
    # electronic classifier parameters.
    input_intensity_gain_map_path: str = ""
    readout_blur_sigma_x_px: float = 0.0
    readout_blur_sigma_y_px: float = 0.0
    readout_log_gain_slope_x_per_px: float = 0.0
    readout_log_gain_slope_y_per_px: float = 0.0
    phase_crosstalk_sigma_simulation_px: float = 0.0

    @property
    def effective_pixel_pitch_m(self) -> float:
        return self.physical_pixel_pitch_m * self.physical_size / self.simulation_size

    @property
    def resolved_phase_control_size(self) -> int:
        return self.simulation_size if self.phase_control_size == 0 else self.phase_control_size

    @property
    def resolved_dmd_active_size(self) -> int:
        return self.physical_size if self.dmd_active_size == 0 else self.dmd_active_size

    @property
    def resolved_slm_active_size(self) -> int:
        return self.physical_size if self.slm_active_size == 0 else self.slm_active_size

    @property
    def resolved_ccd_active_size(self) -> int:
        return self.physical_size if self.ccd_active_size == 0 else self.ccd_active_size


@dataclass(frozen=True)
class NeuronConfig:
    hidden_leak: float = 0.875
    hidden_threshold: float = 1.0
    hidden_gain: float = 2.0
    output_leak: float = 0.875
    output_threshold: float = 1.0
    surrogate_slope: float = 10.0
    membrane_weight: float = 0.01


@dataclass(frozen=True)
class TrainingConfig:
    warmup_epochs: int = 3
    phase_only_epochs: int = 0
    joint_epochs: int = 33
    micro_batch_size: int = 4
    effective_batch_size: int = 32
    phase_learning_rate: float = 3.0e-4
    fc_learning_rate: float = 1.0e-3
    lif_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    hidden_rate_min: float = 0.01
    hidden_rate_max: float = 0.50
    output_rate_min: float = 0.005
    output_rate_max: float = 0.40
    hidden_rate_weight: float = 0.01
    output_rate_weight: float = 0.01
    phase_tv_weight: float = 1.0e-5
    # Optimize the optical phase against the worst detector registration in
    # a square +/- radius neighbourhood.  Zero keeps the nominal readout.
    detector_shift_robustness_px: int = 0
    auxiliary_start_weight: float = 1.0
    num_workers: int = 0


@dataclass(frozen=True)
class OpticalClassifierConfig:
    # Original MNIST digits represented by detector regions.  Labels are
    # remapped to 0..N-1 internally, in this exact detector order.
    class_digits: Tuple[int, ...] = tuple(range(10))
    detector_layout: str = "grid"
    detector_rows: int = 2
    detector_columns: int = 5
    region_height: int = 6
    region_width: int = 3
    region_gap_y: int = 0
    region_gap_x: int = 0
    ring_radius: float = 10.0
    center_offset_x: float = 0.0
    center_offset_y: float = 0.0
    temperature: float = 1.0
    learning_rate: float = 3.0e-3
    epochs: int = 60
    outside_energy_weight: float = 0.05
    # Optional fixed detector masks used by the differentiable simulator.
    # This is a calibration asset, not a trainable electronic classifier.
    simulation_detector_regions_path: str = ""
    # Optional label-free affine correction applied only to the measured
    # 32x32 CCD readout before detector integration.  It is deliberately not
    # part of the simulated/trainable optical model.
    hardware_readout_correction_path: str = ""
    # Optional fixed hardware-only detector rectangles.  This compensates
    # local CCD/readout coordinate mismatch without adding classifier weights.
    hardware_detector_regions_path: str = ""
    # Optional ONN-page alignment derived from the current foundation
    # calibration.  It prevents a sensitive deployment offset from being
    # silently replaced by the latest general alignment.
    hardware_alignment_override_path: str = ""


@dataclass(frozen=True)
class OutputConfig:
    root: Path = Path("outputs/mnist_sdon")


@dataclass(frozen=True)
class Phase1Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    neurons: NeuronConfig = field(default_factory=NeuronConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optical_classifier: OpticalClassifierConfig = field(default_factory=OpticalClassifierConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    source_path: Path = Path("experiments/mnist_sdon.example.toml")


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError("[%s] must be a TOML table" % name)
    return value


def _make_dataclass(cls: Any, values: Dict[str, Any]) -> Any:
    known = cls.__dataclass_fields__
    unknown = sorted(set(values).difference(known))
    if unknown:
        raise ValueError("unknown %s fields: %s" % (cls.__name__, unknown))
    return cls(**values)


def _resolve_config_reference(config_path: Path, value: object) -> Path:
    """Resolve an input artifact referenced by an experiment configuration.

    Normal experiment files live in ``experiments/`` and historically resolve
    relative paths from the repository root (``config_path.parent.parent``).
    Frozen deployment configs live deeper under ``outputs/``.  Searching the
    config directory and its ancestors keeps those copied configs usable while
    preserving the historical repository-root behaviour.
    """
    reference = Path(str(value).strip())
    if reference.is_absolute():
        return reference.resolve()

    local = (config_path.parent / reference).resolve()
    if local.exists():
        return local
    # Prefer the actual project root when the same relative output directory
    # was accidentally created below a frozen deployment by an older build.
    for parent in config_path.parents:
        if (parent / "pyproject.toml").is_file():
            project_candidate = (parent / reference).resolve()
            if project_candidate.exists():
                return project_candidate
    historical = (config_path.parent.parent / reference).resolve()
    if historical.exists():
        return historical
    for parent in config_path.parents:
        candidate = (parent / reference).resolve()
        if candidate.exists():
            return candidate
    return historical


def load_phase1_config(path: Path) -> Phase1Config:
    """Load and validate a phase-one experiment configuration."""
    path = Path(path).resolve()
    data = _load_toml(path)
    experiment = _make_dataclass(ExperimentConfig, _section(data, "experiment"))
    data_values = dict(_section(data, "data"))
    data_root = Path(data_values.get("root", DataConfig().root))
    if not data_root.is_absolute():
        data_root = (path.parent.parent / data_root).resolve()
    data_values["root"] = data_root
    dataset = _make_dataclass(DataConfig, data_values)
    encoding = _make_dataclass(EncodingConfig, _section(data, "encoding"))
    optical_values = dict(_section(data, "optical"))
    gain_map_text = str(optical_values.get("input_intensity_gain_map_path", "")).strip()
    if gain_map_text:
        gain_map_path = _resolve_config_reference(path, gain_map_text)
        optical_values["input_intensity_gain_map_path"] = str(gain_map_path)
    optical = _make_dataclass(OpticalConfig, optical_values)
    neurons = _make_dataclass(NeuronConfig, _section(data, "neurons"))
    training = _make_dataclass(TrainingConfig, _section(data, "training"))
    classifier_values = dict(_section(data, "optical_classifier"))
    simulation_detector_regions_text = str(
        classifier_values.get("simulation_detector_regions_path", "")
    ).strip()
    if simulation_detector_regions_text:
        simulation_detector_regions_path = _resolve_config_reference(
            path, simulation_detector_regions_text
        )
        classifier_values["simulation_detector_regions_path"] = str(
            simulation_detector_regions_path
        )
    correction_text = str(
        classifier_values.get("hardware_readout_correction_path", "")
    ).strip()
    if correction_text:
        correction_path = _resolve_config_reference(path, correction_text)
        classifier_values["hardware_readout_correction_path"] = str(correction_path)
    detector_regions_text = str(
        classifier_values.get("hardware_detector_regions_path", "")
    ).strip()
    if detector_regions_text:
        detector_regions_path = _resolve_config_reference(
            path, detector_regions_text
        )
        classifier_values["hardware_detector_regions_path"] = str(
            detector_regions_path
        )
    alignment_override_text = str(
        classifier_values.get("hardware_alignment_override_path", "")
    ).strip()
    if alignment_override_text:
        alignment_override_path = _resolve_config_reference(
            path, alignment_override_text
        )
        classifier_values["hardware_alignment_override_path"] = str(
            alignment_override_path
        )
    optical_classifier = _make_dataclass(
        OpticalClassifierConfig, classifier_values
    )
    output_values = dict(_section(data, "outputs"))
    output_root = Path(output_values.get("root", OutputConfig().root))
    if not output_root.is_absolute():
        output_root = _resolve_config_reference(path, output_root)
    output_values["root"] = output_root
    outputs = _make_dataclass(OutputConfig, output_values)

    if encoding.time_steps < 1:
        raise ValueError("encoding.time_steps must be positive")
    if encoding.input_mode not in {"rate", "binary"}:
        raise ValueError("encoding.input_mode must be rate or binary")
    if not 0.0 <= encoding.binary_threshold <= 1.0:
        raise ValueError("encoding.binary_threshold must be in [0,1]")
    if optical.logical_size != 32:
        raise ValueError("phase one currently requires optical.logical_size=32")
    if optical.readout_size < 1 or optical.logical_size % optical.readout_size:
        raise ValueError("readout_size must be a positive divisor of logical_size")
    if optical.propagation_mode not in {"direct", "zero", "diffractive"}:
        raise ValueError("propagation_mode must be direct, zero, or diffractive")
    if optical.phase_parameterization not in {
        "bounded",
        "unbounded",
        "positive_bounded",
    }:
        raise ValueError(
            "phase_parameterization must be bounded, unbounded, or positive_bounded"
        )
    if not 0.0 < optical.phase_modulation_depth_rad <= 2.0 * 3.141592653589793:
        raise ValueError("phase_modulation_depth_rad must lie in (0,2pi]")
    if not 0.0 < optical.phase_residual_scale <= 1.0:
        raise ValueError("phase_residual_scale must lie in (0,1]")
    if not 0.0 < optical.phase_command_scale <= 2.0:
        raise ValueError("phase_command_scale must lie in (0,2]")
    if not 0.0 <= optical.phase_modulated_fraction <= 1.0:
        raise ValueError("phase_modulated_fraction must lie in [0,1]")
    if optical.phase_carrier_period_px not in {0} and optical.phase_carrier_period_px < 2:
        raise ValueError("phase_carrier_period_px must be zero or at least two")
    if optical.phase_carrier_axis not in {"x", "y"}:
        raise ValueError("phase_carrier_axis must be x or y")
    if optical.phase_carrier_direction not in {-1, 1}:
        raise ValueError("phase_carrier_direction must be -1 or +1")
    if min(
        optical.readout_blur_sigma_x_px,
        optical.readout_blur_sigma_y_px,
        optical.phase_crosstalk_sigma_simulation_px,
    ) < 0.0:
        raise ValueError("measured forward-model blur/crosstalk sigmas must be non-negative")
    if max(
        abs(optical.readout_log_gain_slope_x_per_px),
        abs(optical.readout_log_gain_slope_y_per_px),
    ) > 1.0:
        raise ValueError("readout log-gain slopes must lie in [-1,1] per pixel")
    if optical.input_intensity_gain_map_path:
        gain_map_path = Path(optical.input_intensity_gain_map_path)
        if not gain_map_path.is_file():
            raise ValueError("input intensity gain map does not exist: %s" % gain_map_path)
    if optical.simulation_size < optical.logical_size:
        raise ValueError("simulation_size must be at least logical_size")
    if optical.resolved_phase_control_size < 1:
        raise ValueError("phase_control_size must be zero or positive")
    if optical.simulation_size % optical.resolved_phase_control_size:
        raise ValueError("phase_control_size must divide simulation_size")
    if optical.physical_size % optical.simulation_size:
        raise ValueError("physical_size must be divisible by simulation_size")
    for name, value in (
        ("dmd_active_size", optical.resolved_dmd_active_size),
        ("slm_active_size", optical.resolved_slm_active_size),
        ("ccd_active_size", optical.resolved_ccd_active_size),
    ):
        if value < 1 or value > optical.physical_size:
            raise ValueError(
                "%s must resolve to a value in [1, physical_size]" % name
            )
        if (value * optical.simulation_size) % optical.physical_size:
            raise ValueError(
                "%s must map to an integer number of simulation pixels" % name
            )
    if optical.resolved_slm_active_size != optical.physical_size:
        raise ValueError("slm_active_size must span physical_size in the current model")
    if optical.padding_factor < 1:
        raise ValueError("padding_factor must be positive")
    if training.effective_batch_size % training.micro_batch_size:
        raise ValueError("effective_batch_size must be divisible by micro_batch_size")
    if not 0 <= int(training.detector_shift_robustness_px) <= 2:
        raise ValueError("detector_shift_robustness_px must lie in [0,2]")
    class_digits = tuple(int(value) for value in optical_classifier.class_digits)
    if not class_digits:
        raise ValueError("optical classifier must define at least two classes")
    if len(class_digits) < 2 or len(set(class_digits)) != len(class_digits):
        raise ValueError("optical classifier class_digits must be unique")
    if any(value < 0 or value > 9 for value in class_digits):
        raise ValueError("optical classifier class_digits must lie in 0..9")
    if optical_classifier.detector_rows * optical_classifier.detector_columns != len(
        class_digits
    ):
        raise ValueError(
            "optical classifier detector grid must match the number of class_digits"
        )
    if optical_classifier.detector_layout not in {"grid", "ring"}:
        raise ValueError("detector_layout must be grid or ring")
    if optical_classifier.detector_layout == "grid":
        if optical_classifier.region_gap_x < 0 or optical_classifier.region_gap_y < 0:
            raise ValueError("optical classifier region gaps must be non-negative")
        used_height = (
            optical_classifier.detector_rows * optical_classifier.region_height
            + (optical_classifier.detector_rows - 1)
            * optical_classifier.region_gap_y
        )
        used_width = (
            optical_classifier.detector_columns * optical_classifier.region_width
            + (optical_classifier.detector_columns - 1)
            * optical_classifier.region_gap_x
        )
        if (
            used_height > optical.readout_size
            or used_width > optical.readout_size
        ):
            raise ValueError("optical classifier regions must fit inside optical.readout_size")
    elif not 0.0 < optical_classifier.ring_radius < optical.readout_size / 2.0:
        raise ValueError("ring_radius must lie inside the optical readout")
    if optical_classifier.temperature <= 0.0:
        raise ValueError("optical classifier temperature must be positive")
    if optical_classifier.learning_rate <= 0.0 or optical_classifier.epochs < 1:
        raise ValueError("optical classifier learning rate and epochs must be positive")
    if optical_classifier.simulation_detector_regions_path:
        simulation_detector_regions_path = Path(
            optical_classifier.simulation_detector_regions_path
        )
        if not simulation_detector_regions_path.is_file():
            raise ValueError(
                "simulation detector region calibration does not exist: %s"
                % simulation_detector_regions_path
            )
    if optical_classifier.hardware_readout_correction_path:
        correction_path = Path(optical_classifier.hardware_readout_correction_path)
        if not correction_path.is_file():
            raise ValueError(
                "hardware readout correction does not exist: %s" % correction_path
            )
    if optical_classifier.hardware_detector_regions_path:
        detector_regions_path = Path(
            optical_classifier.hardware_detector_regions_path
        )
        if not detector_regions_path.is_file():
            raise ValueError(
                "hardware detector region calibration does not exist: %s"
                % detector_regions_path
            )
    if optical_classifier.hardware_alignment_override_path:
        alignment_override_path = Path(
            optical_classifier.hardware_alignment_override_path
        )
        if not alignment_override_path.is_file():
            raise ValueError(
                "hardware alignment override does not exist: %s"
                % alignment_override_path
            )
    if not (0.0 < neurons.hidden_leak < 1.0 and 0.0 < neurons.output_leak < 1.0):
        raise ValueError("LIF leak values must be in (0,1)")
    if min(dataset.fit_samples, dataset.validation_samples, dataset.overfit_samples) < 1:
        raise ValueError("dataset sample counts must be positive")
    if dataset.validation_source not in {"train", "test"}:
        raise ValueError("data.validation_source must be train or test")
    return Phase1Config(
        experiment=experiment,
        data=dataset,
        encoding=encoding,
        optical=optical,
        neurons=neurons,
        training=training,
        optical_classifier=optical_classifier,
        outputs=outputs,
        source_path=path,
    )
