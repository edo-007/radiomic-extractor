"""Opzioni MIRP aggiuntive, separate dall'orchestrazione dell'estrattore CT."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Boundary = Literal["reflect", "constant", "nearest", "mirror", "wrap"]
Discretisation = Literal["none", "fixed_bin_size", "fixed_bin_number"]
Spatial = Literal["2d", "2.5d", "3d"]
DirectionalSpatial = Literal[
    "2d_average", "2d_slice_merge", "2.5d_direction_merge",
    "2.5d_volume_merge", "3d_average", "3d_volume_merge",
]
TexturePooling = Literal["average", "min", "max", "std", "var", "range"]


class MirpOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Preprocessing: default uguali a MIRP, nessuna nuova operazione attivata.
    spline_order: int = Field(3, ge=0, le=5)
    anti_aliasing: bool = True
    smoothing_beta: float = Field(0.98, gt=0, le=1)
    roi_spline_order: int = Field(1, ge=0, le=5)
    roi_interpolation_mask_inclusion_threshold: float = Field(0.5, gt=0, le=1)
    resegmentation_sigma: float | None = Field(None, gt=0)
    tissue_mask_type: Literal["none", "range", "relative_range"] = "relative_range"
    tissue_mask_range: list[float] | None = None
    intensity_normalisation: Literal[
        "none", "range", "relative_range", "quantile_range", "standardisation"
    ] = "none"
    intensity_normalisation_range: list[float] | None = None
    intensity_normalisation_saturation: list[float] | None = None
    intensity_scaling: float | None = None

    # Feature originali: bin_width rimane il nome pubblico del bin size.
    base_discretisation_method: Discretisation | list[Discretisation] = "fixed_bin_size"
    base_discretisation_n_bins: int | list[int] | None = None
    ivh_discretisation_method: Discretisation = "none"
    ivh_discretisation_n_bins: int = Field(1000, ge=2)
    ivh_discretisation_bin_width: float | None = Field(None, gt=0)
    texture_feature_pooling_method: TexturePooling | list[TexturePooling] = "average"
    glcm_distance: float | list[float] = 1.0
    glcm_spatial_method: DirectionalSpatial | list[DirectionalSpatial] | None = None
    glrlm_spatial_method: DirectionalSpatial | list[DirectionalSpatial] | None = None
    glszm_spatial_method: Spatial | list[Spatial] | None = None
    gldzm_spatial_method: Spatial | list[Spatial] | None = None
    ngtdm_spatial_method: Spatial | list[Spatial] | None = None
    ngldm_distance: float | list[float] = 1.0
    ngldm_difference_level: float | list[float] = 0.0
    ngldm_spatial_method: Spatial | list[Spatial] | None = None

    response_map_discretisation_method: Discretisation | list[Discretisation] = "fixed_bin_number"
    response_map_discretisation_bin_width: float | list[float] | None = None
    mean_filter_boundary_condition: Boundary | None = None
    laplacian_of_gaussian_boundary_condition: Boundary | None = None
    laws_boundary_condition: Boundary | None = None
    gabor_boundary_condition: Boundary | None = None
    separable_wavelet_boundary_condition: Boundary | None = None
    nonseparable_wavelet_boundary_condition: Boundary | None = None

    # Avanzate: crop, maschere e perturbazioni disattivati per default.
    config_str: str = ""
    no_approximation: bool = False
    mask_merge: bool = False
    mask_split: bool = False
    mask_select_largest_region: bool = False
    mask_select_largest_slice: bool = False
    crop_around_roi: bool = False
    crop_distance: float = Field(150.0, ge=0)
    perturbation_noise_repetitions: int = Field(0, ge=0)
    perturbation_noise_level: float | None = Field(None, ge=0)
    perturbation_rotation_angles: float | list[float] = 0.0
    perturbation_translation_fraction: float | list[float] = 0.0
    perturbation_roi_adapt_type: Literal["distance", "fraction"] = "distance"
    perturbation_roi_adapt_size: float | list[float] = 0.0
    perturbation_roi_adapt_max_erosion: float = Field(0.8, ge=0, le=1)
    perturbation_randomise_roi_repetitions: int = Field(0, ge=0)
    roi_split_boundary_size: float | list[float] = 0.0
    roi_split_max_erosion: float = Field(0.6, ge=0, le=1)

    def additional_mirp_kwargs(self) -> dict:
        return {name: getattr(self, name) for name in MirpOptions.model_fields}
