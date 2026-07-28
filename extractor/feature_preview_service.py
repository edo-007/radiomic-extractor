from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from .models import DicomMetadataConfig, MirpConfig
except ImportError:
    from models import DicomMetadataConfig, MirpConfig


@dataclass(frozen=True)
class RadiomicFeaturePreviewGroup:
    """Feature expected from one image source, such as the original CT or one response map."""

    source: str
    feature_families: tuple[str, ...]
    feature_names: tuple[str, ...]
    parameters: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class RadiomicFeaturePreview:
    """Preview of the radiomic and metadata columns expected in the final CSV."""

    patient_count: int
    groups: tuple[RadiomicFeaturePreviewGroup, ...]
    metadata_columns: tuple[str, ...] = ()

    @property
    def radiomic_feature_count(self) -> int:
        return sum(group.count for group in self.groups)

    @property
    def csv_column_count(self) -> int:
        return 2 + len(self.metadata_columns) + self.radiomic_feature_count

    @property
    def all_feature_names(self) -> tuple[str, ...]:
        return tuple(
            feature_name
            for group in self.groups
            for feature_name in group.feature_names
        )


class RadiomicFeaturePreviewService:
    """Build a feature-name preview using MIRP settings without reading patient images."""

    def __init__(
        self,
        mirp_config: MirpConfig,
        metadata_config: DicomMetadataConfig | None = None,
        patient_count: int = 0,
    ) -> None:
        self.mirp_config = mirp_config
        self.metadata_config = metadata_config
        self.patient_count = patient_count

    def build(self) -> RadiomicFeaturePreview:
        from mirp._images.generic_image import GenericImage
        from mirp.settings.generic import SettingsClass

        settings = SettingsClass(**self.mirp_config.to_mirp_kwargs())
        image = GenericImage(
            image_data=None,
            image_modality="ct",
            image_spacing=tuple(self.mirp_config.voxel_spacing),
            image_dimensions=(1, 1, 1),
            separate_slices=self.mirp_config.by_slice,
        )

        groups: list[RadiomicFeaturePreviewGroup] = []
        base_feature_names = self._feature_names_for_settings(
            feature_settings=settings.feature_extr,
            image=image,
        )
        if base_feature_names:
            groups.append(
                RadiomicFeaturePreviewGroup(
                    source="original",
                    feature_families=tuple(settings.feature_extr.families),
                    feature_names=tuple(base_feature_names),
                    parameters=(
                        "discretisation_method=fixed_bin_size",
                        f"bin_width={self.mirp_config.bin_width}",
                    ),
                )
            )

        for transformed_image in self._transformed_images(settings=settings, image=image):
            feature_names = self._feature_names_for_settings(
                feature_settings=settings.img_transform.feature_settings,
                image=transformed_image,
            )
            if not feature_names:
                continue

            groups.append(
                RadiomicFeaturePreviewGroup(
                    source=self._format_source(transformed_image),
                    feature_families=tuple(settings.img_transform.feature_settings.families),
                    feature_names=tuple(feature_names),
                    parameters=self._format_parameters(transformed_image),
                )
            )

        return RadiomicFeaturePreview(
            patient_count=self.patient_count,
            groups=tuple(groups),
            metadata_columns=tuple(self._metadata_columns()),
        )

    def _metadata_columns(self) -> list[str]:
        if self.metadata_config is None:
            return []
        return [tag.output_column for tag in self.metadata_config.active_tags()]

    @staticmethod
    def _feature_names_for_settings(feature_settings: Any, image: Any) -> list[str]:
        from mirp._features.feature_generator import generate_features, feature_to_table

        features = list(generate_features(settings=feature_settings))
        if feature_settings.ibsi_compliant:
            features = [
                feature
                for feature in features
                if feature.is_ibsi_compliant(image=image)
            ]

        feature_table = feature_to_table(features)
        if feature_table is None:
            return []

        feature_table = image.parse_feature_names(feature_table)
        return [str(column) for column in feature_table.columns]

    def _transformed_images(self, settings: Any, image: Any) -> list[Any]:
        transformed_images: list[Any] = []
        for filter_kernel in settings.img_transform.spatial_filters or []:
            filter_object = self._build_filter_object(
                filter_kernel=filter_kernel,
                settings=settings,
                image=image,
            )
            transformed_images.extend(
                current_filter.transform(image=image)
                for current_filter in filter_object.generate_object()
            )
        return transformed_images

    @staticmethod
    def _build_filter_object(filter_kernel: str, settings: Any, image: Any) -> Any:
        if settings.img_transform.has_mean_filter(x=filter_kernel):
            from mirp._imagefilters.mean import MeanFilter

            return MeanFilter(image=image, settings=settings, name=filter_kernel)

        if settings.img_transform.has_laplacian_of_gaussian_filter(x=filter_kernel):
            from mirp._imagefilters.laplacian_of_gaussian import LaplacianOfGaussianFilter

            return LaplacianOfGaussianFilter(image=image, settings=settings, name=filter_kernel)

        if settings.img_transform.has_laws_filter(x=filter_kernel):
            from mirp._imagefilters.laws import LawsFilter

            return LawsFilter(image=image, settings=settings, name=filter_kernel)

        if settings.img_transform.has_gabor_filter(x=filter_kernel):
            from mirp._imagefilters.gabor import GaborFilter

            return GaborFilter(image=image, settings=settings, name=filter_kernel)

        if settings.img_transform.has_separable_wavelet_filter(x=filter_kernel):
            from mirp._imagefilters.separable_wavelet import SeparableWaveletFilter

            return SeparableWaveletFilter(image=image, settings=settings, name=filter_kernel)

        if settings.img_transform.has_nonseparable_wavelet_filter(x=filter_kernel):
            from mirp._imagefilters.nonseparable_wavelet import NonseparableWaveletFilter

            return NonseparableWaveletFilter(image=image, settings=settings, name=filter_kernel)

        raise ValueError(f"Filtro MIRP non supportato nella preview: {filter_kernel}")

    @staticmethod
    def _format_source(transformed_image: Any) -> str:
        attributes = transformed_image.get_export_attributes()
        filter_type = str(attributes.get("filter_type", "response_map"))

        descriptor = transformed_image.get_file_name_descriptor()
        if descriptor:
            return "_".join(str(item) for item in descriptor)
        return filter_type

    @staticmethod
    def _format_parameters(transformed_image: Any) -> tuple[str, ...]:
        attributes = transformed_image.get_export_attributes()
        hidden_keys = {"filter_type"}
        parameters: list[str] = []
        for key, value in attributes.items():
            if key in hidden_keys or value is None:
                continue
            parameters.append(f"{key}={value}")
        return tuple(parameters)
