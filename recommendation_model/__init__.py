from recommendation_model.config.core import Package_root

with open(Package_root / "VERSION") as version_file:
    __version__ = version_file.read().strip()

