from .base import BaseLoss, LossOutput

__all__ = [
	"BaseLoss",
	"LossOutput",
	"MSELoss",
	"GeomDistEDMLoss",
	"GeomDistUnsupervisedLoss",
	"GeomDistEDMWithAuxLoss",
	"GeomDistEDMWithContrastiveLoss",
	"GeomDistEDMWithAuxContrastiveLoss",
]


def __getattr__(name):
	if name == "MSELoss":
		from .mse import MSELoss
		return MSELoss
	if name == "GeomDistEDMLoss":
		from .geomdist import GeomDistEDMLoss
		return GeomDistEDMLoss
	if name == "GeomDistUnsupervisedLoss":
		from .geomdist_unsupervised import GeomDistUnsupervisedLoss
		return GeomDistUnsupervisedLoss
	if name == "GeomDistEDMWithAuxLoss":
		from .geomdist_aux import GeomDistEDMWithAuxLoss
		return GeomDistEDMWithAuxLoss
	if name == "GeomDistEDMWithContrastiveLoss":
		from .geomdist_contrastive import GeomDistEDMWithContrastiveLoss
		return GeomDistEDMWithContrastiveLoss
	if name == "GeomDistEDMWithAuxContrastiveLoss":
		from .geomdist_aux_contrastive import GeomDistEDMWithAuxContrastiveLoss
		return GeomDistEDMWithAuxContrastiveLoss
	raise AttributeError(f"module 'cogedi.losses' has no attribute '{name}'")
