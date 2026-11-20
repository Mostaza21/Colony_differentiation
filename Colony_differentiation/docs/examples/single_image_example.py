# %%
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "colony_differentiation"))
from tools import *

image_path = Path(__file__).resolve().parent / "C.CR2"

with rawpy.imread(str(image_path)) as raw:
    original_image = raw.postprocess()

show(original_image)

# %%
plate = SingleImagePlate(str(image_path), name="Example plate")

# Rectangular plate
plate.crop_rectangular_plate(interactive=True)

# Circular plate: automatic
# plate.crop_circular_plate()

# %%
# Circular plate: interactive
# circle = adjust_plate_circle(plate.rgb_image)

# %%
# crop_circular_plate(
#     plate.rgb_image,
#     center=circle.selected_center,
#     radius=circle.selected_radius,
#     plate=plate,)

# %%
# %%
plate.find_candidates(n=50, radius=10, sigma=2)

# %%
plate.candidates.measure_sizes()
plate.candidates.classify_sizes()
plate.candidates.cluster_candidates(n_candidates=None)
plate.candidates.plot_clusters()

# %%
# Option A: Estimate species settings
species = plate.candidates.get_species(name="Example colonies")
print(species)

# %%
# Option B: Define species settings manually
# species = Species(
#     name="Example colonies",
#     vector=np.array([1.0, 1.0, 1.0]),
#     weights=np.array([1.0, 2.0, 2.0]),
#     radius=10,
#     sigma=2,
#     time=0,
#     intensity=1000,)

# %%
plate.find_colonies(species, create_pdf=False)
print("Detected colonies:", plate.count())
plate.plot_statistics(species)
plt.show()