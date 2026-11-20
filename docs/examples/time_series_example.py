# %%
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "colony_differentiation"))
from tools import *

image_folder = Path(__file__).resolve().parent / "image_series"
plate = Plate(folder=str(image_folder), number=1, background_interval=(0, 1.1))

# %%
detection_time = 20
plate.show(time=detection_time)

# %%
plate.find_candidates(time=detection_time, n=50, radius=10, sigma=2)

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
#     time=detection_time,
#     intensity=1000,)

# %%
plate.find_colonies(species, create_pdf=False)
print("Detected colonies:", plate.count())

# %%
plate.get_timeseries(step=1)


if plate.colonies[species] and plate.times:
    plate.plot_timeseries(species)
plt.show()