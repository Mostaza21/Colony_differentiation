# %%
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "colony_differentiation"))
from tools import *

image_path = Path(__file__).resolve().parent / "C.CR2"
plate = SingleImagePlate(str(image_path), name="Example plate")
plate.crop_rectangular_plate(interactive=True)
plate.find_candidates(n=50, radius=10, sigma=2)

# %%
sizes = plate.candidates.measure_sizes(max_radius=15)
size_counts = plate.candidates.classify_sizes(small_limit=10, large_limit=15)

# %%
plate.candidates.classify_colors()
plate.candidates.get_by_color("red")

# %%
clusters = plate.candidates.cluster_candidates(
    n_candidates=None, eps=20, min_samples=3, return_details=True)
plate.candidates.plot_clusters(clusters)

# %%
# Optional: Estimate species settings from a selected color cluster
# selected_ids = [candidate_id for candidate_id, label in
#                 zip(clusters["candidate_ids"], clusters["labels"]) if label == 0]
# if selected_ids:
#     species = plate.candidates.get_species(name="Example colonies", include=selected_ids)
#     print(species)

plt.show()
