import rawpy
import numpy as np
from numpy.linalg import norm
import os
import re
import warnings
from matplotlib.pyplot import subplots, Circle
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter, gaussian_filter
from functools import lru_cache as cache
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
from tqdm import tqdm
from scipy.stats import variation
from termcolor import colored
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from itertools import islice
from IPython.display import display
import pickle
import cv2

MISS_FACTOR = 3
WHITE = np.array([0.31,0.94,0.12])
MAXSIZE = 14

class _array_with_hash(np.ndarray):
    """
    NumPy array that can be used as a key for cached calculations.
    """
    def __hash__(self):
        return hash((
            tuple(self.flat[::1337]),
            self.shape,
            str(self.dtype),
        ))

    def __eq__(self,other):
        return hash(self) == hash(other)

    @classmethod
    def from_array(cls,array):
        if not isinstance(array,np.ndarray):
            array = np.asarray(array)
        elif not array.data.contiguous:
            array = array.copy()
        return cls( array.shape, buffer=array, dtype=array.dtype )


def show(array,cspan=None,figsize=(12,6)):
    """
    Display an image or array with Matplotlib.

    Parameters
    ----------
    array
        Input NumPy array
    cspan
        Optional minimum and maximum values for the color scale
    figsize
        Matplotlib figure size in inches

    Returns
    -------
    tuple
        Matplotlib figure and axes
    """
    fig,axes = subplots(figsize=figsize)
    cspan = cspan or (np.quantile(array,0.01),np.quantile(array,0.99))
    image = axes.imshow( array, vmin=cspan[0], vmax=cspan[1] )
    if array.ndim==2:
        fig.colorbar(image)

    return fig,axes

@cache(maxsize=MAXSIZE)
def _smoothen(image,sigma=10):
    """
    Apply a Gaussian filter to an image.

    Parameters
    ----------
    image
        Input image array
    sigma
        Gaussian smoothing parameter

    Returns
    -------
    ndarray
        Smoothed image
    """
    result = _array_with_hash( shape=image.shape, dtype=float )
    gaussian_filter(
            image,
            sigma = sigma,
            mode = "nearest",
            axes = (0,1),
            output = result,
        )
    return result

@cache
def _base_neighbourhood(radius):
    """
    Get pixel offsets inside a circular neighborhood.

    Parameters
    ----------
    radius
        Radius in pixels

    Returns
    -------
    tuple of ndarray
        Row and column offsets
    """
    shifted_base = np.array([int(radius)+1,int(radius)+1])
    indices = np.where(norm(
            np.indices((int(3*radius),int(3*radius))) - shifted_base[:,None,None],
            axis=0,
        ) < radius)
    for index_set,base_index in zip(indices,shifted_base):
        index_set -= base_index
    return indices

@cache
def _neighbourhood(i,j,radius):
    """
    Get pixel indices around a given position.

    Parameters
    ----------
    i
        Image row coordinate
    j
        Image column coordinate
    radius
        Radius in pixels

    Returns
    -------
    tuple of ndarray
        Row and column indices
    """
    indices = _base_neighbourhood(radius)
    return ( indices[0]+i, indices[1]+j )

@cache
def _base_circle(radius):
    """
    Get pixel offsets along a circle.

    Parameters
    ----------
    radius
        Radius in pixels

    Returns
    -------
    tuple of ndarray
        Offsets along the circle
    """
    size = 2 * int(radius) + 3
    center = size // 2

    distances = norm(
    np.indices((size, size)) - center,
    axis=0,
    )
    indices = np.where(np.logical_and(radius-0.5<=distances,distances<radius+0.5))
    return (
    indices[0] - center,
    indices[1] - center
    )

circle = set(zip(*_base_circle(5)))
for i,j in circle:
    assert ( i,-j) in circle
    assert (-i,-j) in circle
    assert ( i, j) in circle
    assert ( j, i) in circle

@cache(maxsize=MAXSIZE)
def _get_intensities_and_angles(image,vector,sigma):
    projection = image @ vector
    intensities = _smoothen( projection, sigma )

    smoothed = _smoothen( image, sigma )
    with np.errstate(divide='ignore',invalid="ignore"):
        directions = smoothed / norm(smoothed,axis=-1)[:,:,None]
    angles = norm( directions - vector, axis=-1 )
    return intensities,angles

def _species_intensities_and_angles(image,species):
    """
    Calculate image intensities and color differences for a species.

    Parameters
    ----------
    image
        Input image array
    species
        Species settings used for detection

    Returns
    -------
    tuple of ndarray
        Intensity and angle arrays
    """
    return _get_intensities_and_angles(image,tuple(species.vector),species.sigma)

MIN_CIRCULARITY_BIN_SIZE = 10
@cache(maxsize=MAXSIZE)
def _get_circularities(image,radius):
    """
    Calculate a circularity score from color profiles around pixels.

    Parameters
    ----------
    image
        Input image array
    radius
        Radius in pixels

    Returns
    -------
    ndarray
        Circularity score for each image pixel
    """
    invstds = []
    values = []
    for r in range(radius):
        circle = _base_circle(r)
        k = len(circle[0])
        if k < MIN_CIRCULARITY_BIN_SIZE:
            continue
        value_sum = np.zeros_like(image)
        value_sqsum = np.zeros_like(image)
        for i,j in zip(*circle):
            circled = np.roll(image,(i,j),axis=(0,1))
            value_sum += circled
            value_sqsum += circled**2

        values.append(value_sum/k)
        invstds.append( 1/np.maximum( np.sqrt(value_sqsum/k - values[-1]**2), 1/WHITE ) )

    profile = np.std(values,axis=0)
    consistency = np.mean(invstds,axis=0)
    assert profile.shape[-1] == consistency.shape[-1] == 3
    return np.mean(profile*consistency,axis=-1)

@cache
def _footprint(radius):
    """
    Create a circular mask for finding local maxima.

    Parameters
    ----------
    radius
        Radius in pixels

    Returns
    -------
    ndarray
        Boolean circular mask
    """
    result = np.zeros([int(2*radius)-1,int(2*radius)-1],dtype=bool)
    result[_neighbourhood(radius-1,radius-1,radius)] = True
    return result

for i in range(5,20):
    test_foot = _footprint(i)
    assert np.all(np.any(test_foot,axis=0))
    assert np.all(np.any(test_foot,axis=1))

def _local_maxima(scores,radius):
    """
    Find local maxima in a score image.

    Parameters
    ----------
    scores
        Image containing detection scores
    radius
        Radius in pixels

    Returns
    -------
    iterator
        Row and column coordinates of local maxima
    """
    max_filter = maximum_filter(
            scores,
            footprint=_footprint(radius),
            mode='nearest'
        )

    for i,j in zip(*np.where(np.logical_and(
                np.equal(scores,max_filter),
                np.logical_not(np.isneginf(scores)),
            ))):
        yield int(i), int(j)

def _get_direction(i,j,smoothed):
    """
    Get the normalized color direction at a pixel.

    Parameters
    ----------
    i
        Image row coordinate
    j
        Image column coordinate
    smoothed
        Smoothed image array

    Returns
    -------
    ndarray
        Normalized color vector
    """
    direction = smoothed[i,j]
    direction /= norm(direction)
    return direction

dims = np.array([2*1738,2*2604],dtype=int)
a,b,c,d = 300,150,2450,1495
q,r = 1100, 2130
n,m = 1738, 2604

x_ind, y_ind = np.indices((c-a,d-b))
x_corner, y_corner = [],[]
for i,j in zip(x_ind.flat,y_ind.flat):
    if j>q+a-b+i or i>r+n-a-b-j:
        x_corner.append(j)
        y_corner.append(i)
corner_indices = (np.array(x_corner),np.array(y_corner))

def _transpose_if_needed(raw):
    """
    Rotate a raw image to the expected orientation.

    Parameters
    ----------
    raw
        Input image array

    Returns
    -------
    ndarray
        Image with corrected orientation
    """
    if raw.shape[0] > raw.shape[1]:
        raw = np.flip(raw.transpose((1,0,2)),0)
    return raw

@cache(maxsize=MAXSIZE)
def load_image(filename,background=None):
    """
    Load and preprocess a RAW image for analysis.

    Parameters
    ----------
    filename
        Path to a RAW image file
    background
        Optional background image to subtract

    Returns
    -------
    ndarray
        Preprocessed image used for analysis
    """
    rawfile = rawpy.imread(filename)
    raw = _transpose_if_needed(rawfile.raw_image_visible)

    image = _array_with_hash( shape=(d-b,c-a,3), dtype=float )
    image[:,:,0]  = raw[ 2*b  :2*d  :2, 2*a  :2*c  :2 ]
    image[:,:,1]  = raw[ 2*b  :2*d  :2, 2*a+1:2*c+1:2 ]
    image[:,:,1] += raw[ 2*b+1:2*d+1:2, 2*a  :2*c  :2 ]
    image[:,:,2]  = raw[ 2*b+1:2*d+1:2, 2*a+1:2*c+1:2 ]
    image /= WHITE

    if background is not None:
        image -= background
        image[corner_indices] = 0

    rawfile.close()

    return image

@cache(maxsize=MAXSIZE)
def load_RGB(filename):
    """
    Load a RAW image as an RGB image.

    Parameters
    ----------
    filename
        Path to a RAW image file

    Returns
    -------
    ndarray
        RGB image
    """
    rawfile = rawpy.imread(filename)
    rgb = _transpose_if_needed(rawfile.postprocess(half_size=True))[b:d,a:c]
    rgb[corner_indices] = 0
    rawfile.close()
    return rgb

def _is_close_to_border(j,i,R):
    """
    Check whether a position is too close to the image border.

    Parameters
    ----------
    j
        Image column coordinate
    i
        Image row coordinate
    R
        Distance from the border, in pixels

    Returns
    -------
    bool
        Whether the position is near the border
    """
    return i<R or j<R or i>c-a-R or j>d-b-R or j>q+a-b+i-R*np.sqrt(2) or i>r+n-a-b-j-R*np.sqrt(R)

@dataclass
class Species:
    """
    Settings used to detect a particular colony type.
    """
    vector: np.array = (1,1,1)
    sigma: float = 1
    radius: float = 7
    intensity: float = 1000
    time: float = 30
    factor: float = 2
    max_angle: float = 0.3
    min_circularity: float = 1
    weights: tuple[float,float,float] = (1,2,2)
    name: str = ""

    def __post_init__(self):
        self.vector /= norm(self.vector)
        self.weights /= np.sum(self.weights)

    def __hash__(self):
        components = []
        for value in asdict(self).values():
            try:
                components.append(tuple(value))
            except TypeError:
                components.append(value)
        return hash(tuple(components))

    def __eq__(self,other):
        return hash(self)==hash(other)

    def __repr__(self):
        result = ""
        result += f'{self.name} = Species(\n\tname = "{self.name}",\n'
        result += "\tvector = [" + ", ".join(f"{x:.2f}" for x in self.vector) + "],\n"
        for property in ["sigma","radius","intensity","time","factor","max_angle","min_circularity"]:
            value = self.__getattribute__(property)
            if isinstance(value,float):
                value = round(value,-int(np.log10(abs(value)))+2)
            if abs(value) > 100:
                value = int(value)
            result += f"\t{property} = {value},\n"
        result += "\tweights = [" + ", ".join(f"{w:.2f}" for w in self.weights) + "],\n"
        result += ")"
        return result

class Candidates(list):
    """
    Preliminary colony detections with independent characterization methods.

    The collection can be analyzed without a Species definition.

    Notes
    -----
    Candidate IDs are zero-based and remain unchanged when candidates are excluded
    """
    def __init__(self,plate,**kwargs):
        self.plate = plate
        self.kwargs = kwargs
        self.excluded = set()
        super().__init__()

    def exclude_candidates(self, numbers):
        """
        Exclude candidate IDs from further analysis.

        Parameters
        ----------
        numbers
            One candidate ID or a sequence of zero-based candidate IDs
        """
        if isinstance(numbers, (int, np.integer)):
            numbers = [numbers]
        numbers = list(numbers)
        if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
               or i < 0 or i >= len(self) for i in numbers):
            raise ValueError(f"Candidate IDs must be integers from 0 to {len(self)-1}.")
        self.excluded.update(int(i) for i in numbers)
        self._invalidate_analyses()
        print(f"Excluded candidates: {sorted(self.excluded)}")
        print(f"Remaining candidates: {len(self) - len(self.excluded)}")

    def restore_candidates(self, numbers=None):
        """
        Restore excluded candidates.

        Parameters
        ----------
        numbers
            One candidate ID or a sequence of zero-based candidate IDs
        """
        if numbers is None:
            self.excluded.clear()
        else:
            if isinstance(numbers, (int, np.integer)):
                numbers = [numbers]
            numbers = list(numbers)
            if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
                   or i < 0 or i >= len(self) for i in numbers):
                raise ValueError(f"Candidate IDs must be integers from 0 to {len(self)-1}.")
            self.excluded.difference_update(int(i) for i in numbers)
        self._invalidate_analyses()
        print(f"Remaining candidates: {len(self) - len(self.excluded)}")

    def _invalidate_analyses(self):
        for name in ("measured_sizes", "colors", "clustering_result"):
            if hasattr(self, name):
                delattr(self, name)

    def active_indices(self):
        """
        Return the IDs of candidates that are not excluded.

        Returns
        -------
        list of int
            Original IDs of active candidates
        """
        return [i for i in range(len(self)) if i not in self.excluded]

    def active_candidates(self):
        """
        Return the positions of candidates that are not excluded.

        Returns
        -------
        list of tuple
            Positions of active candidates
        """
        return [self[i] for i in self.active_indices()]

    def plot_candidates(self, start=0, count=100, q=30):
        """
        Show candidates with their IDs and excluded entries.

        Parameters
        ----------
        start
            Index of the first candidate to display
        count
            Maximum number of candidates to display
        q
            Half-width of the displayed image crop, in pixels

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the selected candidates
        """
        return plot_candidate_page(self.plate.rgb_image, self,
                                   getattr(self.plate, "candidate_intensities", None),
                                   start=start, count=count, q=q,
                                   excluded=self.excluded)

    def measure_sizes(self, max_radius=15):
        """
        Estimate colony diameters from connected image regions.

        Parameters
        ----------
        max_radius
            Maximum acceptable detected radius, in pixels

        Returns
        -------
        ndarray
            Estimated colony diameters in pixels
        """
        import cv2

        gray = cv2.cvtColor(self.plate.rgb_image, cv2.COLOR_RGB2GRAY)

        sizes = []

        for index, (y, x) in enumerate(self):
            if index in self.excluded:
                sizes.append(np.nan)
                continue
            y, x = int(y), int(x)

            y0 = max(0, y - max_radius)
            y1 = min(gray.shape[0], y + max_radius + 1)
            x0 = max(0, x - max_radius)
            x1 = min(gray.shape[1], x + max_radius + 1)

            patch = gray[y0:y1, x0:x1]
            cy, cx = y - y0, x - x0

            background = np.percentile(patch, 25)
            center = np.median(patch[max(0, cy-1):cy+2, max(0, cx-1):cx+2])

            threshold = (background + center) / 2

            binary = (patch > threshold).astype(np.uint8)

            count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

            component = labels[cy, cx]

            if component == 0:
                sizes.append(np.nan)
                continue

            area = stats[component, cv2.CC_STAT_AREA]


            diameter = 2 * np.sqrt(area / np.pi)
            sizes.append(float(diameter))

        self.measured_sizes = np.array(sizes)

        print(f"Measured colonies: {np.sum(~np.isnan(self.measured_sizes))}")
        if np.any(np.isfinite(self.measured_sizes)):
            print(f"Median diameter: {np.nanmedian(self.measured_sizes):.1f} px")

        return self.measured_sizes

    def classify_sizes(self, small_limit=10, large_limit=15):
        """
        Group measured colonies into small, medium and large.

        Parameters
        ----------
        small_limit
            Upper diameter limit for the small category, in pixels
        large_limit
            Lower diameter limit for the large category, in pixels

        Returns
        -------
        dict
            Counts for small, medium and large colonies
        """
        if not hasattr(self, "measured_sizes"):
            self.measure_sizes()

        sizes = self.measured_sizes
        valid_sizes = sizes[~np.isnan(sizes)]

        if len(valid_sizes) == 0:
            print("No valid size measurements available.")
            return {}

        small = np.sum(valid_sizes < small_limit)
        medium = np.sum((valid_sizes >= small_limit) & (valid_sizes < large_limit))
        large = np.sum(valid_sizes >= large_limit)

        total = len(valid_sizes)

        results = {
            "small": int(small),
            "medium": int(medium),
            "large": int(large)
        }

        print(f"Analyzed colonies: {total}")
        print()
        print(f"Small  (<{small_limit} px): {small} ({small/total*100:.1f} %)")
        print(f"Medium ({small_limit}–<{large_limit} px): {medium} ({medium/total*100:.1f} %)")
        print(f"Large  (≥{large_limit} px): {large} ({large/total*100:.1f} %)")


        import matplotlib.pyplot as plt

        indices = np.arange(len(sizes))
        valid = np.isfinite(sizes)

        small_mask = valid & (sizes < small_limit)
        medium_mask = valid & (sizes >= small_limit) & (sizes < large_limit)
        large_mask = valid & (sizes >= large_limit)

        categories = [
            ("Small", small_mask, "#55B8E8"),
            ("Medium", medium_mask, "#F2C14D"),
            ("Large", large_mask, "#EF7D54")
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1.3]})


        for name, mask, color in categories:
            ax1.scatter(
                indices[mask],
                sizes[mask],
                color=color,
                label=name,
                s=45,
                alpha=0.85
            )

        ax1.axhline(
            small_limit, color="gray",
            linestyle="--", alpha=0.5
        )
        ax1.axhline(
            large_limit, color="gray",
            linestyle="--", alpha=0.5
        )

        ax1.set_title("Colony Size Distribution")
        ax1.set_xlabel("Colony index")
        ax1.set_ylabel("Colony diameter (px)")
        ax1.legend()
        ax1.grid(alpha=0.2)


        names = ["Small", "Medium", "Large"]
        counts = [small, medium, large]

        percentages = [100 * count / total for count in counts]

        bars = ax2.bar(
            names,
            percentages,
            color=["#55B8E8", "#F2C14D", "#EF7D54"],
            width=0.65
        )

        for bar, count, percentage in zip(
            bars, counts, percentages
        ):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{percentage:.1f}%\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=10
            )

        ax2.set_title("Size Classes")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_ylim(0, max(percentages) + 15)
        ax2.grid(axis="y", alpha=0.2)
        ax2.set_axisbelow(True)

        plt.tight_layout()
        plt.show()

        return results





    def get_species(self, name, exclude=None, include=None):
        """
        Estimate Species settings from representative preliminary colonies.

        Select colonies representative of the target type; color clusters are not automatically converted into Species. This method connects exploratory 
        analysis with species-based detection. The returned Species can be used with find_colonies(), which performs a
        separate detection.

        Parameters
        ----------
        name
            Name assigned to the species
        exclude
            Candidate IDs to leave out of species estimation
        include
            Candidate IDs to use for species estimation

        Returns
        -------
        Species
            Estimated settings for the selected candidates
        """

        if exclude and include:
            raise ValueError("Either exclude or include must be None.")
        if exclude:
            candidates = [candidate for i,candidate in enumerate(self) if i not in exclude and i not in self.excluded]
        elif include:
            candidates = [candidate for i,candidate in enumerate(self) if i in include and i not in self.excluded]
        else:
            candidates = self.active_candidates()

        if not candidates:
            raise ValueError("No active candidates available for species estimation.")

        if isinstance(self.plate, SingleImagePlate):
            image = _array_with_hash.from_array(self.plate.image)
        else:
            detection_file = self.plate.first_file_after_time(self.kwargs["time"])
            image = load_image(detection_file, self.plate.background)

        smoothed = _smoothen(image, self.kwargs["sigma"])
        direction = np.median([
                _get_direction(i, j, smoothed)
                for i, j in candidates
            ], axis=0)


        species = Species(vector=direction,name=name,**self.kwargs)

        intensities,angles = _species_intensities_and_angles(image,species)
        species.max_angle = np.clip( float(1.5*np.quantile( [angles[i,j] for i,j in candidates], 0.8 )), 0.1, 1 )
        candidate_intensities = [ intensities[i,j] for i,j in candidates ]
        species.intensity = float(np.median(candidate_intensities))
        species.factor = min( 5, float(2*species.intensity/np.quantile(candidate_intensities,0.2)) )

        circularities = _get_circularities(image,self.kwargs["radius"])
        species.min_circularity = 0.5*min(circularities[i,j] for i,j in candidates)

        return species



    def classify_colors(self):
        """
        Sort active colonies into simple color categories.

        Notes
        -----
        Classification uses fixed RGB thresholds
        The results are stored in self.colors
        """
        colors = {
            "white" : [],
            "yellow" : [],
            "green" : [],
            "red" : [],
            "blue" : [],
            "unknown" : []
        }


        for y, x in self.active_candidates():
            patch = self.plate.rgb_image[y-2 : y+3, x-2 : x+3]
            rgb = np.median(patch, axis = (0,1))

            if max(rgb) - min(rgb) < 30 and min(rgb) > 150:
                colors["white"].append((y,x))

            elif rgb[0] > 180 and rgb[1] > 120 and rgb[2] < 50:
                colors["yellow"].append((y,x))

            elif abs(int(rgb[0]) - int(rgb[1])) < 20 and rgb[2] < 50:
                colors["green"].append((y,x))

            elif (rgb[0] > 1.35 * max(float(rgb[1]), float(rgb[2]), 1.0)
                  and rgb[0] - rgb[1] > 45
                  and rgb[0] - rgb[2] > 45):

                colors["red"].append((y,x))

            elif rgb[2]  > rgb[0] and rgb[2]  > rgb[1]:
                colors["blue"].append((y,x))

            else:
                colors["unknown"].append((y,x))

        self.colors = colors

        for color, positions in colors.items():
            print(color, len(positions))




    def get_by_color(self, color, q=30):
        """
        Display colonies assigned to a chosen color category.

        Parameters
        ----------
        color
            Color category to display
        q
            Half-width of the displayed image crop, in pixels

        Notes
        -----
        Call classify_colors before using this method
        """
        positions = self.colors[color]
        rgb_image = self.plate.rgb_image

        if len(positions) == 0:
            print(f"No {color} colonies found.")
            return

        m = min(10, len(positions))
        n = len(positions)

        fig, axes = subplots((n+m-1)//m, m, figsize=(2*m, 2.5*((n+m-1)//m)))

        for ax, (i, j) in zip(np.ravel(axes), positions):
            ax.imshow(rgb_image)
            ax.plot(j, i, "r+")
            ax.set_xlim(j-q, j+q)
            ax.set_ylim(i+q, i-q)
            ax.set_xticks([])
            ax.set_yticks([])


        for ax in np.ravel(axes)[len(positions):]:
            ax.axis("off")



    def cluster_candidates(self, n_candidates=100,
                           eps=20, min_samples=3, return_details=False,
                           verbose=True):
        """
        Cluster colonies by their median RGB color using DBSCAN.

        Parameters
        ----------
        n_candidates
            Maximum number of active candidates to cluster; None uses all
        eps
            Maximum RGB distance between neighboring DBSCAN samples
        min_samples
            Minimum number of samples needed for a DBSCAN core point
        return_details
            If True, return a dictionary with clustering information
        verbose
            If True, print a summary of the clusters

        Returns
        -------
        ndarray or dict
            Cluster labels or a details dictionary when return_details is True

        Notes
        -----
        Clustering uses median RGB values from a 5 × 5 pixel region around each candidate
        The results are stored in self.clustering_result
        """

        if eps <= 0:
            raise ValueError("eps must be positive.")
        if int(min_samples) != min_samples or min_samples < 1:
            raise ValueError("min_samples must be a positive integer.")
        if n_candidates is not None and (int(n_candidates) != n_candidates or n_candidates < 1):
            raise ValueError("n_candidates must be positive or None.")

        rgb_values = []
        positions = []
        candidate_ids = []
        rgb_image = np.asarray(self.plate.rgb_image)
        height, width = rgb_image.shape[:2]
        active_ids = self.active_indices()
        candidates = active_ids if n_candidates is None else active_ids[:int(n_candidates)]
        for candidate_id in candidates:
            y, x = self[candidate_id]
            y, x = int(y), int(x)
            patch = rgb_image[max(0, y-2):min(height, y+3),
                              max(0, x-2):min(width, x+3)]
            if patch.size == 0:
                continue
            rgb_values.append(np.median(patch, axis=(0, 1)))
            positions.append((y, x))
            candidate_ids.append(candidate_id)

        if rgb_values:
            rgb_array = np.asarray(rgb_values, dtype=float)
            labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(rgb_array)
        else:
            rgb_array = np.empty((0, 3), dtype=float)
            labels = np.empty(0, dtype=int)

        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)
                        if label != -1}
        noise_count = int(np.count_nonzero(labels == -1))
        details = {
            "positions": positions,
            "candidate_ids": candidate_ids,
            "rgb_values": rgb_array,
            "labels": labels,
            "n_clusters": len(cluster_counts),
            "cluster_counts": cluster_counts,
            "noise_count": noise_count,
            "n_candidates": len(positions),
            "eps": float(eps),
            "min_samples": int(min_samples),
        }
        self.clustering_result = details
        if verbose:
            print(f"Number of clusters: {details['n_clusters']}")
            print(f"Analyzed colonies: {len(positions)}")
            print(f"Outliers: {noise_count} colonies")
            for label, count in cluster_counts.items():
                print(f"Cluster {label}: {count} colonies")
        return details if return_details else labels


    def plot_clusters(self, result=None, figsize=(12, 7), marker_size=140,
                      show_numbers=True, show=True, full_image=False):
        """
        Plot the color clusters on the plate image.

        Parameters
        ----------
        result
            Optional result dictionary returned by cluster_candidates
        figsize
            Matplotlib figure size in inches
        marker_size
            Size of the markers in the cluster plot
        show_numbers
            If True, display the original candidate IDs
        show
            If True, display the figure
        full_image
            If True, plot the whole image instead of the selected area

        Returns
        -------
        tuple
            Matplotlib figure and axes

        Notes
        -----
        Run cluster_candidates first
        """
        from matplotlib.lines import Line2D
        from matplotlib import colors as mcolors

        if result is None:
            result = getattr(self, "clustering_result", None)
        if result is None or not isinstance(result, dict):
            raise ValueError("Run cluster_candidates() before plot_clusters().")

        positions = np.asarray(result["positions"], dtype=float).reshape(-1, 2)
        labels = np.asarray(result["labels"], dtype=int)
        if len(positions) != len(labels):
            raise ValueError("Positions and cluster labels have different lengths.")

        image = np.asarray(self.plate.rgb_image)
        height, width = image.shape[:2]
        x1, y1, x2, y2 = 0, 0, width, height
        if not full_image:
            mask = getattr(self.plate, "selection_mask", None)
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape == (height, width) and np.any(mask):
                    ys, xs = np.where(mask)
                    x1, x2 = int(xs.min()), int(xs.max()) + 1
                    y1, y2 = int(ys.min()), int(ys.max()) + 1
            elif getattr(self.plate, "cropped_bounds", None) is not None:
                bx1, by1, bx2, by2 = self.plate.cropped_bounds
                x1, y1 = max(0, int(bx1)), max(0, int(by1))
                x2, y2 = min(width, int(bx2)), min(height, int(by2))
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image[y1:y2, x1:x2], extent=(x1, x2, y2, y1))
        ax.set_xlim(x1, x2)
        ax.set_ylim(y2, y1)
        ax.set_title(f"Color clustering — {result['n_clusters']} clusters, "
                     f"{result['noise_count']} outliers")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        cluster_ids = sorted(set(labels) - {-1})
        palette = plt.get_cmap("tab20")
        cluster_colors = {cluster_id: palette(i % 20)
                          for i, cluster_id in enumerate(cluster_ids)}
        legend_handles = []
        for cluster_id in cluster_ids + ([-1] if -1 in labels else []):
            selected = labels == cluster_id
            color = "#b000f0" if cluster_id == -1 else cluster_colors[cluster_id]
            label = (f"Outliers ({np.sum(selected)})" if cluster_id == -1
                     else f"Cluster {cluster_id} ({np.sum(selected)})")
            ax.scatter(positions[selected, 1], positions[selected, 0],
                       s=marker_size, facecolors="none", edgecolors=[color],
                       linewidths=2.2, zorder=4)
            legend_handles.append(Line2D([0], [0], marker="o", linestyle="none",
                                         markerfacecolor="none", markeredgecolor=color,
                                         markeredgewidth=2, markersize=10, label=label))
            if show_numbers:
                for idx in np.flatnonzero(selected):
                    y, x = positions[idx]
                    ids = result.get("candidate_ids")
                    ax.annotate(str(ids[idx] if ids is not None else idx), (x, y), xytext=(6, -6),
                                textcoords="offset points", fontsize=9,
                                fontweight="bold", color=color, zorder=5)
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left",
                      bbox_to_anchor=(1.01, 1), borderaxespad=0)
            fig.tight_layout()
        if show:
            plt.show()
        return fig, ax


def _candidate_circularities_fast(image, radius, progress_callback=None):
    """
    Ring-profile score using OpenCV convolution instead of pixelwise rolls.

    Computes the same ring means, inverse standard deviations and their
    aggregate score as get_circularities, with float32 intermediates.
    """
    height, width = image.shape[:2]
    mean_sum = np.zeros((height, width, 3), dtype=np.float32)
    mean_sqsum = np.zeros_like(mean_sum)
    invstd_sum = np.zeros_like(mean_sum)
    count = 0
    radii = range(radius)
    for r in radii:
        offsets_y, offsets_x = _base_circle(r)
        k = len(offsets_y)
        if k >= MIN_CIRCULARITY_BIN_SIZE:
            size = 2 * r + 3
            kernel = np.zeros((size, size), dtype=np.float32)
            center = size // 2
            kernel[offsets_y + center, offsets_x + center] = 1.0 / k

            ring_mean = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
            ring_sqmean = cv2.filter2D(image * image, -1, kernel,
                                       borderType=cv2.BORDER_REPLICATE)
            variance = np.maximum(ring_sqmean - ring_mean * ring_mean, 0)
            std = np.sqrt(variance)
            invstd_sum += 1.0 / np.maximum(std, 1.0 / WHITE.astype(np.float32))
            mean_sum += ring_mean
            mean_sqsum += ring_mean * ring_mean
            count += 1
        if progress_callback is not None:
            progress_callback((r + 1) / max(1, radius))
    if count == 0:
        return np.zeros((height, width), dtype=np.float32)
    profile = np.sqrt(np.maximum(mean_sqsum / count - (mean_sum / count)**2, 0))
    return np.mean(profile * (invstd_sum / count), axis=-1)


class GeneralPlate:
    """
    Common methods for detecting and analyzing colonies on a plate.
    """
    def __init__( self, name, bad_regions=(), area_of_interest = None ):
        self.bad_regions = bad_regions
        self.area_of_interest = area_of_interest
        self.name = name
        self.clean()
        self._circulatities = {}

    def clean(self):
        """
        Clear stored detection and analysis results.
        """
        self.scores = {}
        self.angles = {}
        self.intensities = {}
        self.circularities = {}

        self.colonies = {}
        self.candidates = Candidates(plate=self)
        self.species_candidates = {}
        self.misses = {}
        self.colour_directions = {}

    def _clean_species(self,species):
        """
        Remove stored results for a species.

        Parameters
        ----------
        species
            Species settings used for detection
        """
        name = species.name if isinstance(species,Species) else species
        for other in list(self.intensities):
            if other.name == name:
                self.scores.pop(other,None)
                self.angles.pop(other,None)
                self.intensities.pop(other,None)
                self.circularities.pop(other,None)

                self.colonies.pop(other,None)
                self.species_candidates.pop(other,None)
                self.misses.pop(other,None)
                self.colour_directions.pop(other,None)

    def _is_in_bad_region(self,i,j):
        """
        Check whether a point is in an excluded circular region.

        Parameters
        ----------
        i
            Image row coordinate
        j
            Image column coordinate

        Returns
        -------
        bool
            Whether the position is in an excluded region
        """
        for k,l,radius in self.bad_regions:
            if np.hypot(i-k,j-l) < radius:
                return True
        else:
            return False

    def first_file_after_time(self,time):
        """
        Find the first image recorded after a given time.

        Parameters
        ----------
        time
            Time in hours used to select an image

        Returns
        -------
        str
            Filename of the first matching image
        """
        for file in self.files:
            if self.files[file]>time:
                return file

    def _compute_scores(self,image,species):
        """
        Calculate detection scores for a species.

        Parameters
        ----------
        image
            Input image array
        species
            Species settings used for detection
        """
        if species in self.scores:
            return

        S = species
        self.intensities[S], self.angles[S] = _species_intensities_and_angles(image,S)
        self.circularities[S] = _get_circularities(image,S.radius)

        with np.errstate(divide='ignore',invalid='ignore'):
            int_score = -np.abs(np.log(self.intensities[S]/S.intensity))
            ang_score = -np.log(self.angles[S])
            circ_score = np.log(self.circularities[S])

        result = sum([
                S.weights[0]*int_score,
                S.weights[1]*ang_score,
                S.weights[2]*circ_score,
            ])
        result[np.isnan(result)] = -np.inf

        self.scores[S] = result

    def _categorise_colony(self,i,j,species):
        """
        Classify a possible colony using species thresholds.

        Parameters
        ----------
        i
            Image row coordinate
        j
            Image column coordinate
        species
            Species settings used for detection

        Returns
        -------
        str
            One of good, missed or nothing
        """
        S = species
        angle_deviation = self.angles[S][i,j]/S.max_angle
        with np.errstate(divide='ignore'):
            intensity_deviation = np.abs(np.log(self.intensities[S][i,j]/S.intensity))
        circularity_deviation = S.min_circularity/self.circularities[S][i,j]

        thresh = np.log(S.factor)
        if (
                    intensity_deviation   <= thresh
                and angle_deviation       <= 1
                and circularity_deviation <= 1
            ):
            return "good"
        elif sum([
                S.weights[0]*np.log(max(intensity_deviation  ,thresh)),
                S.weights[1]*np.log(max(angle_deviation      ,1     )),
                S.weights[2]*np.log(max(circularity_deviation,1     )),
            ]) < np.log(MISS_FACTOR):
            return "missed"
        else:
            return "nothing"

    def _check_candidate(self,i,j,species):
        """
        Check whether a colony candidate should be accepted.

        Parameters
        ----------
        i
            Image row coordinate
        j
            Image column coordinate
        species
            Species settings used for detection

        Returns
        -------
        str
            Reason for accepting or rejecting the candidate
        """
        S = species
        if _is_close_to_border(i,j,S.radius):
            return "close to border"

        if self._is_in_bad_region(i,j):
            return "in bad region"

        if self.area_of_interest is not None:
            x_min, x_max, y_min, y_max = self.area_of_interest

            if not _area_of_interest(j, i, x_min, x_max, y_min, y_max):
                return "colony is not in area of interest"


        for k,l in self.species_candidates[species]:
            if (k,l) != (i,j):
                if np.hypot(i-k,j-l)<=S.radius:
                    if self.scores[S][i,j] < self.scores[S][k,l]:
                        return f"outscored: {k},{l}"

        for other_S,other_colonies in self.colonies.items():
            if other_S == S:
                break
            for k,l in other_colonies:
                if np.hypot(i-k,j-l) <= other_S.radius:
                    return f"existing {other_S.name}: {k},{l}"

        return "good"

    def find_colonies(self,species,create_pdf=True):
        """Detect colonies matching the given Species settings.

        This method performs a separate detection and does not require find_candidates() to be called first. Detected colonies are stored in self.colonies.

        Parameters
        ----------
        species
            Species settings used for detection
        create_pdf
            If True, save a PDF plot of the detected colonies
        """

        S = species
        self._clean_species(S)

        if isinstance(self, SingleImagePlate):
            detection_file = self.first_file_after_time(S.time)
            image = _array_with_hash.from_array(self.image)
        else:
            detection_file = self.first_file_after_time(S.time)
            image = load_image(detection_file, self.background)

        self._compute_scores(image, S)

        self.species_candidates[S] = set()
        self.misses[S] = set()



        detection_mask = np.ones(image.shape[:2], dtype=bool)
        if self.area_of_interest is not None:
            x_min, x_max, y_min, y_max = self.area_of_interest
            aoi_mask = np.zeros_like(detection_mask)
            aoi_mask[max(0, y_min):min(image.shape[0], y_max),
                     max(0, x_min):min(image.shape[1], x_max)] = True
            detection_mask &= aoi_mask

        selection_mask = getattr(self, "selection_mask", None)
        if selection_mask is not None:
            selection_mask = np.asarray(selection_mask, dtype=bool)
            if selection_mask.shape != detection_mask.shape:
                raise ValueError("Selection mask and image dimensions do not match.")


            radius = max(1, int(np.ceil(S.radius)))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            safe_selection = cv2.erode(
                selection_mask.astype(np.uint8), kernel
            ).astype(bool)
            detection_mask &= safe_selection

        filtered_scores = self.scores[S].copy()
        filtered_scores[~detection_mask] = -np.inf
        filtered_scores[self.circularities[S] < S.min_circularity] = -np.inf
        filtered_scores[self.angles[S] > S.max_angle] = -np.inf
        with np.errstate(divide='ignore',invalid="ignore"):
            intensity_deviation = np.abs(np.log(np.maximum(self.intensities[S],0)/S.intensity))
        filtered_scores[intensity_deviation>np.log(S.factor)] = -np.inf
        for i,j in _local_maxima(filtered_scores,S.radius):
            self.species_candidates[S].add((i,j))

        missed_scores = self.scores[S].copy()
        missed_scores[~detection_mask] = -np.inf
        for i,j in _local_maxima(missed_scores,S.radius):
            if self._categorise_colony(i,j,S) == "missed":
                self.misses[S].add((i,j))

        self.colonies[S] = set()
        for (i,j) in self.species_candidates[S]:
            match self._check_candidate(i,j,S):
                case "good": self.colonies[S].add((i,j))
                case "close to border": pass
                case _: self.misses[S].add((i,j))


        if self.colonies[S]:
            smoothed = _smoothen(image, S.sigma)
            direction = np.median([
                _get_direction(i, j, smoothed)
                for i, j in self.colonies[S]
            ], axis=0)

            length = norm(direction)
            direction = direction / length if length > 0 else np.zeros(3)
        else:
            direction = np.zeros(3)

        self.colour_directions[S] = direction

        if create_pdf:
            if isinstance(self, SingleImagePlate):
                rgb_image = self.rgb_image
            else:
                rgb_image = load_RGB(detection_file)

            fig, axes = show(rgb_image)

            for x,y in self.colonies[S]:
                axes.add_patch(Circle(
                    (y,x),
                    S.radius+1,
                    linewidth=0.7, alpha=0.7, color="red",
                    fill=False, zorder=2
                ))
            for x,y,radius in self.bad_regions:
                axes.add_patch(Circle(
                    (y,x),
                    radius,
                    linewidth=0.7, alpha=0.7, color="white",
                    fill=False, zorder=2
                ))
            for other_species in self.colonies:
                if other_species==S:
                    break
                for x,y in self.colonies[other_species]:
                    axes.plot(y,x,"b+",linewidth=0.3,alpha=0.5,markersize=5)
            with np.printoptions(precision=4):
                axes.set_title(direction)
            axes.set_xlim(0,image.shape[1])
            axes.set_ylim(image.shape[0],0)

            fig.suptitle(S.name)
            name = S.name.replace(" ","_")
            fig.savefig(f"Colonies_{self.name}_{name}.pdf",bbox_inches="tight")

    def show(self, time=20):
        """
        Display a plate image and its excluded regions.

        Parameters
        ----------
        time
            Time in hours used to select an image

        Returns
        -------
        tuple
            Matplotlib figure and axes
        """
        detection_file = self.first_file_after_time(time)

        if isinstance(self, SingleImagePlate):
            rgb_image = self.rgb_image
        else:
            rgb_image = load_RGB(detection_file)
        fig,axes = show(rgb_image)
        for x,y,radius in self.bad_regions:
            axes.add_patch(Circle(
                (y,x),
                radius,
                linewidth=0.7, alpha=0.7, color="white",
                fill=False, zorder=2
            ))
        axes.set_xlim(0,rgb_image.shape[1])
        axes.set_ylim(rgb_image.shape[0],0)
        return fig,axes

    def evaluate_around(self,i,j,species,q=50):
        """
        Inspect detection scores near a selected position.

        Parameters
        ----------
        i
            Image row coordinate
        j
            Image column coordinate
        species
            Species settings used for detection
        q
            Half-width of the displayed image crop, in pixels

        Returns
        -------
        tuple
            Matplotlib figure and axes
        """
        S = species

        detection_file = self.first_file_after_time(S.time)
        image = load_image(detection_file,self.background)
        self._compute_scores(image,S)
        smoothed = _smoothen(image,S.sigma)

        candidates = sorted(
                {(i,j),*self.colonies[S],*self.misses[S]},
                key = lambda x: np.hypot(x[0]-i,x[1]-j)
            )

        fig,axes = show( load_RGB(detection_file), figsize=(3,3) )
        axes.set_xlim(j-q,j+q)
        axes.set_ylim(i+q,i-q)

        print( "det.", "num.", "x", "y", "score", " int", "angle", "circ", "direction", sep="\t" )
        for n,(k,l) in enumerate(candidates):
            if abs(i-k)>q or abs(j-l)>q:
                break

            detected = (k,l) in self.colonies[S]
            axes.text(
                    l, k, n,
                    color = "blue" if detected else "red",
                    ha = "center", va = "center",
                )

            intensity = self.intensities[S][k,l]
            intensity_crit = S.intensity/S.factor < intensity < S.intensity*S.factor
            with np.printoptions(precision=2,sign=" "):
                print(
                        ("✔" if detected else "✘"),
                        n,
                        l, k,
                        f"{self.scores[S][k,l]: .3f}",
                        colored(
                                f"{int(intensity): 5d}",
                                None if S.intensity/S.factor < intensity < S.intensity*S.factor else "red",
                            ),
                        colored(
                                f"{self.angles[S][k,l]:.4f}",
                                None if self.angles[S][k,l] < S.max_angle else "red",
                            ),
                        colored(
                                f"{self.circularities[S][k,l]:.4f}",
                                None if self.circularities[S][k,l] > S.min_circularity else "red",
                            ),
                        _get_direction(k,l,smoothed),

                        self._check_candidate(k,l,S),
                        sep="\t"
                    )

        return fig,axes

    def plot_statistics(self,species):
        """
        Plot distributions of colony detection measurements.

        Parameters
        ----------
        species
            Species settings used for detection

        Returns
        -------
        tuple
            Matplotlib figure and axes
        """
        S = species
        fig, (axes_int,axes_ang,axes_circ) = subplots(1,3,figsize=(10,3),sharey=True)

        for measure,axes in [
                (self.intensities[S]  , axes_int ),
                (self.angles[S]       , axes_ang ),
                (self.circularities[S], axes_circ),
            ]:
            for colonies,colour,label in [
                    (self.misses[S]|self.colonies[S],"black"  ,"all"   ),
                    (self.colonies[S]               ,"blue"   ,"hits"  ),
                    (self.misses[S]                 ,"orange" ,"misses"),
                ]:
                values = [ measure[i,j] for (i,j) in colonies ]
                if values:
                    axes.ecdf(values,color=colour,label=label)
            axes.set_xscale("log")

        for factor in [1/S.factor,S.factor]:
            axes_int.axvline(factor*S.intensity,color="black",linewidth=0.5)
        axes_ang.axvline(S.max_angle,color="black",linewidth=0.5)
        axes_circ.axvline(S.min_circularity,color="black",linewidth=0.5)

        axes_int.set_xlabel("intensity")
        axes_ang.set_xlabel("angle")
        axes_circ.set_xlabel("circularity")
        axes_int.set_ylabel("CDF")
        axes_circ.legend()

        return fig, (axes_int,axes_ang,axes_circ)

    def get_timeseries(self,step=1,t_max=None):
        """
        Extract intensity and angle measurements over time.

        Parameters
        ----------
        step
            Use every step-th image in the time series
        t_max
            Optional maximum time to include, in hours

        Notes
        -----
        Run find_colonies for the required species before extracting the time series
        """
        for name in ["timeseries","misses_timeseries","angle_ts","angle_misses_ts"]:
            setattr(self,name,{
                S: defaultdict(lambda:[])
                for S in self.colonies
            })

        used_files = [
            (filename,time)
            for (filename,time) in list(self.files.items())[::step]
            if t_max is None or time <= t_max
        ]

        self.times = []
        for filename,time in tqdm(used_files):
            self.times.append(time)
            image = load_image(filename,self.background)

            for S in self.colonies:
                intensities,angles = _species_intensities_and_angles(image,S)
                for colonies,timeseries,angle_ts in [
                        ( self.colonies, self.timeseries       , self.angle_ts        ),
                        ( self.misses  , self.misses_timeseries, self.angle_misses_ts ),
                    ]:
                    for colony in colonies[S]:
                        timeseries[S][colony].append(intensities[*colony])
                        angle_ts[S][colony].append(angles[*colony])

    def plot_timeseries(self,species):
        """
        Plot the time series for a detected species.

        Parameters
        ----------
        species
            Species settings used for detection

        Returns
        -------
        tuple
            Matplotlib figure and axes

        Notes
        -----
        Call get_timeseries before plotting
        """
        S = species
        fig, (int_axes,ang_axes) = subplots(2,1,sharex=True)
        for timeseries,angle_ts,colour,label in [
                (self.misses_timeseries,self.angle_misses_ts,"orange","misses"),
                (self.timeseries       ,self.angle_ts       ,"blue"  ,"hits"  ),
            ]:
            first = True
            for ts in timeseries[S].values():
                ts = np.array(ts)
                ts[ts<0] = np.nan
                int_axes.plot(
                    self.times,
                    ts,
                    color=colour,
                    linewidth = 0.5,
                    alpha=0.5,
                    label = (label if first else None)
                )
                first = False
            for ts in angle_ts[S].values():
                ang_axes.plot(
                    self.times,
                    ts,
                    color=colour,
                    linewidth = 0.5
                )

        int_axes.set_yscale("log")
        int_axes.set_ylim(1e2,None)
        int_axes.axvline(S.time,color="black",linewidth=0.5)
        for factor in [1/S.factor,S.factor]:
            int_axes.axhline(factor*S.intensity,color="black",linewidth=0.5)
        int_axes.set_ylabel("intensity")
        int_axes.legend(loc="upper left")

        max_plotted_angle = max( max(ts) for ts in angle_ts[S].values() )
        ang_axes.set_ylim( 0, min(MISS_FACTOR*1.3*S.max_angle,1.1*max_plotted_angle) )
        ang_axes.axvline(S.time,color="black",linewidth=0.5)
        ang_axes.axhline(S.max_angle,color="black",linewidth=0.5)
        ang_axes.set_ylabel("angle")
        ang_axes.set_xlabel("time")
        ang_axes.set_xlim(min(self.times),max(self.times))

        int_axes.set_title(S.name)

        return fig,(int_axes,ang_axes)

    def find_candidates(self, time=20, n=50, radius=10,
                        threshold=None, q=30, sigma=2,
                        progress_callback=None, **kwargs):


        """
        Detect preliminary colonies without a Species definition.

        The preliminary colony list is stored in self.candidates. It can be analyzed independently by size and color or used to estimate Species settings with
        get_species(). Species-based detection does not require this step.

        Parameters
        ----------
        time
            Time in hours used to select an image
        n
            Maximum number of candidates to keep; None keeps all accepted candidates
        radius
            Radius in pixels
        threshold
            Minimum intensity required for a candidate peak
        q
            Half-width of the displayed image crop, in pixels
        sigma
            Gaussian smoothing parameter
        progress_callback
            Optional function receiving progress values from 0 to 100
        **kwargs : optional
            Additional settings stored for subsequent species estimation

        Returns
        -------
        matplotlib.figure.Figure
            Figure showing the detected candidates

        Notes
        -----
        The number of detected colonies may be smaller than n. Using n=None
        removes the count limit but does not guarantee complete detection.
        """

        if n is not None and (not isinstance(n, (int, np.integer)) or n < 1):
            raise ValueError("n must be a positive integer or None.")
        radius = int(radius)
        if radius < 1:
            raise ValueError("radius must be positive.")
        kwargs = dict(time=time, radius=radius, sigma=sigma, **kwargs)
        last_progress = -1

        def report(value):
            nonlocal last_progress
            value = max(0, min(100, int(value)))
            if progress_callback is not None and value != last_progress:
                progress_callback(value)
                last_progress = value

        report(0)
        if isinstance(self, SingleImagePlate):
            image = self.image
            rgb_image = self.rgb_image
        else:
            detection_file = self.first_file_after_time(time)
            image = load_image(detection_file, self.background)
            rgb_image = load_RGB(detection_file)

        x_offset = y_offset = 0
        if self.area_of_interest is not None:
            x_min, x_max, y_min, y_max = self.area_of_interest
            x_offset, y_offset = x_min, y_min
            image = image[y_min:y_max, x_min:x_max]

        report(8)
        image = np.ascontiguousarray(image, dtype=np.float32)


        circularities = _candidate_circularities_fast(
            image, radius, progress_callback=lambda fraction: report(8 + 57 * fraction)
        )
        report(66)
        normed = np.linalg.norm(image, axis=-1)
        intensities = cv2.GaussianBlur(
            normed, (0, 0), sigmaX=max(float(sigma), 0.01)
        ) if sigma else normed
        scores = circularities * intensities
        del circularities, normed
        report(72)

        mask = getattr(self, "selection_mask", None)
        safe_mask = None
        if mask is not None:
            if self.area_of_interest is not None:
                mask = mask[y_min:y_max, x_min:x_max]
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            safe_mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
            if safe_mask.shape != scores.shape:
                raise ValueError("Mask and image dimensions do not match.")
            scores[~safe_mask] = -np.inf

        if threshold is None:
            if isinstance(self, SingleImagePlate) and (
                not self.files or
                str(self.first_file_after_time(time)).lower().endswith('.png')
            ):
                threshold = 0
            else:
                threshold = np.quantile(intensities, 0.99)

        is_single = isinstance(self, SingleImagePlate)
        file_name = str(self.first_file_after_time(time)).lower() if is_single else ""
        min_contrast = -1000 if is_single and (
            not self.files or file_name.endswith((".jpg", ".jpeg", ".png"))
        ) else 1000



        peak_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        finite = np.isfinite(scores)
        scores[~finite] = -np.inf
        maxima = cv2.dilate(scores, peak_kernel)
        peaks = finite & (scores >= maxima) & (intensities > threshold)
        margin = max(radius, 2 * radius if is_single and not self.files else radius)
        peaks[:margin, :] = False
        peaks[-margin:, :] = False
        peaks[:, :margin] = False
        peaks[:, -margin:] = False
        if safe_mask is not None:
            peaks &= safe_mask
        yy, xx = np.nonzero(peaks)
        del maxima, peaks, finite
        report(80)
        order = np.argsort(scores[yy, xx])[::-1]
        yy, xx = yy[order], xx[order]
        del order, scores

        candidates = Candidates(plate=self, **kwargs)
        bucket_size = max(1, 2 * radius)
        buckets = {}
        total = len(yy)
        update_interval = max(1, total // 100)
        for index, (y, x) in enumerate(zip(yy, xx)):
            if index % update_interval == 0:
                report(80 + 15 * index / max(1, total))
            i, j = int(y + y_offset), int(x + x_offset)
            if self._is_in_bad_region(i, j):
                continue
            center = float(np.mean(image[y-2:y+3, x-2:x+3]))
            surroundings = float(np.mean(image[
                y-radius:y+radius+1, x-radius:x+radius+1
            ]))
            if center - surroundings <= min_contrast:
                continue
            if any(
                np.hypot(i-k, j-l) <= other_S.radius + radius
                for other_S, other_colonies in self.colonies.items()
                for k, l in other_colonies
            ):
                continue
            by, bx = i // bucket_size, j // bucket_size
            if any(
                (i-k)**2 + (j-l)**2 <= (2 * radius)**2
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                for k, l in buckets.get((by+dy, bx+dx), ())
            ):
                continue
            candidates.append((i, j))
            buckets.setdefault((by, bx), []).append((i, j))
            if n is not None and len(candidates) >= n:
                break

        self.candidates = candidates
        self.candidate_intensities = [
            int(intensities[i-y_offset, j-x_offset]) for i, j in candidates
        ]
        report(96)
        fig = plot_candidate_page(
            rgb_image=rgb_image, positions=self.candidates,
            intensities=self.candidate_intensities, start=0, count=100, q=q,
            excluded=candidates.excluded
        )
        report(100)
        return fig

    def count(self):
        """
        Count detected colonies for each species name.

        Returns
        -------
        dict
            Number of colonies for each species name
        """
        result = defaultdict(lambda:0)
        for species,colonies in self.colonies.items():
            name = species.name.partition(".")[0]
            result[name] += len(colonies)
        return dict(result)

    def save_counts(self):
        """
        Save colony counts to a pickle file.

        Returns
        -------
        dict
            Saved colony counts
        """
        counts = self.count()
        pickle.dump( counts, open(f"counts_2/{self.name}.pickle","wb") )
        return counts



def plot_candidate_page(rgb_image, positions, intensities=None, start=0, count=100, q=30, excluded=None):
    """
    Show detected candidates as a page of small image crops.

    Parameters
    ----------
    rgb_image
        Input RGB image array
    positions
        Candidate positions as (row, column) pairs
    intensities
        Optional intensity values for candidate labels
    start
        Index of the first candidate to display
    count
        Maximum number of candidates to display
    q
        Half-width of the displayed image crop, in pixels
    excluded
        Optional set of excluded candidate IDs


        Returns
        -------
        matplotlib.figure.Figure
            Figure showing the selected candidates
        """

    from matplotlib import pyplot as plt

    excluded = set() if excluded is None else set(excluded)
    page = positions[start:start + count]
    if not page:
        return None
    columns = 10
    rows = (len(page) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(10, 1.3 * rows),
        squeeze=False
    )

    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.94,
        bottom=0.06,
        hspace=0.45,
        wspace=0.15
    )

    fig.set_layout_engine(None)
    height, width = rgb_image.shape[:2]
    for offset, (i, j) in enumerate(page):
        ax = axes.flat[offset]
        i, j = int(i), int(j)
        y0, y1 = max(0, i-q), min(height, i+q+1)
        x0, x1 = max(0, j-q), min(width, j+q+1)
        ax.imshow(rgb_image[y0:y1, x0:x1], aspect="equal",
                  extent=(x0, x1, y1, y0))
        ax.plot(j, i, "r+", markersize=5, markeredgewidth=1.2)
        ax.set_xlim(j - q, j + q)
        ax.set_ylim(i + q, i - q)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        candidate_id = start + offset
        label = f"{candidate_id}"
        if candidate_id in excluded:
            ax.plot([j-q*0.7, j+q*0.7], [i-q*0.7, i+q*0.7],
                    color="red", linewidth=2.5)
            ax.plot([j-q*0.7, j+q*0.7], [i+q*0.7, i-q*0.7],
                    color="red", linewidth=2.5)
            label += " EXCLUDED"
        if intensities is not None and start+offset < len(intensities):
            label += f" | {intensities[start+offset]}"
        ax.set_title(label, fontsize=8, pad=3)
    for ax in axes.flat[len(page):]:
        ax.axis('off')
    return fig

def _area_of_interest(x, y, x_min, x_max, y_min, y_max):
    """
    Check whether a point is inside rectangular bounds.

    Parameters
    ----------
    x
        Horizontal image coordinate
    y
        Vertical image coordinate
    x_min
        Minimum horizontal coordinate
    x_max
        Maximum horizontal coordinate
    y_min
        Minimum vertical coordinate
    y_max
        Maximum vertical coordinate

    Returns
    -------
    bool
        Whether the point lies within the bounds
    """
    return(x_min <= x <= x_max) and (y_min <= y <= y_max)


def _show_area_of_interest(image, x_min, x_max, y_min, y_max):
    """
    Display a rectangular part of an image.

    Parameters
    ----------
    image
        Input image array
    x_min
        Minimum horizontal coordinate
    x_max
        Maximum horizontal coordinate
    y_min
        Minimum vertical coordinate
    y_max
        Maximum vertical coordinate
    """
    show(image[y_min : y_max, x_min : x_max])


class Plate(GeneralPlate):
    """
    A plate represented by a time series of RAW images.

    Acquisition times are read from image metadata. When unavailable, timestamps
    in filenames of the form ``YYYY-MM-DD_HH-MM-SS`` are used as a fallback.
    Images must belong to the same plate; use ``number`` to filter numbered
    filenames when a folder contains multiple plates.
    """

    def __init__(self, folder, number=None, background_interval=(1, 4),
                 bad_regions=(), area_of_interest=None):
        self._find_files(folder, number)
        self._load_background(background_interval)
        super().__init__(f"{number:02d}" if number is not None else "Time series",
                         bad_regions, area_of_interest)

    def _load_background(self, background_interval):
        """Calculate a background image from RAW images in a time interval (hours)."""
        backgrounds = [
            _smoothen(load_image(file), 5)
            for file, time in self.files.items()
            if background_interval[0] <= time <= background_interval[1]
        ]
        if not backgrounds:
            raise ValueError(
                f"No images in background_interval={background_interval}. "
                "Choose an interval covered by the available acquisition times."
            )
        self.background = _array_with_hash(backgrounds[0].shape,
                                           dtype=backgrounds[0].dtype)
        np.median(backgrounds, out=self.background, axis=0)

    @staticmethod
    def _acquisition_time(path):
        """Read the acquisition datetime from RAW metadata, with filename fallback."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Reading RAW timestamps requires Pillow: pip install Pillow"
            ) from exc

        try:
            with Image.open(path) as image:
                exif = image.getexif()


                timestamp = exif.get(36867) or exif.get(306)
                if timestamp:
                    return datetime.strptime(str(timestamp), "%Y:%m:%d %H:%M:%S")
        except (OSError, ValueError, TypeError):
            pass

        match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})",
                          os.path.basename(path))
        if match:
            warnings.warn(f"No readable EXIF time for {path}; using filename timestamp.",
                          stacklevel=2)
            return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
        raise ValueError(
            f"No acquisition timestamp found in RAW metadata or filename: {path}"
        )

    def _find_files(self, folder, number):
        """
        Find RAW files, optionally filter by plate number, and order by EXIF time.

        Parameters
        ----------
        folder : str or path-like
            Directory containing the RAW time series.
        number : int, optional
            Restrict to files beginning with this plate number followed by ``_``.
            Omit for a directory containing only one plate.
        """
        folder = os.fspath(folder)
        if not os.path.isdir(folder):
            raise NotADirectoryError(f"Image folder not found: {folder}")

        candidates = []
        prefixes = set()
        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith((".cr2", ".cr3", ".nef", ".arw", ".dng")):
                continue
            prefix = re.match(r"^(\d+)_", filename)
            if prefix:
                prefixes.add(int(prefix.group(1)))
            if number is not None and (not prefix or int(prefix.group(1)) != number):
                continue
            candidates.append(os.path.join(folder, filename))

        if number is None and len(prefixes) > 1:
            raise ValueError(
                "Multiple numbered plates found in the folder. "
                "Specify number=... or separate the images into folders."
            )
        if not candidates:
            raise ValueError(f"No matching RAW images found in {folder} "
                             f"for plate number {number!r}.")

        dated = sorted(((self._acquisition_time(file), file) for file in candidates),
                       key=lambda item: (item[0], item[1]))
        unique = []
        seen = set()
        for timestamp, file in dated:
            if timestamp in seen:
                warnings.warn(
                    f"Duplicate acquisition timestamp {timestamp} in {file}; "
                    "skipping this image.", stacklevel=2
                )
                continue
            seen.add(timestamp)
            unique.append((timestamp, file))

        start = unique[0][0]
        self.files = {
            file: (timestamp - start).total_seconds() / 3600
            for timestamp, file in unique
        }


class SingleImagePlate(GeneralPlate):
    """
    A plate represented by one image or a NumPy array.

    Notes
    -----
    PNG files, RAW images and NumPy image arrays are supported
    """
    def __init__(self, filename, name, background=None, bad_regions=(),bg_smoothing=9, area_of_interest = None, mask=None):
        self.files = {} if isinstance(filename, np.ndarray) else {filename: 0}
        super().__init__(name, bad_regions, area_of_interest)
        self.selection_mask = mask

        if isinstance(filename, np.ndarray):
            self.rgb_image = filename.copy()
            self.image = filename.astype(float)

            if background is None:
                self.background = gaussian_filter(
                    self.image,
                    sigma=(bg_smoothing, bg_smoothing, 0)
                )
            else:
                self.background = background

            self.image = self.image - self.background

        elif isinstance(filename, str) and filename.lower().endswith(".png"):
            self.rgb_image = cv2.cvtColor(
                cv2.imread(filename),
                cv2.COLOR_BGR2RGB
            )

            self.image = self.rgb_image.astype(float)

            if background is None:
                self.background = gaussian_filter(
                    self.image,
                    sigma=(bg_smoothing, bg_smoothing, 0)
                )
            else:
                self.background = background

            self.image = self.image - self.background

        else:
            self.rgb_image = load_RGB(filename)

            if background is None:
                self.background = _smoothen(load_image(filename), bg_smoothing)
            else:
                self.background = background

            self.image = load_image(filename, self.background)


    def crop_circular_plate(self, start_angle=None, end_angle=None, interactive=False):
        """
        Crop the circular area of this image.

        Parameters
        ----------
        start_angle
            Start angle of the selected plate sector, in degrees
        end_angle
            End angle of the selected plate sector, in degrees
        interactive
            If True, enable manual selection in a Jupyter notebook
        """
        return crop_circular_plate(
            self.rgb_image,
            start_angle=start_angle,
            end_angle=end_angle,
            interactive=interactive,
            plate=self
        )

    def crop_rectangular_plate(self, bounds=None, show_plot=False,
                               manual_polygon=None, interactive=False):
        """
        Crop the rectangular agar area of this image.

        Parameters
        ----------
        bounds
            Optional rectangular bounds as (x_min, y_min, x_max, y_max)
        show_plot
            If True, show a plot of the selected area
        manual_polygon
            Optional polygon vertices as (x, y) points
        interactive
            If True, enable manual selection in a Jupyter notebook
        """
        source = self.rgb_image
        origin = (0, 0)
        if self.files:
            filename = next(iter(self.files))
            if not str(filename).lower().endswith(".png"):
                with rawpy.imread(filename) as raw:
                    source = _transpose_if_needed(raw.postprocess(half_size=True))
                origin = (a, b)
                if bounds is not None:
                    bounds = tuple(np.asarray(bounds) + [a, b, a, b])
                if manual_polygon is not None:
                    manual_polygon = np.asarray(manual_polygon, dtype=float) + [a, b]

        return crop_rectangular_plate(
            source, bounds=bounds, plate=self, show_plot=show_plot,
            manual_polygon=manual_polygon, interactive=interactive,
            _analysis_origin=origin)

    def select_rectangular_aoi(self, use_cropped=False, max_width=720,
                               max_height=440):
        """
        Select a rectangular area of interest with the mouse.

        Parameters
        ----------
        use_cropped
            If True, select within the previously cropped image
        max_width
            Maximum preview width in pixels
        max_height
            Maximum preview height in pixels

        Notes
        -----
        Interactive selection requires a Jupyter Matplotlib widget backend
        """
        from matplotlib.widgets import RectangleSelector
        from IPython.display import display, clear_output
        from PIL import Image as PILImage
        import ipywidgets as widgets

        if use_cropped:
            if not hasattr(self, "cropped_image"):
                raise ValueError("No cropped image available. Omit use_cropped=True to select on the original.")
            source = np.asarray(self.cropped_image)
            offset_x, offset_y = map(int, self.cropped_bounds[:2])
        elif self.files:
            filename = next(iter(self.files))
            if str(filename).lower().endswith((".cr2", ".cr3", ".nef", ".arw", ".dng")):


                with rawpy.imread(filename) as raw:
                    source = _transpose_if_needed(raw.postprocess(half_size=True))
                offset_x, offset_y = -a, -b
            else:
                source = np.asarray(PILImage.open(filename).convert("RGB"))
                offset_x = offset_y = 0
        else:
            source = np.asarray(self.rgb_image)
            offset_x = offset_y = 0

        height, width = source.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        preview_width = max(1, round(width * scale))
        preview_height = max(1, round(height * scale))
        preview = cv2.resize(source, (preview_width, preview_height),
                             interpolation=cv2.INTER_AREA) if scale < 1 else source



        with plt.ioff():
            fig, ax = plt.subplots(figsize=(preview_width / 120, preview_height / 120),
                                   dpi=120)
        ax.imshow(preview)
        ax.set_title("Drag to select an area of interest", fontsize=10)
        ax.set_axis_off()
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.89)

        output = widgets.Output()
        state = {"bounds": None, "image": None}

        def on_select(start, end):
            if any(v is None for v in (start.xdata, start.ydata,
                                       end.xdata, end.ydata)):
                return
            px1, px2 = sorted((start.xdata, end.xdata))
            py1, py2 = sorted((start.ydata, end.ydata))
            x1 = int(np.clip(np.floor(px1 / scale), 0, width))
            x2 = int(np.clip(np.ceil(px2 / scale), 0, width))
            y1 = int(np.clip(np.floor(py1 / scale), 0, height))
            y2 = int(np.clip(np.ceil(py2 / scale), 0, height))
            if x2 <= x1 or y2 <= y1:
                return

            selected = source[y1:y2, x1:x2].copy()
            self.aoi_cropped_image = selected
            self.aoi_bounds = (x1, y1, x2, y2)
            state["bounds"] = self.aoi_bounds
            state["image"] = selected

            image_h, image_w = self.rgb_image.shape[:2]
            bx1 = int(np.clip(x1 + offset_x, 0, image_w))
            bx2 = int(np.clip(x2 + offset_x, 0, image_w))
            by1 = int(np.clip(y1 + offset_y, 0, image_h))
            by2 = int(np.clip(y2 + offset_y, 0, image_h))
            selection = np.zeros((image_h, image_w), dtype=bool)
            if bx2 > bx1 and by2 > by1:
                selection[by1:by2, bx1:bx2] = True
            if use_cropped and getattr(self, "cropped_mask", None) is not None:
                existing = np.zeros_like(selection)
                cx1, cy1, cx2, cy2 = self.cropped_bounds
                existing[cy1:cy2, cx1:cx2] = self.cropped_mask
                selection &= existing
            self.selection_mask = selection

            with output:
                clear_output(wait=True)
                out_h, out_w = selected.shape[:2]
                ratio = min(720 / out_w, 440 / out_h, 1.0)
                shown = selected if ratio == 1 else cv2.resize(
                    selected, (max(1, round(out_w * ratio)),
                               max(1, round(out_h * ratio))),
                    interpolation=cv2.INTER_AREA)
                display(PILImage.fromarray(shown.astype(np.uint8)))

        selector = RectangleSelector(
            ax, on_select, useblit=False, button=[1],
            interactive=True, drag_from_anywhere=True
        )
        self._rect_aoi_ui = {"figure": fig, "selector": selector,
                             "output": output, "state": state}

        display(fig.canvas)
        display(output)
        return None

    def show_area_of_interest(self):
        """
        Display the current area of interest.
        """
        if self.area_of_interest is None:
            show(self.rgb_image)
            return

        x_min, x_max, y_min, y_max = self.area_of_interest
        show(self.rgb_image[y_min:y_max, x_min:x_max])

    def first_file_after_time(self, time):
        """
        Return the filename associated with this single image.

        Parameters
        ----------
        time
            Time in hours used to select an image

        Returns
        -------
        str
            Image filename
        """
        return list(self.files.keys())[0] if self.files else None

    def __getitem__(self, index):
        return self.image.__getitem__(index)


def adjust_plate_circle(rgb_image, center=None, radius=None):
    """
    Adjust the detected plate circle interactively.

    Parameters
    ----------
    rgb_image
        Input RGB image array
    center
        Optional plate center as (x, y)
    radius
        Radius in pixels

    Returns
    -------
    matplotlib.patches.Circle
        Circle artist with controls for adjusting the crop
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.widgets import Slider, Button
    from IPython.display import display
    from PIL import Image
    import ipywidgets as widgets

    if center is None or radius is None:
        result = detect_plate_circle(rgb_image)

        if result is None:
            raise ValueError("No agar plate detected.")

        detected_x, detected_y, detected_radius = result

        if center is None:
            center = (detected_x, detected_y)

        if radius is None:
            radius = detected_radius

    x, y = center

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(
        left=0.15,
        right=0.95,
        bottom=0.25,
        top=0.95
    )

    ax.imshow(rgb_image)
    ax.set_title("Adjust Agar Plate Circle")

    circle = Circle(
        (x, y),
        radius,
        fill=False,
        edgecolor="red",
        linewidth=2
    )
    ax.add_patch(circle)

    slider_ax = fig.add_axes((0.2, 0.12, 0.6, 0.03))
    radius_slider = Slider(
        slider_ax,
        "Radius",
        1,
        max(rgb_image.shape[:2]),
        valinit=radius
    )

    reset_ax = fig.add_axes((0.25, 0.04, 0.2, 0.05))
    reset_button = Button(reset_ax, "Reset")

    apply_ax = fig.add_axes((0.55, 0.04, 0.2, 0.05))
    apply_button = Button(apply_ax, "Apply & Crop")

    output = widgets.Output()
    dragging = False

    def on_press(event):
        nonlocal dragging

        if event.inaxes == ax and event.xdata is not None:
            dragging = True
            circle.center = (event.xdata, event.ydata)
            fig.canvas.draw_idle()

    def on_move(event):
        if dragging and event.inaxes == ax and event.xdata is not None:
            circle.center = (event.xdata, event.ydata)
            fig.canvas.draw_idle()

    def on_release(event):
        nonlocal dragging
        dragging = False

    def update_radius(value):
        circle.radius = value
        fig.canvas.draw_idle()

    def reset(event):
        circle.center = (x, y)
        radius_slider.reset()
        fig.canvas.draw_idle()

    def apply_crop(event):
        selected_center = tuple(map(round, circle.center))
        selected_radius = round(circle.radius)

        circle.selected_center = selected_center
        circle.selected_radius = selected_radius

        with output:
            output.clear_output(wait=True)

            try:
                cropped_image = crop_circular_plate(
                    rgb_image,
                    center=selected_center,
                    radius=selected_radius,
                    show_plot=False
                )

                display(
                    Image.fromarray(
                        np.clip(cropped_image, 0, 255).astype(np.uint8)
                    )
                )

            except Exception as error:
                print(f"Cropping failed: {error}")

    radius_slider.on_changed(update_radius)
    reset_button.on_clicked(reset)
    apply_button.on_clicked(apply_crop)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)

    circle.radius_slider = radius_slider
    circle.reset_button = reset_button
    circle.apply_button = apply_button
    circle.output = output

    plt.show()
    display(output)

    return circle


def detect_plate_circle(rgb_image, min_radius=100, max_radius=None):
    """
    Estimate the center and inner radius of a round agar plate.

    Parameters
    ----------
    rgb_image
        Input RGB image array
    min_radius
        Minimum acceptable detected radius, in pixels
    max_radius
        Maximum acceptable detected radius, in pixels

    Returns
    -------
    tuple
        Plate center (x, y) and inner radius
    """
    image = np.asarray(rgb_image)

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA image.")

    if image.dtype != np.uint8:
        if np.max(image) <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[2] == 4:
        image = image[:, :, :3]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((15, 15), np.uint8)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel
    )

    binary[:10, :] = 0
    binary[-10:, :] = 0
    binary[:, :10] = 0
    binary[:, -10:] = 0

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    (x, y), outer_radius = cv2.minEnclosingCircle(contour)

    if max_radius is None:
        max_radius = max(image.shape[:2])

    if not (min_radius <= outer_radius <= max_radius):
        return None

    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    radii = np.arange(
        max(1, int(outer_radius * 0.75)),
        int(outer_radius) + 1
    )

    profiles = []

    for radius in radii:
        xs = np.rint(x + radius * np.cos(angles)).astype(int)
        ys = np.rint(y + radius * np.sin(angles)).astype(int)

        valid = (
            (xs >= 0) & (xs < gray.shape[1]) &
            (ys >= 0) & (ys < gray.shape[0])
        )

        if np.mean(valid) < 0.8:
            profiles.append(np.nan)
        else:
            profiles.append(
                np.median(blurred[ys[valid], xs[valid]])
            )

    profiles = np.asarray(profiles, dtype=float)

    valid = np.isfinite(profiles)

    if np.sum(valid) < 5:
        return int(x), int(y), int(outer_radius)

    profiles = np.interp(
        np.arange(len(profiles)),
        np.flatnonzero(valid),
        profiles[valid]
    )

    profiles = gaussian_filter(profiles, sigma=3)
    changes = np.abs(np.gradient(profiles))

    inner_radius = radii[np.argmax(changes)]

    if np.max(changes) < 1:
        inner_radius = outer_radius
    print("Outer radius:", outer_radius)
    print("Inner radius:", inner_radius)
    return int(x), int(y), int(inner_radius)



def crop_circular_plate(rgb_image, start_angle=None, end_angle=None,
                    interactive=False, plate=None, center=None, radius=None,
                    show_plot=True):

    """
    Crop a circular plate region, optionally selecting a sector.

    Parameters
    ----------
    rgb_image
        Input RGB image array
    start_angle
        Start angle of the selected plate sector, in degrees
    end_angle
        End angle of the selected plate sector, in degrees
    interactive
        If True, enable manual selection in a Jupyter notebook
    plate
        Plate object that contains the image data
    center
        Optional plate center as (x, y)
    radius
        Radius in pixels
    show_plot
        If True, show a plot of the selected area

    Returns
    -------
    ndarray
        Cropped image in non-interactive mode when show_plot is False

    Notes
    -----
    The selected image and mask are stored on the plate object when one is provided
    """
    if interactive and (start_angle is not None or end_angle is not None):
        raise ValueError("Choose either interactive mode or angles.")

    if (start_angle is None) != (end_angle is None):
        raise ValueError("Provide both start_angle and end_angle.")

    if center is None or radius is None:
        result = detect_plate_circle(rgb_image)

        if result is None:
            raise ValueError("No agar plate detected.")

        detected_x, detected_y, detected_radius = result

    if center is None:
        x, y = detected_x, detected_y
    else:
        x, y = map(int, center)

    if radius is None:
        radius = detected_radius

    radius = int(radius)

    if radius <= 0:
        raise ValueError("Radius must be positive.")

    height, width = rgb_image.shape[:2]

    x_min = max(0, x - radius)
    x_max = min(width, x + radius)
    y_min = max(0, y - radius)
    y_max = min(height, y + radius)

    cropped_rgb = np.asarray(
        rgb_image[y_min:y_max, x_min:x_max]).copy()

    cx = x - x_min
    cy = y - y_min

    crop_height, crop_width = cropped_rgb.shape[:2]
    yy, xx = np.ogrid[:crop_height, :crop_width]

    circle_mask = (
        (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2)

    def make_selection(angle1=None, angle2=None):
        mask = circle_mask.copy()

        if angle1 is not None and angle2 is not None:
            angles = (
                np.degrees(np.arctan2(cy - yy, xx - cx)) + 360
            ) % 360

            start = angle1 % 360
            end = angle2 % 360


            if (end - start) % 360 > 180:
                start, end = end, start

            if start <= end:
                sector = (angles >= start) & (angles <= end)
            else:
                sector = (angles >= start) | (angles <= end)

            mask &= sector

        rgba = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask.astype(np.uint8) * 255

        return rgba, mask

    def save_selection(mask):

        selected_rgb = cropped_rgb.copy()
        selected_rgb[~mask] = 0

        if plate is not None:
            plate.cropped_image = selected_rgb
            plate.cropped_mask = mask.copy()

            full_mask = np.zeros(plate.rgb_image.shape[:2], dtype=bool)
            full_mask[y_min:y_max, x_min:x_max] = mask
            plate.selection_mask = full_mask

        crop_circular_plate.last_image = selected_rgb
        crop_circular_plate.last_mask = mask.copy()

    initial_rgba, initial_mask = make_selection()
    save_selection(initial_mask)

    if not show_plot and not interactive:
        return crop_circular_plate.last_image

    fig, ax = subplots(figsize=(7, 7))
    image_artist = ax.imshow(initial_rgba)
    ax.set_title("Cropped agar plate")

    if interactive:
        selected_angles = []
        ax.set_title("Click two points on the plate boundary")

        def onclick(event):
            if event.inaxes != ax:
                return

            if event.xdata is None or event.ydata is None:
                return

            angle = (
                np.degrees(
                    np.arctan2(cy - event.ydata, event.xdata - cx)) + 360) % 360

            selected_angles.append(angle)

            if len(selected_angles) == 1:
                ax.set_title("Click the second boundary point")

            elif len(selected_angles) == 2:
                angle1, angle2 = selected_angles
                rgba, mask = make_selection(angle1, angle2)

                image_artist.set_data(rgba)
                save_selection(mask)

                ax.set_title(
                    f"Selected sector: {angle1:.1f}° – {angle2:.1f}°"
                )

                fig.canvas.mpl_disconnect(click_id)

            fig.canvas.draw_idle()

        click_id = fig.canvas.mpl_connect(
            "button_press_event", onclick)

    elif start_angle is not None:
        rgba, mask = make_selection(start_angle, end_angle)
        image_artist.set_data(rgba)
        save_selection(mask)

        ax.set_title(
            f"Selected sector: {start_angle}° – {end_angle}°")

    if show_plot:
        fig.canvas.draw_idle()
    else:
        plt.close(fig)


def _detect_agar_mask(rgb_image, bounds=None, max_processing_size=1100):
    """
    Detect the inner agar area of a rectangular plate.

    Parameters
    ----------
    rgb_image
        Input RGB image array
    bounds
        Optional rectangular bounds as (x_min, y_min, x_max, y_max)
    max_processing_size
        Maximum image dimension used during segmentation

    Returns
    -------
    ndarray
        Boolean mask of the agar surface
    """
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA image.")
    if image.dtype != np.uint8:
        arr = np.nan_to_num(image)
        if arr.max() <= 1:
            arr = arr * 255
        image = np.clip(arr, 0, 255).astype(np.uint8)
    image = image[:, :, :3]
    height, width = image.shape[:2]
    if min(height, width) < 80:
        raise ValueError("Image is too small for reliable agar segmentation.")

    scale = min(1.0, float(max_processing_size) / max(height, width))
    if scale < 1:
        small = cv2.resize(image, (max(1, round(width * scale)),
                                   max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = image
    sh, sw = small.shape[:2]



    mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)
    if bounds is None:
        x0, y0, x1, y1 = 0, 0, sw, sh
    else:
        if len(bounds) != 4:
            raise ValueError("bounds must be (x_min, y_min, x_max, y_max).")
        x0, y0, x1, y1 = [int(round(v * scale)) for v in bounds]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(sw, x1), min(sh, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Invalid rectangular bounds.")
    rw, rh = x1 - x0, y1 - y0


    def rectangle_fraction(fx0, fy0, fx1, fy1):
        return (int(x0 + rw * fx0), int(y0 + rh * fy0),
                int(x0 + rw * fx1), int(y0 + rh * fy1))
    ax, ay, bx, by = rectangle_fraction(.10, .09, .93, .88)
    mask[ay:by, ax:bx] = cv2.GC_PR_FGD
    ax, ay, bx, by = rectangle_fraction(.24, .25, .78, .73)
    mask[ay:by, ax:bx] = cv2.GC_FGD
    edge = max(2, round(min(rw, rh) * .025))
    mask[:edge, :] = cv2.GC_BGD
    mask[-edge:, :] = cv2.GC_BGD
    mask[:, :edge] = cv2.GC_BGD
    mask[:, -edge:] = cv2.GC_BGD

    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, None, bg_model, fg_model,
                    5, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        raise ValueError("Could not segment the inner agar surface.") from exc
    foreground = np.where((mask == cv2.GC_FGD) |
                          (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel_size = max(5, round(min(sh, sw) * .028) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No connected agar surface detected.")
    contour = max(contours, key=cv2.contourArea)
    area_fraction = cv2.contourArea(contour) / (sh * sw)
    if not (.20 <= area_fraction <= .93):
        raise ValueError("Agar segmentation is not reliable for this image.")
    if cv2.pointPolygonTest(contour, (sw / 2, sh / 2), False) < 0:
        raise ValueError("Detected surface does not contain the image centre.")



    hull = cv2.convexHull(contour)
    small_mask = np.zeros((sh, sw), dtype=np.uint8)
    cv2.fillPoly(small_mask, [hull], 255)
    if scale < 1:
        full = cv2.resize(small_mask, (width, height),
                          interpolation=cv2.INTER_NEAREST)
    else:
        full = small_mask
    return full.astype(bool)


def detect_plate_rectangle(rgb_image, min_area_fraction=0.05,
                           max_area_fraction=0.97):
    """
    Find the bounding box of a rectangular agar area.

    Parameters
    ----------
    rgb_image
        Input RGB image array
    min_area_fraction
        Minimum area fraction, retained for compatibility
    max_area_fraction
        Maximum area fraction, retained for compatibility

    Returns
    -------
    tuple
        Bounds (x_min, y_min, x_max, y_max)
    """
    try:
        mask = _detect_agar_mask(rgb_image)
    except ValueError:
        return None
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return (int(xs.min()), int(ys.min()),
            int(xs.max()) + 1, int(ys.max()) + 1)


def crop_rectangular_plate(rgb_image, bounds=None, plate=None,
                           show_plot=False, manual_polygon=None,
                           interactive=False, *, _analysis_origin=(0, 0)):
    """
    Crop a rectangular agar area using automatic or manual selection.

    Parameters
    ----------
    rgb_image
        Input RGB image array
    bounds
        Optional rectangular bounds as (x_min, y_min, x_max, y_max)
    plate
        Plate object that contains the image data
    show_plot
        If True, show a plot of the selected area
    manual_polygon
        Optional polygon vertices as (x, y) points
    interactive
        If True, enable manual selection in a Jupyter notebook

    Notes
    -----
    The selected image and masks are stored on the plate object when one is provided
    """
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA image.")
    height, width = image.shape[:2]
    if manual_polygon is not None and interactive:
        raise ValueError("Use either manual_polygon or interactive=True, not both.")

    if interactive:


        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.backend_bases import MouseButton

        suggested_mask = _detect_agar_mask(image, bounds=bounds)
        contours, _ = cv2.findContours(
            suggested_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No agar boundary available for editing.")
        outline = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(outline, True)

        tolerance = max(1.5, perimeter * 0.006)
        vertices = cv2.approxPolyDP(outline, tolerance, True).reshape(-1, 2).astype(float)
        if len(vertices) < 3:
            raise ValueError("Could not construct editable agar boundary.")
        if len(vertices) > 36:
            ids = np.linspace(0, len(vertices) - 1, 36, dtype=int)
            vertices = vertices[ids]

        scale = min(720 / width, 440 / height, 1.0)
        pw, ph = max(1, round(width * scale)), max(1, round(height * scale))
        preview = cv2.resize(image, (pw, ph), interpolation=cv2.INTER_AREA)
        with plt.ioff():
            fig, (ax_original, ax_crop) = plt.subplots(
                2, 1, figsize=(6, (ph + 440 + 55) / 120), dpi=120,
                constrained_layout=True)
        ax_original.imshow(preview, extent=(-.5, width-.5, height-.5, -.5))
        ax_original.set_title("Drag blue points to adjust the agar boundary", fontsize=10)
        ax_original.axis("off")
        polygon_artist = MplPolygon(vertices, closed=True, fill=False,
                                    edgecolor="red", linewidth=2)
        ax_original.add_patch(polygon_artist)
        (handles,) = ax_original.plot(vertices[:, 0], vertices[:, 1], "o",
                                      color="dodgerblue", markersize=7,
                                      markeredgecolor="white")
        preview_artist = ax_crop.imshow(np.zeros((2, 2, 3), dtype=np.uint8))
        ax_crop.set_title("Cropped agar", fontsize=10)
        ax_crop.axis("off")
        editor = {"vertices": vertices, "dragging": None}

        def update_crop():
            points = editor["vertices"]
            polygon_artist.set_xy(points)
            handles.set_data(points[:, 0], points[:, 1])

            crop_rectangular_plate(image, plate=plate, manual_polygon=points,
                                   show_plot=False, _analysis_origin=_analysis_origin)
            result = crop_rectangular_plate.last_image
            rh, rw = result.shape[:2]
            ratio = min(720 / rw, 440 / rh, 1.0)
            shown = cv2.resize(result, (max(1, round(rw * ratio)),
                                       max(1, round(rh * ratio))),
                               interpolation=cv2.INTER_AREA)
            preview_artist.set_data(shown)
            if plate is not None and np.count_nonzero(plate.selection_mask) < np.count_nonzero(crop_rectangular_plate.last_full_mask):
                ax_crop.set_title("Cropped agar (extends beyond analysis image)", fontsize=10)
            else:
                ax_crop.set_title("Cropped agar", fontsize=10)
            preview_artist.set_extent((0, result.shape[1], result.shape[0], 0))
            ax_crop.set_xlim(0, result.shape[1])
            ax_crop.set_ylim(result.shape[0], 0)
            fig.canvas.draw_idle()

        def on_press(event):
            if event.inaxes is not ax_original or event.button != MouseButton.LEFT:
                return
            if event.x is None or event.y is None:
                return
            pixels = ax_original.transData.transform(editor["vertices"])
            distances = np.hypot(pixels[:, 0] - event.x,
                                 pixels[:, 1] - event.y)
            index = int(np.argmin(distances))
            if distances[index] <= 18:
                editor["dragging"] = index

        def on_move(event):
            if editor["dragging"] is None or event.inaxes is not ax_original:
                return
            if event.xdata is None or event.ydata is None:
                return
            index = editor["dragging"]
            editor["vertices"][index] = (
                np.clip(event.xdata, 0, width - 1),
                np.clip(event.ydata, 0, height - 1))
            polygon_artist.set_xy(editor["vertices"])
            handles.set_data(editor["vertices"][:, 0],
                             editor["vertices"][:, 1])
            fig.canvas.draw_idle()

        def on_release(event):
            if editor["dragging"] is not None:
                editor["dragging"] = None
                update_crop()

        editor["connections"] = [
            fig.canvas.mpl_connect("button_press_event", on_press),
            fig.canvas.mpl_connect("motion_notify_event", on_move),
            fig.canvas.mpl_connect("button_release_event", on_release),
        ]
        update_crop()

        crop_rectangular_plate._active_editor = editor
        crop_rectangular_plate._active_figure = fig
        if plate is not None:
            plate._rect_crop_ui = {"figure": fig, "editor": editor}
        if hasattr(fig.canvas, "_ipython_display_") or hasattr(fig.canvas, "_repr_mimebundle_"):
            display(fig.canvas)
        else:
            plt.show()
        return None

    if manual_polygon is not None:
        vertices = np.array(manual_polygon, dtype=float, copy=True)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("manual_polygon must contain at least three (x, y) points.")
        if not np.isfinite(vertices).all():
            raise ValueError("Polygon points must be finite numbers.")
        vertices[:, 0] = np.clip(vertices[:, 0], 0, width - 1)
        vertices[:, 1] = np.clip(vertices[:, 1], 0, height - 1)
        full_mask_u8 = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(full_mask_u8, [np.rint(vertices).astype(np.int32)], 1)
        full_mask = full_mask_u8.astype(bool)
    else:
        full_mask = _detect_agar_mask(image, bounds=bounds)

    ys, xs = np.nonzero(full_mask)
    if not len(xs):
        raise ValueError("No agar surface detected.")
    x_min, y_min, x_max, y_max = (int(xs.min()), int(ys.min()),
                                  int(xs.max()) + 1, int(ys.max()) + 1)
    cropped_mask = full_mask[y_min:y_max, x_min:x_max].copy()
    cropped = image[y_min:y_max, x_min:x_max].copy()
    cropped[~cropped_mask] = 0

    if plate is not None:
        ox, oy = _analysis_origin
        ah, aw = plate.rgb_image.shape[:2]
        if ox < 0 or oy < 0 or ox + aw > width or oy + ah > height:
            raise ValueError("Plate and image dimensions do not match.")
        if _analysis_origin == (0, 0) and (ah, aw) != (height, width):
            raise ValueError("Plate and image dimensions do not match.")

        selection = full_mask[oy:oy+ah, ox:ox+aw].copy()
        sy, sx = np.nonzero(selection)
        if not len(sx):
            raise ValueError("The selected region does not overlap the analysis image.")
        bx1, by1 = int(sx.min()), int(sy.min())
        bx2, by2 = int(sx.max()) + 1, int(sy.max()) + 1
        plate.cropped_mask = selection[by1:by2, bx1:bx2].copy()
        plate.cropped_image = plate.rgb_image[by1:by2, bx1:bx2].copy()
        plate.cropped_image[~plate.cropped_mask] = 0
        plate.selection_mask = selection
        plate.cropped_bounds = (bx1, by1, bx2, by2)

    crop_rectangular_plate.last_image = cropped
    crop_rectangular_plate.last_mask = cropped_mask
    crop_rectangular_plate.last_full_mask = full_mask
    crop_rectangular_plate.last_bounds = (x_min, y_min, x_max, y_max)

    if show_plot:
        fig, axes = subplots(2, 1, figsize=(10, 11))
        axes[0].imshow(image)
        axes[0].contour(full_mask, levels=[.5], colors='red', linewidths=1.5)
        axes[0].set_title("Detected agar boundary" if manual_polygon is None
                          else "Manually adjusted agar boundary")
        axes[1].imshow(cropped)
        axes[1].set_title("Cropped agar")
        for ax in axes:
            ax.axis('off')
        fig.tight_layout()
