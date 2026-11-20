import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colony_differentiation import tools
from colony_differentiation.tools import GeneralPlate, SingleImagePlate, Plate, Candidates, Species, show, _array_with_hash, _smoothen, _base_neighbourhood, _neighbourhood, _base_circle, _footprint, _local_maxima, _get_direction, _transpose_if_needed, _get_intensities_and_angles, _get_circularities, _area_of_interest, _is_close_to_border


def test_array_with_hash():
    """Array with hash."""
    original = np.array([1,2,3])
    result = _array_with_hash.from_array(original)

    assert isinstance(result, _array_with_hash)
    assert np.array_equal(result, original)


def test_array_hash():
    """Array hash."""
    a = _array_with_hash.from_array([1, 2, 3])
    b = _array_with_hash.from_array([1, 2, 3])

    assert hash(a) == hash(b)


def test_array_with_hash_noncontiguous():
    """Array with hash noncontiguous."""
    original = np.arange(10)[::2]
    result = _array_with_hash.from_array(original)

    assert isinstance(result, _array_with_hash)
    assert np.array_equal(result, original)
    assert result.flags.c_contiguous


def test_show_2d():
    """Show 2d."""
    image = np.zeros((10, 10))

    fig, ax = show(image)

    assert fig is not None
    assert ax is not None


def test_show_rgb():
    """Show rgb."""
    image = np.zeros((10, 10, 3))

    fig, ax = show(image)

    assert fig is not None
    assert ax is not None


def test_smoothen():
    """Smoothen."""
    image = np.zeros((21, 21))
    image[10, 10] = 1

    result = _smoothen(_array_with_hash.from_array(image), sigma=2)

    assert result.shape == image.shape
    assert result[10, 10] < 1
    assert result[10, 11] > 0


def test_smoothen_constant_image():
    """Smoothen constant image."""
    image = np.ones((10, 10))

    result = _smoothen(_array_with_hash.from_array(image), sigma=2)

    assert np.allclose(np.asarray(result), image)


def test_base_neighbourhood():
    """Base neighbourhood."""
    positions = _base_neighbourhood(1)

    actual = set(zip(*positions))
    expected = {(0, 0)}

    assert actual == expected


def test_base_neighbourhood_radius_2():
    """Base neighbourhood radius 2."""
    positions = _base_neighbourhood(2)

    actual = set(zip(*positions))
    expected = {
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),  (0, 0),  (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    }

    assert actual == expected


def test_neighbourhood():
    """Neighbourhood."""
    positions = _neighbourhood(5, 5, 1)

    actual = set(zip(*positions))
    expected = {(5, 5)}

    assert actual == expected


def test_neighbourhood_radius_2():
    """Neighbourhood radius 2."""
    positions = _neighbourhood(5, 5, 2)

    actual = set(zip(*positions))
    expected = {
        (4, 4), (4, 5), (4, 6),
        (5, 4), (5, 5), (5, 6),
        (6, 4), (6, 5), (6, 6)
    }

    assert actual == expected


def test_base_circle_radius_1():
    """Base circle radius 1."""
    positions = set(zip(*_base_circle(1)))

    expected = {
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    }

    assert positions == expected


def test_base_circle_radius_5():
    """Base circle radius 5."""
    positions = set(zip(*_base_circle(5)))

    assert len(positions) > 4

    for i, j in positions:
        distance = np.hypot(i, j)
        assert 4.5 <= distance < 5.5
        assert (-i, -j) in positions


def test_footprint():
    """Footprint."""
    result = _footprint(2)

    assert result.shape == (3, 3)
    assert result.dtype == bool
    assert np.all(result)


def test_local_maxima():
    """Local maxima."""
    scores = np.zeros((7, 7))
    scores[3, 3] = 10

    maxima = list(_local_maxima(scores, radius=2))

    assert (3, 3) in maxima


def test_get_direction():
    """Get direction."""
    smoothed = np.zeros((3, 3, 3), dtype=float)
    smoothed[1, 1] = [3, 4, 0]

    direction = _get_direction(1, 1, smoothed)

    assert np.allclose(direction, [0.6, 0.8, 0])
    assert np.isclose(np.linalg.norm(direction), 1)


def test_transpose_if_needed():
    """Transpose if needed."""
    image = np.zeros((10, 20, 3))
    result = _transpose_if_needed(image)

    assert result.shape == (10, 20, 3)

    portrait = np.zeros((20, 10, 3))
    result_portrait = _transpose_if_needed(portrait)

    assert result_portrait.shape == (10, 20, 3)


def test_get_intensities_and_angles():
    """Get intensities and angles."""
    image = np.ones((10, 10, 3), dtype=float)
    image = _array_with_hash.from_array(image)

    species = Species(vector=(1, 0, 0), sigma=1)

    intensities, angles = _get_intensities_and_angles(image, tuple(species.vector), species.sigma)

    assert intensities.shape == (10, 10)
    assert angles.shape == (10, 10)
    assert np.allclose(np.asarray(intensities), 1)
    assert np.all(np.isfinite(angles))


def test_get_intensities_and_angles_zero_image():
    """Get intensities and angles zero image."""
    image = _array_with_hash.from_array(np.zeros((10, 10, 3)))

    species = Species(vector=(1, 0, 0), sigma=1)

    intensities, angles = _get_intensities_and_angles(image, tuple(species.vector), species.sigma)

    assert np.allclose(np.asarray(intensities), 0)
    assert angles.shape == (10, 10)


def test_get_circularities_shape():
    """Get circularities shape."""
    image = _array_with_hash.from_array(
        np.random.default_rng(42).random((20, 20, 3))
    )

    result = _get_circularities(image, radius=5)

    assert result.shape == (20, 20)
    assert np.all(np.isfinite(result))


def test_area_of_interest_inside():
    """Area of interest inside."""
    assert _area_of_interest(5, 5, 0, 10, 0, 10)


def test_area_of_interest_outside():
    """Area of interest outside."""
    assert not _area_of_interest(15, 5, 0, 10, 0, 10)


def test_area_of_interest_boundary():
    """Area of interest boundary."""
    assert _area_of_interest(0, 0, 0, 10, 0, 10)
    assert _area_of_interest(10, 10, 0, 10, 0, 10)


def test_species_normalizes_vector():
    """Species normalizes vector."""
    species = Species(
        vector=np.array([3.0, 4.0, 0.0]),
        weights=np.array([1.0, 2.0, 2.0])
    )

    assert np.allclose(species.vector, [0.6, 0.8, 0])
    assert np.isclose(np.linalg.norm(species.vector), 1)


def test_species_normalizes_weights():
    """Species normalizes weights."""
    species = Species(
        vector=np.array([1.0, 0.0, 0.0]),
        weights=np.array([1.0, 2.0, 2.0])
    )

    assert np.allclose(species.weights, [0.2, 0.4, 0.4])


def test_species_hash():
    """Species hash."""
    a = Species(
        name="test",
        vector=np.array([1.0, 0.0, 0.0]),
        weights=np.array([1.0, 2.0, 2.0])
    )
    b = Species(
        name="test",
        vector=np.array([1.0, 0.0, 0.0]),
        weights=np.array([1.0, 2.0, 2.0])
    )

    assert hash(a) == hash(b)


def test_candidates_initialization():
    """Candidates initialization."""
    plate = GeneralPlate("test_plate")
    candidates = Candidates(plate, time=20, radius=5, sigma=2)

    candidates.append((10, 15))

    assert candidates.plate is plate
    assert candidates.kwargs["time"] == 20
    assert candidates == [(10, 15)]


def test_general_plate_initialization():
    """General plate initialization."""
    plate = GeneralPlate("test_plate")

    assert plate.name == "test_plate"
    assert plate.colonies == {}
    assert plate.scores == {}
    assert plate.candidates == []


def test_general_plate_clean():
    """General plate clean."""
    plate = GeneralPlate("test_plate")

    plate.colonies["example"] = [(5, 5)]
    plate.scores["example"] = np.ones((3, 3))

    plate.clean()

    assert plate.colonies == {}
    assert plate.scores == {}


def test_general_plate_bad_regions():
    """General plate bad regions."""
    plate = GeneralPlate(
        "test_plate",
        bad_regions=[(10, 10, 5)]
    )

    assert plate._is_in_bad_region(10, 10)
    assert plate._is_in_bad_region(12, 12)
    assert not plate._is_in_bad_region(20, 20)
    assert not plate._is_in_bad_region(15, 10)


def test_categorise_colony_good():
    """Categorise colony good."""
    plate = GeneralPlate("test")
    species = Species(name="A")

    plate.intensities[species] = np.array([[1000.0]])
    plate.angles[species] = np.array([[0.1]])
    plate.circularities[species] = np.array([[2.0]])

    assert plate._categorise_colony(0, 0, species) == "good"


def test_categorise_colony_missed():
    """Categorise colony missed."""
    plate = GeneralPlate("test")
    species = Species(name="A")

    plate.intensities[species] = np.array([[1000.0]])
    plate.angles[species] = np.array([[0.45]])
    plate.circularities[species] = np.array([[2.0]])

    assert plate._categorise_colony(0, 0, species) == "missed"


def test_categorise_colony_nothing():
    """Categorise colony nothing."""
    plate = GeneralPlate("test")
    species = Species(name="A")

    plate.intensities[species] = np.array([[1000.0]])
    plate.angles[species] = np.array([[3.0]])
    plate.circularities[species] = np.array([[0.1]])

    assert plate._categorise_colony(0, 0, species) == "nothing"


def test_first_file_after_time():
    """First file after time."""
    plate = GeneralPlate("test")
    plate.files = {
        "image_1": 10,
        "image_2": 20,
        "image_3": 30,
    }

    assert plate.first_file_after_time(15) == "image_2"
    assert plate.first_file_after_time(20) == "image_3"


def test_clean_species():
    """Clean species."""
    plate = GeneralPlate("test")
    species = Species(name="A")

    plate.intensities[species] = np.ones((3, 3))
    plate.scores[species] = np.ones((3, 3))
    plate.colonies[species] = [(1, 1)]

    plate._clean_species(species)

    assert species not in plate.intensities
    assert species not in plate.scores
    assert species not in plate.colonies


def test_check_candidate_good(monkeypatch):
    """Check candidate good."""
    plate = GeneralPlate("test")
    species = Species(name="A", radius=5)

    plate.species_candidates[species] = {(100, 100)}
    plate.scores[species] = np.zeros((200, 200))

    monkeypatch.setattr(
        "colony_differentiation.tools._is_close_to_border",
        lambda i, j, radius: False
    )

    assert plate._check_candidate(100, 100, species) == "good"


def test_check_candidate_bad_region(monkeypatch):
    """Check candidate bad region."""
    plate = GeneralPlate(
        "test",
        bad_regions=[(100, 100, 10)]
    )
    species = Species(name="A", radius=5)

    monkeypatch.setattr(
        "colony_differentiation.tools._is_close_to_border",
        lambda i, j, radius: False
    )

    assert plate._check_candidate(100, 100, species) == "in bad region"


def test_check_candidate_close_to_border(monkeypatch):
    """Check candidate close to border."""
    plate = GeneralPlate("test")
    species = Species(name="A", radius=5)

    monkeypatch.setattr(
        "colony_differentiation.tools._is_close_to_border",
        lambda i, j, radius: True
    )

    assert plate._check_candidate(0, 0, species) == "close to border"


def test_compute_scores_shape(monkeypatch):
    """Compute scores shape."""
    plate = GeneralPlate("test")
    species = Species(name="A")
    image = np.zeros((2, 2, 3))

    monkeypatch.setattr(
        "colony_differentiation.tools._get_intensities_and_angles",
        lambda image, vector, sigma: (
            np.full((2, 2), 1000.0),
            np.full((2, 2), 0.2)
        )
    )
    monkeypatch.setattr(
        "colony_differentiation.tools._get_circularities",
        lambda image, radius: np.ones((2, 2))
    )

    plate._compute_scores(image, species)

    assert plate.scores[species].shape == (2, 2)


def test_compute_scores_best_intensity(monkeypatch):
    """Compute scores best intensity."""
    plate = GeneralPlate("test")
    species = Species(name="A", intensity=1000)
    image = np.zeros((1, 2, 3))

    monkeypatch.setattr(
        "colony_differentiation.tools._get_intensities_and_angles",
        lambda image, vector, sigma: (
            np.array([[1000.0, 100.0]]),
            np.array([[0.2, 0.2]])
        )
    )
    monkeypatch.setattr(
        "colony_differentiation.tools._get_circularities",
        lambda image, radius: np.ones((1, 2))
    )

    plate._compute_scores(image, species)

    assert plate.scores[species][0, 0] > plate.scores[species][0, 1]


def test_compute_scores_zero_intensity(monkeypatch):
    """Compute scores zero intensity."""
    plate = GeneralPlate("test")
    species = Species(name="A")
    image = np.zeros((1, 1, 3))

    monkeypatch.setattr(
        "colony_differentiation.tools._get_intensities_and_angles",
        lambda image, vector, sigma: (
            np.array([[0.0]]),
            np.array([[0.2]])
        )
    )
    monkeypatch.setattr(
        "colony_differentiation.tools._get_circularities",
        lambda image, radius: np.ones((1, 1))
    )

    plate._compute_scores(image, species)

    assert np.isneginf(plate.scores[species][0, 0])


def test_candidates_get_species_rejects_include_and_exclude():
    """Candidates get species rejects include and exclude."""
    plate = GeneralPlate("test")
    candidates = Candidates(plate, time=10, sigma=1, radius=3)
    with pytest.raises(ValueError, match="Either exclude or include"):
        candidates.get_species("A", include=[0], exclude=[1])


def test_candidates_get_by_color_empty(capsys):
    """Candidates get by color empty."""
    plate = GeneralPlate("test")
    plate.rgb_image = np.zeros((20, 20, 3))
    candidates = Candidates(plate)
    candidates.colors = {"red": []}
    assert candidates.get_by_color("red") is None
    assert "No red colonies found" in capsys.readouterr().out


def test_plate_count_groups_species_by_prefix():
    """Plate count groups species by prefix."""
    plate = GeneralPlate("test")
    a = Species(name="Ecoli.1")
    b = Species(name="Ecoli.2")
    c = Species(name="Yeast")
    plate.colonies = {a: {(1, 1), (2, 2)}, b: {(3, 3)}, c: {(4, 4)}}
    assert plate.count() == {"Ecoli": 3, "Yeast": 1}


def test_plate_count_empty():
    """Plate count empty."""
    assert GeneralPlate("test").count() == {}


def test_plate_save_counts(tmp_path, monkeypatch):
    """Plate save counts."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "counts_2").mkdir()
    plate = GeneralPlate("sample")
    species = Species(name="A")
    plate.colonies[species] = {(2, 3)}
    assert plate.save_counts() == {"A": 1}
    import pickle
    with (tmp_path / "counts_2" / "sample.pickle").open("rb") as file:
        assert pickle.load(file) == {"A": 1}


def test_plot_statistics_with_colonies():
    """Plot statistics with colonies."""
    import matplotlib.pyplot as plt
    plate = GeneralPlate("test")
    species = Species(name="A")
    plate.colonies[species] = {(2, 2)}
    plate.misses[species] = {(3, 3)}
    plate.intensities[species] = np.full((10, 10), 1000.0)
    plate.angles[species] = np.full((10, 10), 0.1)
    plate.circularities[species] = np.full((10, 10), 2.0)
    fig, axes = plate.plot_statistics(species)
    assert len(axes) == 3
    plt.close(fig)


def test_plot_timeseries_with_data():
    """Plot timeseries with data."""
    import matplotlib.pyplot as plt
    plate = GeneralPlate("test")
    species = Species(name="A", time=1)
    plate.times = [0, 1, 2]
    plate.timeseries = {species: {(2, 2): [100.0, 200.0, 300.0]}}
    plate.misses_timeseries = {species: {(3, 3): [50.0, 80.0, 100.0]}}
    plate.angle_ts = {species: {(2, 2): [0.1, 0.2, 0.3]}}
    plate.angle_misses_ts = {species: {(3, 3): [0.2, 0.3, 0.4]}}
    fig, axes = plate.plot_timeseries(species)
    assert len(axes) == 2
    plt.close(fig)


def test_single_image_plate_initialization(monkeypatch):
    """Single image plate initialization."""
    monkeypatch.setattr("colony_differentiation.tools.load_RGB", lambda *args: np.zeros((10, 10, 3)))
    monkeypatch.setattr("colony_differentiation.tools.load_image",
                        lambda *args: _array_with_hash.from_array(np.ones((10, 10, 3))))
    plate = SingleImagePlate("image.CR2", "test")
    assert plate.first_file_after_time(100) == "image.CR2"
    assert plate[0, 0].shape == (3,)
    assert plate.image.shape == (10, 10, 3)


def test_single_image_plate_supplied_background(monkeypatch):
    """Single image plate supplied background."""
    monkeypatch.setattr("colony_differentiation.tools.load_RGB", lambda *args: np.zeros((10, 10, 3)))
    monkeypatch.setattr("colony_differentiation.tools.load_image",
                        lambda *args: _array_with_hash.from_array(np.ones((10, 10, 3))))
    background = np.zeros((10, 10, 3))
    plate = SingleImagePlate("image.CR2", "test", background=background)
    assert plate.background is background


def test_is_close_to_border_detects_edges():
    """Is close to border detects edges."""
    from colony_differentiation.tools import _is_close_to_border
    assert _is_close_to_border(0, 0, 5)
    assert not _is_close_to_border(100, 100, 5)


def test_check_candidate_outside_area_of_interest(monkeypatch):
    """Check candidate outside area of interest."""
    plate = GeneralPlate(
        "test",
        area_of_interest=(10, 30, 15, 35)
    )
    species = Species(name="A", radius=3)

    plate.species_candidates[species] = {(20, 40)}
    plate.scores[species] = np.zeros((50, 50))

    monkeypatch.setattr(
        "colony_differentiation.tools._is_close_to_border",
        lambda *args: False
    )

    assert plate._check_candidate(20, 40, species) == (
        "colony is not in area of interest"
    )


def test_check_candidate_inside_area_of_interest(monkeypatch):
    """Check candidate inside area of interest."""
    plate = GeneralPlate(
        "test",
        area_of_interest=(10, 30, 15, 35)
    )
    species = Species(name="A", radius=3)

    plate.species_candidates[species] = {(20, 25)}
    plate.scores[species] = np.zeros((50, 50))

    monkeypatch.setattr(
        "colony_differentiation.tools._is_close_to_border",
        lambda *args: False
    )

    assert plate._check_candidate(20, 25, species) == "good"


def test_exclude_and_restore_candidates():
    """Excludes and restores candidate points."""
    candidates = Candidates(GeneralPlate("test"))
    candidates.extend([(5, 5), (10, 10), (15, 15)])
    candidates.exclude_candidates(1)
    assert candidates.active_candidates() == [(5, 5), (15, 15)]
    candidates.restore_candidates(1)
    assert len(candidates.active_candidates()) == 3


def test_exclude_invalid_candidate():
    """Rejects an invalid candidate number."""
    candidates = Candidates(GeneralPlate("test"))
    candidates.append((5, 5))
    with pytest.raises(ValueError):
        candidates.exclude_candidates(5)


def test_classify_sizes():
    """Groups colonies by their diameter."""
    candidates = Candidates(GeneralPlate("test"))
    candidates.measured_sizes = np.array([5, 12, 20])
    result = candidates.classify_sizes()
    assert result == {"small": 1, "medium": 1, "large": 1}


def test_measure_sizes():
    """Measures the size of a bright colony."""
    plate = GeneralPlate("test")
    plate.rgb_image = np.zeros((40, 40, 3), dtype=np.uint8)
    plate.rgb_image[17:24, 17:24] = 255
    candidates = Candidates(plate)
    candidates.append((20, 20))
    sizes = candidates.measure_sizes()
    assert len(sizes) == 1
    assert sizes[0] > 0


def test_single_image_numpy():
    """Loads an image from a NumPy array."""
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    plate = SingleImagePlate(image, "test")
    assert plate.image.shape == (30, 30, 3)


def test_single_image_png(tmp_path):
    """Loads a PNG image."""
    from PIL import Image
    filename = tmp_path / "image.png"
    Image.fromarray(np.zeros((30, 30, 3), dtype=np.uint8)).save(filename)
    plate = SingleImagePlate(str(filename), "test")
    assert plate.image.shape == (30, 30, 3)


def test_find_candidates_single_image():
    """Searches for candidates in one image."""
    image = np.zeros((70, 70, 3), dtype=np.uint8)
    image[30:40, 30:40] = [200, 50, 50]
    plate = SingleImagePlate(image, "test")
    plate.find_candidates(n=5, radius=3, threshold=1, q=4)
    assert isinstance(plate.candidates, Candidates)



def test_find_colonies_without_candidates(monkeypatch):
    """Runs species detection without finding candidates first."""
    plate = GeneralPlate("test")
    plate.files = {"image": 40}
    plate.background = None
    monkeypatch.setattr(tools, "load_image", lambda *args: _array_with_hash.from_array(np.zeros((40, 40, 3))))
    species = Species(name="A", vector=np.array([1., 0., 0.]),
                      weights=np.array([1., 1., 1.]), radius=3)
    plate.find_colonies(species, create_pdf=False)
    assert species in plate.colonies


def test_check_candidate_outscored(monkeypatch):
    """Check candidate outscored."""
    plate = GeneralPlate("test")
    species = Species(name="A", radius=5)

    plate.species_candidates[species] = {(100, 100), (102, 102)}

    scores = np.zeros((200, 200))
    scores[100, 100] = 5
    scores[102, 102] = 10
    plate.scores[species] = scores

    monkeypatch.setattr(
        "colony_differentiation.tools._is_close_to_border",
        lambda i, j, radius: False
    )

    assert plate._check_candidate(100, 100, species) == "outscored: 102,102"

def test_check_candidate_existing_colony(monkeypatch):
    """Check candidate existing colony."""
    plate = GeneralPlate("test")

    existing_species = Species(name="existing", radius=5)
    new_species = Species(name="new", radius=5)

    plate.colonies[existing_species] = {(100, 100)}
    plate.species_candidates[new_species] = {(102, 102)}
    plate.scores[new_species] = np.zeros((200, 200))

    monkeypatch.setattr(
        "colony_differentiation.tools._is_close_to_border",
        lambda i, j, radius: False
    )

    assert (
        plate._check_candidate(102, 102, new_species)
        == "existing existing: 100,100"
    )

def test_compute_scores_best_angle(monkeypatch):
    """Compute scores best angle."""
    plate = GeneralPlate("test")
    species = Species(name="A")
    image = np.zeros((1, 2, 3))

    monkeypatch.setattr(
        "colony_differentiation.tools._get_intensities_and_angles",
        lambda image, vector, sigma: (
            np.full((1, 2), 1000.0),
            np.array([[0.1, 0.8]])
        )
    )
    monkeypatch.setattr(
        "colony_differentiation.tools._get_circularities",
        lambda image, radius: np.ones((1, 2))
    )

    plate._compute_scores(image, species)

    assert plate.scores[species][0, 0] > plate.scores[species][0, 1]

def test_compute_scores_cached(monkeypatch):
    """Compute scores cached."""
    plate = GeneralPlate("test")
    species = Species(name="A")
    image = np.zeros((1, 1, 3))

    calls = []

    def fake_intensities(image, vector, sigma):
        calls.append(1)
        return np.array([[1000.0]]), np.array([[0.2]])

    monkeypatch.setattr(
        "colony_differentiation.tools._get_intensities_and_angles",
        fake_intensities
    )
    monkeypatch.setattr(
        "colony_differentiation.tools._get_circularities",
        lambda image, radius: np.ones((1, 1))
    )

    plate._compute_scores(image, species)
    plate._compute_scores(image, species)

    assert len(calls) == 1

def test_bad_region_exact_boundary():
    """Bad region exact boundary."""
    plate = GeneralPlate("test", bad_regions=[(20, 20, 5)])
    assert plate._is_in_bad_region(20, 25) is False

def test_bad_region_just_inside_boundary():
    """Bad region just inside boundary."""
    plate = GeneralPlate("test", bad_regions=[(20, 20, 5)])
    assert plate._is_in_bad_region(20, 24) is True

def test_species_repr_contains_settings():
    """Species repr contains settings."""
    species = Species(name="E coli", radius=5, intensity=1000.0)
    description = repr(species)
    assert 'E coli = Species(' in description
    assert 'radius = 5' in description
    assert 'weights = [' in description

def test_candidates_get_by_color_shows_patches():
    """Candidates get by color shows patches."""
    import matplotlib.pyplot as plt
    plate = GeneralPlate("test")
    plate.rgb_image = np.zeros((40, 40, 3))
    candidates = Candidates(plate)
    candidates.colors = {"red": [(10, 10), (20, 20)]}
    before = set(plt.get_fignums())
    candidates.get_by_color("red", q=5)
    assert len(set(plt.get_fignums()) - before) == 1
    plt.close("all")

def test_plot_statistics_without_colonies():
    """Plot statistics without colonies."""
    import matplotlib.pyplot as plt
    plate = GeneralPlate("test")
    species = Species(name="A")
    plate.colonies[species] = set()
    plate.misses[species] = set()
    plate.intensities[species] = np.ones((10, 10))
    plate.angles[species] = np.ones((10, 10))
    plate.circularities[species] = np.ones((10, 10))
    with pytest.warns(UserWarning, match="No artists with labels"):
        fig, axes = plate.plot_statistics(species)
    assert len(axes) == 3
    plt.close(fig)

def test_show_area_of_interest_crops_image(monkeypatch):
    """Show area of interest crops image."""
    from colony_differentiation.tools import _show_area_of_interest
    seen = []
    monkeypatch.setattr("colony_differentiation.tools.show", lambda image: seen.append(image.shape))
    _show_area_of_interest(np.zeros((20, 30, 3)), 2, 10, 4, 12)
    assert seen == [(8, 8, 3)]

def test_get_species_single_image():
    """Estimates species settings from a single image."""
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    image[25:35, 25:35] = [220, 80, 30]
    plate = SingleImagePlate(image, "test")
    candidates = Candidates(plate, radius=3, sigma=1, time=0)
    candidates.append((30, 30))
    species = candidates.get_species("Red")
    assert isinstance(species, Species)
    assert species.name == "Red"


def test_find_colonies_single_image():
    """Detects species directly in a NumPy image."""
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    image[25:35, 25:35] = [220, 80, 30]
    plate = SingleImagePlate(image, "test")
    species = Species(name="Red", vector=np.array([1., 0., 0.]),
                      weights=np.array([1., 1., 1.]), radius=3)
    plate.find_colonies(species, create_pdf=False)
    assert species in plate.colonies


def test_classify_colors():
    """Classifies candidate colors in an RGB image."""
    plate = GeneralPlate("test")
    plate.rgb_image = np.zeros((40, 40, 3), dtype=np.uint8)
    plate.rgb_image[8:13, 8:13] = [255, 0, 0]
    candidates = Candidates(plate)
    candidates.append((10, 10))
    candidates.classify_colors()
    assert candidates.colors["red"] == [(10, 10)]


def test_cluster_candidates():
    """Groups candidates with similar colors."""
    plate = GeneralPlate("test")
    plate.rgb_image = np.zeros((50, 50, 3), dtype=np.uint8)
    candidates = Candidates(plate)
    candidates.extend([(10, 10), (20, 20), (30, 30)])
    labels = candidates.cluster_candidates(min_samples=2, verbose=False)
    assert len(labels) == 3
    assert len(set(labels)) == 1


def test_get_species_no_candidates():
    """Rejects species estimation without candidates."""
    plate = SingleImagePlate(np.zeros((30, 30, 3), dtype=np.uint8), "test")
    candidates = Candidates(plate)
    with pytest.raises(ValueError):
        candidates.get_species("Empty")


def test_detect_plate_circle():
    """Detect a round plate."""
    image = np.zeros((240, 240, 3), dtype=np.uint8)
    import cv2
    cv2.circle(image, (120, 120), 85, (220, 220, 220), -1)
    result = tools.detect_plate_circle(image, min_radius=30)
    assert result is not None
    assert abs(result[0] - 120) < 5


def test_detect_plate_circle_empty():
    """Reject an empty image."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    assert tools.detect_plate_circle(image, min_radius=30) is None


def test_crop_circular_plate():
    """Crop a round plate."""
    image = np.ones((100, 100, 3), dtype=np.uint8) * 100
    result = tools.crop_circular_plate(image, center=(50, 50), radius=30, show_plot=False)
    assert result.shape == (60, 60, 3)
    assert result[0, 0].sum() == 0


def test_crop_circular_plate_invalid_radius():
    """Reject an invalid radius."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        tools.crop_circular_plate(image, center=(50, 50), radius=0, show_plot=False)


def test_crop_circular_plate_sector():
    """Select a circular sector."""
    image = np.ones((100, 100, 3), dtype=np.uint8)
    tools.crop_circular_plate(image, center=(50, 50), radius=30,
                              start_angle=0, end_angle=90, show_plot=True)
    assert tools.crop_circular_plate.last_mask.sum() < 2000
    plt.close('all')


def test_crop_rectangular_plate_polygon():
    """Crop a polygon."""
    image = np.ones((100, 100, 3), dtype=np.uint8) * 120
    tools.crop_rectangular_plate(image, manual_polygon=[(10, 10), (80, 10), (80, 80), (10, 80)])
    assert tools.crop_rectangular_plate.last_image.shape[0] > 60
    assert tools.crop_rectangular_plate.last_mask.any()


def test_crop_rectangular_plate_invalid_polygon():
    """Reject a short polygon."""
    image = np.ones((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        tools.crop_rectangular_plate(image, manual_polygon=[(0, 0), (10, 10)])


def test_detect_plate_rectangle_small_image():
    """Reject a small image."""
    assert tools.detect_plate_rectangle(np.zeros((20, 20, 3), dtype=np.uint8)) is None


def test_detect_agar_mask_bounds():
    """Check rectangular bounds."""
    image = np.ones((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        tools._detect_agar_mask(image, bounds=(80, 80, 10, 10))


def test_plate_acquisition_time_filename(tmp_path):
    """Read a filename timestamp."""
    filename = tmp_path / '01_2024-09-27_08-34-02.CR2'
    filename.write_bytes(b'')
    with pytest.warns(UserWarning):
        timestamp = Plate._acquisition_time(str(filename))
    assert timestamp.year == 2024
    assert timestamp.hour == 8


def test_plate_find_files(tmp_path):
    """Find time-series files."""
    (tmp_path / '01_2024-09-27_08-34-02.CR2').write_bytes(b'')
    (tmp_path / '01_2024-09-27_09-34-02.CR2').write_bytes(b'')
    plate = Plate.__new__(Plate)
    with pytest.warns(UserWarning):
        plate._find_files(tmp_path, 1)
    assert len(plate.files) == 2


def test_plate_find_files_missing(tmp_path):
    """Reject an empty folder."""
    plate = Plate.__new__(Plate)
    with pytest.raises(ValueError):
        plate._find_files(tmp_path, 1)
