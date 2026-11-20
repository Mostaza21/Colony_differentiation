API Reference
=============

This section provides an overview of the public classes, methods and functions available in Colony Differentiation, including their parameters and return values.

The main classes are ``SingleImagePlate`` for one image and ``Plate`` for a time series. Both share detection methods through ``GeneralPlate``.


Image Analysis
--------------

``SingleImagePlate`` loads a single image. ``Plate`` loads a series of RAW images, orders them by acquisition time from metadata (with a filename timestamp fallback), and estimates a background from the recordings.

Image Loading and Display
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tools.SingleImagePlate

.. autoclass:: tools.Plate

.. autofunction:: tools.show

.. automethod:: tools.GeneralPlate.show

Rectangular Agar Plate
~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: tools.SingleImagePlate.crop_rectangular_plate

.. automethod:: tools.SingleImagePlate.select_rectangular_aoi

.. automethod:: tools.SingleImagePlate.show_area_of_interest

With ``use_cropped=True``, the rectangular AOI is selected within the previously cropped agar region. Otherwise, the selection replaces the current selection mask.

Circular Agar Plate
~~~~~~~~~~~~~~~~~~~

.. automethod:: tools.SingleImagePlate.crop_circular_plate

The ``interactive=True`` option selects a sector by two mouse clicks. It does not adjust the center or radius of the circle. Sector selection through this method uses automatic circle detection.

.. autofunction:: tools.adjust_plate_circle

After clicking **Apply & Crop**, apply the selected center and radius using the following function with ``plate=plate`` to store the selection for subsequent detection.

.. autofunction:: tools.crop_circular_plate


Exploratory Analysis
--------------------

Exploratory analysis does not require a ``Species`` definition. ``find_candidates()`` detects preliminary colony locations and stores them in ``plate.candidates``. The ``Candidates`` collection can be used to inspect, exclude and characterize these locations.

.. automethod:: tools.GeneralPlate.find_candidates

.. autoclass:: tools.Candidates

.. automethod:: tools.Candidates.plot_candidates

.. automethod:: tools.Candidates.exclude_candidates

.. automethod:: tools.Candidates.restore_candidates

.. automethod:: tools.Candidates.active_candidates

.. automethod:: tools.Candidates.active_indices

Colony Characterization
~~~~~~~~~~~~~~~~~~~~~~~

Candidate sizes are measured in pixels. Color categories and clusters describe image appearance, not biological species.

.. automethod:: tools.Candidates.measure_sizes

.. automethod:: tools.Candidates.classify_sizes

.. automethod:: tools.Candidates.classify_colors

.. automethod:: tools.Candidates.get_by_color

.. automethod:: tools.Candidates.cluster_candidates

.. automethod:: tools.Candidates.plot_clusters


Species-Based Detection
-----------------------

``Species`` contains detection settings that can be specified manually or estimated from selected candidates using ``get_species()``. ``find_colonies()`` performs a separate detection and stores the results in ``plate.colonies``. Exploratory analysis is not required before species-based detection.

.. autoclass:: tools.Species

.. automethod:: tools.Candidates.get_species

.. automethod:: tools.GeneralPlate.find_colonies

.. automethod:: tools.GeneralPlate.count

.. automethod:: tools.GeneralPlate.plot_statistics


Time-Series Analysis
--------------------

These methods are intended for ``Plate`` objects containing multiple recordings.

.. automethod:: tools.GeneralPlate.get_timeseries

.. automethod:: tools.GeneralPlate.plot_timeseries
