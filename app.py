import streamlit as st
from PIL import Image, ImageDraw
import rawpy
import io
import csv
import tempfile
import os
import numpy as np
import cv2
from colony_differentiation import tools as colony_tools
from colony_differentiation.tools import (
    detect_plate_circle, SingleImagePlate, plot_candidate_page, crop_circular_plate,
    _detect_agar_mask, crop_rectangular_plate
)
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas


def order_boundary_points(points):
    """Order an approximately convex agar boundary without crossing edges."""
    vertices = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(vertices) < 3:
        return vertices.tolist()
    center = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1],
                        vertices[:, 0] - center[0])
    return vertices[np.argsort(angles)].tolist()


@st.cache_data
def load_image(file_bytes, filename):
    """
    Load an uploaded image and convert it to RGB format.

    Returns
    Loaded RGB image.
    """
    if filename.lower().endswith((".cr2", ".cr3")):
        suffix = os.path.splitext(filename)[1].lower()

        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:
            with rawpy.imread(temp_path) as raw:
                rgb = raw.postprocess()

            image = Image.fromarray(rgb)

        finally:
            os.remove(temp_path)

    else:
        image = Image.open(io.BytesIO(file_bytes))
        image = image.convert("RGB")

    return image


st.set_page_config(
    page_title="Colony Analyzer",
    page_icon="🧫",
    layout="wide"
)

st.title("🧫 Colony Analyzer")

st.write(
    "Welcome to Colony Analyzer! "
    "Upload an agar plate image to get started."
)

upload_mode = st.radio(
    "Select upload mode",
    ["Single Image", "Image Series"],
    horizontal=True
)

uploaded_files = st.file_uploader(
    "Upload your agar plate images",
    type=["cr2", "cr3", "tif", "tiff", "png", "jpg", "jpeg"],
    accept_multiple_files=upload_mode == "Image Series",
    key=upload_mode
)

uploaded_file = None

if upload_mode == "Single Image":
    uploaded_file = uploaded_files


elif uploaded_files:
    st.write(f"{len(uploaded_files)} images uploaded")

    if "selected_image" not in st.session_state:
        st.session_state.selected_image = 0

    if st.session_state.selected_image >= len(uploaded_files):
        st.session_state.selected_image = 0

    gallery_columns = st.columns(6)

    for i, file in enumerate(uploaded_files):
        image_preview = load_image(
            file.getvalue(),
            file.name
        )

        image_preview.thumbnail((120, 120))

        with gallery_columns[i % 6]:
            st.image(
                image_preview,
                width="stretch"
            )

            if st.button(
                file.name,
                key=f"select_image_{i}",
                width="stretch"
            ):
                st.session_state.selected_image = i
                st.session_state.zoom_level = 380

    uploaded_file = uploaded_files[
        st.session_state.selected_image
    ]

    
if uploaded_file is not None:

    image = load_image(
        uploaded_file.getvalue(),
        uploaded_file.name
    )

    st.success("Image uploaded successfully!")

    st.subheader("Uploaded Agar Plate")

    if "zoom_level" not in st.session_state:
        st.session_state.zoom_level = 380

    st.session_state.zoom_level = min(st.session_state.zoom_level, 600)

    image_col, controls_col = st.columns([3, 1])

    with controls_col:

        if st.button("➕", help="Zoom in"):
            st.session_state.zoom_level = min(
                st.session_state.zoom_level + 60,
                600
            )

        if st.button("➖", help="Zoom out"):
            st.session_state.zoom_level = max(
                st.session_state.zoom_level - 60,
                180
            )

        if st.button("⟲", help="Reset zoom"):
            st.session_state.zoom_level = 380

    with image_col:

        st.image(
            image,
            width=st.session_state.zoom_level
        )

        st.caption(f"📁 {uploaded_file.name}")

    st.subheader("Plate Shape")

    plate_shape = st.radio(
        "What shape is your agar plate?",
        ["Round", "Rectangular"],
        horizontal=True,
        key="plate_shape"
    )

    st.subheader("Cropping Method")

    cropping_options = ["Automatic", "Semi-Automatic", "No Cropping"]

    cropping_method = st.radio(
        "Choose a cropping method",
        cropping_options,
        key=f"cropping_method_{plate_shape.lower()}"
    )
    
    if cropping_method == "Automatic":
        st.info("💡 Semi-Automatic cropping is recommended for more accurate results.")



    image_array = np.array(image)
    image_key = (uploaded_file.name, uploaded_file.getvalue())

    if st.session_state.get("plate_image_key") != image_key:
        st.session_state.plate_image_key = image_key
        st.session_state.plate_circle = None
        st.session_state.adjusted_circle = None
        st.session_state.plate_rectangle = None
        st.session_state.adjusted_polygon = None
        st.session_state.rect_manual_points = []
        st.session_state.applied_rectangle = None
        st.session_state.applied_polygon = None
        st.session_state.rect_canvas_version = 0
        st.session_state.canvas_version = 0
        st.session_state.applied_crop = None
        st.session_state.applied_mask = None
        st.session_state.applied_circle = None
        st.session_state.aoi_mask = None
        st.session_state.rect_aoi_crop = None
        st.session_state.rect_aoi_pending = None
        st.session_state.rect_aoi_version = 0
        st.session_state.aoi_show_original = True
        st.session_state.aoi_clicked_angles = None
        st.session_state.candidate_positions = None

    crop_choice = (plate_shape, cropping_method)
    if st.session_state.get("previous_crop_choice") != crop_choice:
        st.session_state.previous_crop_choice = crop_choice
        st.session_state.plate_rectangle = None
        st.session_state.adjusted_polygon = None
        st.session_state.rect_manual_points = []
        st.session_state.applied_rectangle = None
        st.session_state.applied_polygon = None
        st.session_state.rect_canvas_version = st.session_state.get("rect_canvas_version", 0) + 1
        st.session_state.applied_crop = None
        st.session_state.applied_mask = None
        st.session_state.applied_circle = None
        st.session_state.plate_circle = None
        st.session_state.adjusted_circle = None
        st.session_state.candidate_positions = None
        st.session_state.aoi_mask = None
        st.session_state.rect_aoi_crop = None
        st.session_state.rect_aoi_pending = None
        st.session_state.rect_aoi_version = st.session_state.get("rect_aoi_version", 0) + 1
        st.session_state.aoi_clicked_angles = None

    if plate_shape == "Rectangular" and cropping_method == "Automatic":
        if st.button("Crop Automatically", key="rect_auto_crop"):
            try:
                rect_plate = SingleImagePlate(
                    filename=image_array,
                    name=os.path.splitext(uploaded_file.name)[0]
                )
                with st.spinner("Detecting and cropping rectangular agar plate..."):
                    crop_rectangular_plate(image_array, plate=rect_plate)
                st.session_state.applied_crop = rect_plate.cropped_image
                st.session_state.applied_mask = rect_plate.selection_mask
                st.session_state.applied_rectangle = rect_plate.cropped_bounds
                st.session_state.applied_polygon = None
                st.session_state.aoi_mask = None
                st.session_state.candidate_positions = None
                st.success("Agar plate cropped successfully!")
            except Exception as exc:
                st.error(f"Automatic cropping failed: {exc}")
        if st.session_state.get("applied_crop") is not None:
            st.subheader("Cropped Agar Plate")
            st.image(st.session_state.applied_crop, width=380)

    if plate_shape == "Rectangular" and cropping_method == "Semi-Automatic":
        st.subheader("Semi-Automatic Cropping")
        st.caption(
            "Detect the inner agar boundary, then move or resize it. "
            "For individual corners, choose Draw new boundary and click points along the agar edge."
        )
        if st.button("Detect Agar Plate", key="rect_detect"):
            try:
                with st.spinner("Detecting inner agar boundary..."):
                    full_mask = _detect_agar_mask(image_array)
                    contours, _ = cv2.findContours(
                        full_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    if not contours:
                        raise ValueError("No inner agar boundary detected.")
                    contour = max(contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(contour, True)
                    points = cv2.approxPolyDP(
                        contour, max(1.5, perimeter * 0.006), True
                    ).reshape(-1, 2)
                    if len(points) > 36:
                        points = points[np.linspace(0, len(points) - 1, 36, dtype=int)]
                    if len(points) < 3:
                        raise ValueError("The detected boundary has too few points.")
                st.session_state.plate_rectangle = points.astype(float).tolist()
                st.session_state.adjusted_polygon = points.astype(float).tolist()
                st.session_state.rect_manual_points = []
                st.session_state.rect_canvas_version += 1
                st.session_state.applied_crop = None
                st.session_state.applied_mask = None
                st.session_state.applied_rectangle = None
                st.session_state.applied_polygon = None
                st.session_state.candidate_positions = None
            except Exception as exc:
                st.error(f"Could not detect agar boundary: {exc}")

        if st.session_state.get("plate_rectangle") is not None:
            scale = min(380 / image.width, 380 / image.height, 1)
            canvas_width = max(1, round(image.width * scale))
            canvas_height = max(1, round(image.height * scale))
            preview = image.resize((canvas_width, canvas_height))

            editing_mode = st.radio(
                "Boundary adjustment",
                ["Move / resize boundary", "Draw new boundary"],
                horizontal=True, key="rect_edit_mode"
            )
            drawing_new = editing_mode == "Draw new boundary"
            points = np.asarray(st.session_state.adjusted_polygon, dtype=float) * scale
            left, top = points.min(axis=0)
            local_points = points - np.array([left, top])
            initial_polygon = {
                "type": "polygon",
                "left": float(left), "top": float(top),
                "points": [
                    {"x": float(x), "y": float(y)} for x, y in local_points
                ],
                "fill": "rgba(255, 0, 0, 0.08)",
                "stroke": "#FF0000", "strokeWidth": 2,
                "objectCaching": False,
            }

            if drawing_new:
                st.caption(
                    "Click along the inner agar edge to place boundary points "
                    "(at least 3). The points are joined automatically when cropping."
                )
            else:
                st.caption("Use the canvas Edit tool to move or resize the red boundary.")

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.8)" if drawing_new else "rgba(255, 0, 0, 0.08)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=preview,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="point" if drawing_new else "rect",
                point_display_radius=4,
                return_image_data=True,
                initial_drawing={
                    "version": "4.4.0",
                    "objects": [] if drawing_new else [initial_polygon]
                },
                key=f"rect_agar_canvas_{st.session_state.rect_canvas_version}_{editing_mode}"
            )

            objects = (
                canvas_result.json_data.get("objects", [])
                if canvas_result.json_data is not None else []
            )
            if drawing_new:
                clicked_canvas = []
                for obj in objects:
                    obj_type = obj.get("type")
                    if obj_type in ("circle", "ellipse", "point"):
                        clicked_canvas.append((float(obj.get("left", 0)),
                                               float(obj.get("top", 0))))

                canvas_image = None
                if not clicked_canvas:
                    try:
                        canvas_image = canvas_result.image_data
                    except (RuntimeError, AttributeError):
                        canvas_image = None
                if not clicked_canvas and canvas_image is not None:
                    rgba = np.asarray(canvas_image)
                    if rgba.ndim == 3 and rgba.shape[2] >= 3:
                        rgb_canvas = rgba[:, :, :3].astype(np.int16)
                        base_rgb = np.asarray(preview.convert("RGB"), dtype=np.int16)
                        if rgb_canvas.shape[:2] == base_rgb.shape[:2]:
                            red = (rgb_canvas[:, :, 0] > 170) & \
                                  (rgb_canvas[:, :, 1] < 120) & \
                                  (rgb_canvas[:, :, 2] < 120)
                            different = np.max(np.abs(rgb_canvas - base_rgb), axis=2) > 45
                            marks = (red & different).astype(np.uint8)
                            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                                marks, connectivity=8
                            )
                            for i in range(1, count):
                                area = int(stats[i, cv2.CC_STAT_AREA])
                                if 8 <= area <= 350:
                                    clicked_canvas.append(tuple(centroids[i]))

                if clicked_canvas:
                    clicked_points = [
                        [float(np.clip(px / scale, 0, image.width - 1)),
                         float(np.clip(py / scale, 0, image.height - 1))]
                        for px, py in clicked_canvas
                    ]
                    st.session_state.rect_manual_points = order_boundary_points(
                        clicked_points
                    )
                manual_points = st.session_state.get("rect_manual_points", [])
                st.caption(f"Boundary points selected: {len(manual_points)} (minimum 3)")
                if len(manual_points) >= 3:
                    overlay = np.asarray(preview).copy()
                    manual_points = order_boundary_points(manual_points)
                    outline = np.rint(np.asarray(manual_points) * scale).astype(np.int32)
                    cv2.polylines(overlay, [outline], True, (0, 255, 255), 2)
                    st.image(overlay, width=canvas_width, caption="New boundary preview (cyan)")
                if st.button("Clear drawn points", key="rect_clear_drawn"):
                    st.session_state.rect_manual_points = []
                    st.session_state.rect_canvas_version += 1
                    st.rerun()
            elif objects:
                selected = next((obj for obj in reversed(objects)
                                 if obj.get("type") == "polygon"), None)
                if selected is not None:
                    polygon_points = np.asarray([
                        [pt["x"], pt["y"]] for pt in selected.get("points", [])
                    ], dtype=float)
                    if len(polygon_points) >= 3:
                        min_xy = polygon_points.min(axis=0)
                        sx = float(selected.get("scaleX", 1))
                        sy = float(selected.get("scaleY", 1))
                        obj_left = float(selected.get("left", 0))
                        obj_top = float(selected.get("top", 0))
                        angle = float(selected.get("angle", 0))
                        if abs(angle) < 0.01:
                            canvas_xy = np.column_stack([
                                obj_left + (polygon_points[:, 0] - min_xy[0]) * sx,
                                obj_top + (polygon_points[:, 1] - min_xy[1]) * sy,
                            ])
                            vertices = canvas_xy / scale
                            vertices[:, 0] = np.clip(vertices[:, 0], 0, image.width - 1)
                            vertices[:, 1] = np.clip(vertices[:, 1], 0, image.height - 1)
                            st.session_state.adjusted_polygon = vertices.tolist()
                        else:
                            st.warning("Rotation is not supported. Move or resize the boundary instead.")

            if st.button("Reset Boundary", key="rect_reset"):
                st.session_state.adjusted_polygon = st.session_state.plate_rectangle
                st.session_state.rect_manual_points = []
                st.session_state.rect_canvas_version += 1
                st.session_state.applied_crop = None
                st.session_state.applied_mask = None
                st.session_state.applied_rectangle = None
                st.session_state.applied_polygon = None
                st.rerun()

            if st.button("Apply & Crop", key="rect_apply", type="primary"):
                try:
                    if drawing_new:
                        polygon = order_boundary_points(
                            st.session_state.get("rect_manual_points", [])
                        )
                        if len(polygon) < 3:
                            raise ValueError(
                                "Place at least three boundary points before applying the crop."
                            )
                    else:
                        polygon = st.session_state.get("adjusted_polygon")
                        if polygon is None or len(polygon) < 3:
                            raise ValueError("No valid boundary selected.")
                    rect_plate = SingleImagePlate(
                        filename=image_array,
                        name=os.path.splitext(uploaded_file.name)[0]
                    )
                    with st.spinner("Applying boundary and cropping agar plate..."):
                        crop_rectangular_plate(
                            image_array, plate=rect_plate, manual_polygon=polygon
                        )
                    st.session_state.applied_crop = rect_plate.cropped_image
                    st.session_state.applied_mask = rect_plate.selection_mask
                    st.session_state.applied_rectangle = rect_plate.cropped_bounds
                    st.session_state.applied_polygon = np.asarray(polygon, dtype=float).tolist()
                    st.session_state.aoi_mask = None
                    st.session_state.candidate_positions = None
                    st.success("Agar plate cropped successfully!")
                except Exception as exc:
                    st.error(f"Could not apply the boundary: {exc}")

            st.write("**Cropped Agar Plate**")
            if st.session_state.get("applied_crop") is not None:
                st.image(st.session_state.applied_crop, width=380)
            else:
                st.caption("Click Apply & Crop to display the result here.")

    if plate_shape == "Round" and cropping_method == "Automatic":

        if st.button("Crop Automatically", key="auto_crop"):

            plate = SingleImagePlate(
                filename=image_array,
                name=os.path.splitext(uploaded_file.name)[0]
            )

            try:
                with st.spinner("Detecting and cropping round agar plate..."):
                    plate.crop_circular_plate()
                if getattr(plate, "cropped_image", None) is not None:
                    st.session_state.applied_crop = plate.cropped_image
                    st.session_state.applied_mask = plate.selection_mask
                    st.session_state.applied_circle = detect_plate_circle(image_array)
                    st.session_state.aoi_mask = None
                    st.session_state.candidate_positions = None
                    st.success("Agar plate cropped successfully!")
                else:
                    st.warning("Automatic cropping failed.")
            except Exception as exc:
                st.error(f"Automatic cropping failed: {exc}")

        if st.session_state.get("applied_crop") is not None:
            st.subheader("Cropped Agar Plate")
            st.image(st.session_state.applied_crop, width=380)

    if plate_shape == "Round" and cropping_method == "Semi-Automatic":
        st.subheader("Semi-Automatic Cropping")

        if st.button("Detect Agar Plate"):
            with st.spinner("Detecting round agar plate..."):
                result = detect_plate_circle(image_array)

            if result is None:
                st.warning("No agar plate detected.")
                st.session_state.plate_circle = None
                st.session_state.applied_crop = None
                st.session_state.applied_mask = None
                st.session_state.applied_circle = None
            else:
                st.session_state.plate_circle = result
                st.session_state.adjusted_circle = result
                st.session_state.canvas_version += 1
                st.session_state.applied_crop = None
                st.session_state.applied_mask = None
                st.session_state.applied_circle = None

        if st.session_state.plate_circle is not None:
            x, y, radius = st.session_state.plate_circle

            scale = min(380 / image.width, 380 / image.height, 1)
            canvas_width = round(image.width * scale)
            canvas_height = round(image.height * scale)

            preview = image.resize((canvas_width, canvas_height))

            circle_col, crop_col = st.columns(2, gap="medium")

            with circle_col:
                st.write("**Adjust Agar Plate Boundary**")

                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 0)",
                    stroke_width=3,
                    stroke_color="#FF0000",
                    background_image=preview,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    initial_drawing={
                        "version": "4.4.0",
                        "objects": [
                            {
                                "type": "circle",
                                "left": x * scale,
                                "top": y * scale,
                                "originX": "center",
                                "originY": "center",
                                "radius": radius * scale,
                                "fill": "rgba(0, 0, 0, 0)",
                                "stroke": "#FF0000",
                                "strokeWidth": 3
                            }
                        ]
                    },
                    key=f"plate_canvas_{st.session_state.canvas_version}"
                )

                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])

                    if objects:
                        selected = objects[0]

                        left = selected["left"]
                        top = selected["top"]
                        original_radius = selected["radius"]

                        scale_x = selected.get("scaleX", 1)
                        scale_y = selected.get("scaleY", 1)

                        adjusted_x = left / scale
                        adjusted_y = top / scale
                        adjusted_radius = (
                            original_radius * (scale_x + scale_y) / (2 * scale)
                        )

                        st.session_state.adjusted_circle = (
                            round(adjusted_x),
                            round(adjusted_y),
                            round(adjusted_radius)
                        )

                if st.session_state.adjusted_circle is not None:
                    st.write(
                        "Selected circle:",
                        st.session_state.adjusted_circle
                    )

                reset_col, apply_col, _aoi_button_space = st.columns([1.2, 1.2, 3.6])

                with reset_col:
                    if st.button("Reset Circle"):
                        st.session_state.adjusted_circle = st.session_state.plate_circle
                        st.session_state.canvas_version += 1
                        st.session_state.applied_crop = None
                        st.session_state.applied_mask = None
                        st.session_state.applied_circle = None
                        st.rerun()


                with apply_col:
                    if st.button("Apply & Crop", type="primary"):
                        selected = st.session_state.adjusted_circle
                        if selected is None:
                            st.warning("Please select a circle first.")
                        else:
                            cx, cy, selected_radius = map(int, selected)
                            height, width = image_array.shape[:2]
                            x_min = max(0, cx - selected_radius)
                            x_max = min(width, cx + selected_radius)
                            y_min = max(0, cy - selected_radius)
                            y_max = min(height, cy + selected_radius)

                            if selected_radius <= 0 or x_min >= x_max or y_min >= y_max:
                                st.error("The selected circle is outside the image or invalid.")
                            else:
                                with st.spinner("Applying circle and cropping agar plate..."):
                                    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                                    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= selected_radius ** 2
                                    cropped = image_array[y_min:y_max, x_min:x_max].copy()
                                    cropped[~circle_mask] = 0

                                    full_mask = np.zeros((height, width), dtype=bool)
                                    full_mask[y_min:y_max, x_min:x_max] = circle_mask

                                st.session_state.applied_crop = cropped
                                st.session_state.applied_mask = full_mask
                                st.session_state.applied_circle = (cx, cy, selected_radius)
                                st.session_state.aoi_mask = None
                                st.session_state.candidate_positions = None
                                st.toast("Agar plate cropped successfully!", icon="✅")

            with crop_col:
                st.write("**Cropped Agar Plate**")
                if st.session_state.get("applied_crop") is not None:
                    st.image(st.session_state.applied_crop, width=canvas_width)
                    st.write("Applied circle:", st.session_state.applied_circle)
                else:
                    st.caption("Click Apply & Crop to display the result here.")



    base_mask = st.session_state.get("applied_mask")
    has_crop = base_mask is not None
    if has_crop and plate_shape == "Round":
        st.subheader("Area of Interest")
        use_aoi = st.checkbox(
            "Select a sector of the cropped plate",
            key="use_sector_aoi"
        )

        if use_aoi:
            if "aoi_reset_version" not in st.session_state:
                st.session_state.aoi_reset_version = 0

            if st.button("Reset Area of Interest", key="reset_aoi"):
                st.session_state.aoi_mask = None
                st.session_state.last_aoi_angles = None
                st.session_state.aoi_clicked_angles = None
                st.session_state.candidate_positions = None
                st.session_state.candidate_intensities = None
                st.session_state.aoi_reset_version += 1
                for old_key in list(st.session_state.keys()):
                    if old_key.startswith(("aoi_points_", "aoi_start_angle_", "aoi_end_angle_")):
                        del st.session_state[old_key]
                st.session_state.aoi_show_original = True
                st.rerun()

            selection_mode = st.radio(
                "Select sector by",
                ["Enter angles", "Click two points"],
                horizontal=True,
                key="aoi_selection_mode"
            )

            circle = st.session_state.get("applied_circle")
            if circle is None:
                st.warning("Plate center is missing. Please crop the plate again.")
            else:
                cx, cy, radius = map(int, circle)
                start_angle = None
                end_angle = None

                ys, xs = np.where(np.asarray(base_mask, dtype=bool))
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                local_rgb = image_array[y0:y1, x0:x1].copy()
                local_mask = np.asarray(base_mask[y0:y1, x0:x1], dtype=bool)
                active_aoi = st.session_state.get("aoi_mask")
                if active_aoi is not None and not st.session_state.get("aoi_show_original", False):
                    local_mask = local_mask & np.asarray(
                        active_aoi[y0:y1, x0:x1], dtype=bool
                    )
                local_rgb[~local_mask] = 0

                display_size = 380
                margin = 35
                available = display_size - 2 * margin
                scale_aoi = min(available / (x1 - x0), available / (y1 - y0))
                rendered_w = max(1, round((x1 - x0) * scale_aoi))
                rendered_h = max(1, round((y1 - y0) * scale_aoi))
                offset_x = (display_size - rendered_w) // 2
                offset_y = (display_size - rendered_h) // 2

                preview = Image.new("RGB", (display_size, display_size), (18, 18, 18))
                resized_plate = Image.fromarray(local_rgb).resize(
                    (rendered_w, rendered_h), Image.Resampling.LANCZOS
                )
                preview.paste(resized_plate, (offset_x, offset_y))
                draw = ImageDraw.Draw(preview)
                display_cx = offset_x + (cx - x0) * scale_aoi
                display_cy = offset_y + (cy - y0) * scale_aoi
                display_radius = radius * scale_aoi

                for degree in range(0, 360, 30):
                    radians = np.deg2rad(degree)
                    px = display_cx + (display_radius + 15) * np.cos(radians)
                    py = display_cy - (display_radius + 15) * np.sin(radians)
                    label = f"{degree}°"
                    bbox = draw.textbbox((0, 0), label)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text((px - tw / 2, py - th / 2), label, fill="cyan")

                draw.ellipse(
                    (display_cx - 2, display_cy - 2,
                     display_cx + 2, display_cy + 2),
                    fill="cyan"
                )

                if selection_mode == "Enter angles":
                    st.image(preview, width=display_size)
                    angle_col1, angle_col2 = st.columns(2)
                    with angle_col1:
                        start_angle = st.number_input(
                            "Start angle (degrees)", min_value=0,
                            max_value=359, value=30, step=1,
                            key=f"aoi_start_angle_{st.session_state.aoi_reset_version}"
                        )
                    with angle_col2:
                        end_angle = st.number_input(
                            "End angle (degrees)", min_value=0,
                            max_value=359, value=120, step=1,
                            key=f"aoi_end_angle_{st.session_state.aoi_reset_version}"
                        )
                else:
                    st.write("Click two points on the plate shown with angle labels.")
                    point_canvas = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.8)",
                        stroke_color="#FF0000",
                        stroke_width=2,
                        background_image=preview,
                        drawing_mode="point",
                        point_display_radius=5,
                        update_streamlit=True,
                        height=display_size,
                        width=display_size,
                        key=(f"aoi_points_{st.session_state.canvas_version}_"
                             f"{st.session_state.plate_image_key[0]}_"
                             f"{st.session_state.aoi_reset_version}")
                    )
                    objects = (point_canvas.json_data or {}).get("objects", [])
                    if len(objects) >= 2:
                        angles = []
                        for obj in objects[-2:]:
                            px = (float(obj.get("left", 0)) - offset_x) / scale_aoi + x0
                            py = (float(obj.get("top", 0)) - offset_y) / scale_aoi + y0
                            angle = (
                                np.degrees(np.arctan2(cy - py, px - cx)) + 360
                            ) % 360
                            angles.append(int(round(angle)) % 360)
                        new_angles = tuple(angles)
                        if st.session_state.get("aoi_clicked_angles") != new_angles:
                            st.session_state.aoi_clicked_angles = new_angles
                            st.session_state.aoi_mask = None
                            st.session_state.aoi_show_original = True
                            st.session_state.candidate_positions = None

                    saved_angles = st.session_state.get("aoi_clicked_angles")
                    if saved_angles is not None:
                        start_angle, end_angle = saved_angles
                        st.caption(f"Selected angles: {start_angle}° and {end_angle}°")
                    else:
                        st.caption("Select two points to define the sector.")

                if selection_mode == "Enter angles":
                    angle_choice = (selection_mode, start_angle, end_angle)
                    if st.session_state.get("last_aoi_angles") != angle_choice:
                        st.session_state.last_aoi_angles = angle_choice
                        st.session_state.aoi_mask = None
                        st.session_state.aoi_show_original = True
                        st.session_state.candidate_positions = None

                if start_angle is not None and end_angle is not None:
                    if start_angle == end_angle:
                        st.warning("Choose two different angles.")
                    elif st.button("Apply Area of Interest", key="apply_aoi"):
                        try:
                            aoi_plate = SingleImagePlate(
                                filename=image_array,
                                name=os.path.splitext(uploaded_file.name)[0]
                            )
                            crop_circular_plate(
                                image_array,
                                start_angle=int(start_angle),
                                end_angle=int(end_angle),
                                center=(cx, cy), radius=radius,
                                plate=aoi_plate, show_plot=True
                            )
                            plt.close("all")
                            st.session_state.aoi_mask = (
                                np.asarray(aoi_plate.selection_mask, dtype=bool)
                                & np.asarray(base_mask, dtype=bool)
                            )
                            st.session_state.candidate_positions = None
                            st.session_state.aoi_show_original = False
                            st.success("Area of Interest applied.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Area of Interest failed: {exc}")

        else:
            if st.session_state.get("aoi_mask") is not None:
                st.session_state.aoi_mask = None
                st.session_state.aoi_clicked_angles = None
                st.session_state.candidate_positions = None

    if plate_shape == "Rectangular" and (has_crop or cropping_method == "No Cropping"):
        st.subheader("Area of Interest")
        use_rect_aoi = st.checkbox(
            "Select a rectangular area of interest", key="use_rectangular_aoi"
        )
        if use_rect_aoi:
            if has_crop and cropping_method != "No Cropping":
                source_image = np.asarray(st.session_state.applied_crop)
                crop_bounds = st.session_state.get("applied_rectangle")
                if crop_bounds is None:
                    yy, xx = np.where(np.asarray(base_mask, dtype=bool))
                    origin_x, origin_y = int(xx.min()), int(yy.min())
                else:
                    origin_x, origin_y = map(int, crop_bounds[:2])
            else:
                source_image = image_array
                origin_x = origin_y = 0

            src_h, src_w = source_image.shape[:2]
            display_scale = min(520 / src_w, 440 / src_h, 1.0)
            disp_w = max(1, round(src_w * display_scale))
            disp_h = max(1, round(src_h * display_scale))
            background = Image.fromarray(source_image.astype(np.uint8)).resize(
                (disp_w, disp_h), Image.Resampling.LANCZOS
            )
            st.caption("Draw a rectangle with the mouse, then click Apply Area of Interest.")
            rect_canvas = st_canvas(
                background_image=background,
                drawing_mode="rect",
                fill_color="rgba(0, 220, 255, 0.12)",
                stroke_color="#00DCFF",
                stroke_width=2,
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                key=f"rect_aoi_canvas_{st.session_state.get('rect_aoi_version', 0)}",
                return_image_data=True,
            )
            shapes = (rect_canvas.json_data or {}).get("objects", [])

            rectangles = [obj for obj in shapes if obj.get("type") == "rect"]
            if rectangles:
                st.session_state.rect_aoi_pending = dict(rectangles[-1])
            elif rect_canvas.image_data is not None:
                rgba = np.asarray(rect_canvas.image_data)
                if rgba.ndim == 3 and rgba.shape[2] >= 3:
                    rgb = rgba[:, :, :3].astype(np.int16)
                    cyan = ((rgb[:, :, 0] < 110) &
                            (rgb[:, :, 1] > 145) &
                            (rgb[:, :, 2] > 145))
                    if rgba.shape[2] == 4:
                        cyan &= rgba[:, :, 3] > 30
                    ys, xs = np.where(cyan)
                    if len(xs) >= 12 and xs.max() - xs.min() >= 5 and ys.max() - ys.min() >= 5:
                        st.session_state.rect_aoi_pending = {
                            "type": "rect",
                            "left": float(xs.min()), "top": float(ys.min()),
                            "width": float(xs.max() - xs.min()),
                            "height": float(ys.max() - ys.min()),
                            "scaleX": 1, "scaleY": 1
                        }
            reset_col, apply_col, _aoi_button_space = st.columns([1.2, 1.2, 3.6])
            with reset_col:
                reset_aoi_clicked = st.button(
                    "Reset Area of Interest", key="reset_rect_aoi",
                    use_container_width=False
                )
            with apply_col:
                apply_aoi_clicked = st.button(
                    "Apply Area of Interest", key="apply_rect_aoi",
                    type="primary", use_container_width=False
                )

            if reset_aoi_clicked:
                st.session_state.aoi_mask = None
                st.session_state.rect_aoi_crop = None
                st.session_state.rect_aoi_pending = None
                st.session_state.candidate_positions = None
                st.session_state.rect_aoi_version = st.session_state.get("rect_aoi_version", 0) + 1
                st.rerun()

            if apply_aoi_clicked:
                obj = st.session_state.get("rect_aoi_pending")
                if obj is None:
                    st.warning("Draw a rectangle first.")
                else:
                    x_a = float(obj.get("left", 0))
                    y_a = float(obj.get("top", 0))
                    x_b = x_a + float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
                    y_b = y_a + float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
                    x1 = max(0, min(src_w, int(np.floor(min(x_a, x_b) / display_scale))))
                    x2 = max(0, min(src_w, int(np.ceil(max(x_a, x_b) / display_scale))))
                    y1 = max(0, min(src_h, int(np.floor(min(y_a, y_b) / display_scale))))
                    y2 = max(0, min(src_h, int(np.ceil(max(y_a, y_b) / display_scale))))
                    if x2 <= x1 or y2 <= y1:
                        st.warning("Select a non-empty rectangle.")
                    else:
                        full_mask = np.zeros(image_array.shape[:2], dtype=bool)
                        fx1, fx2 = origin_x + x1, origin_x + x2
                        fy1, fy2 = origin_y + y1, origin_y + y2
                        full_mask[fy1:fy2, fx1:fx2] = True
                        if has_crop and cropping_method != "No Cropping":
                            full_mask &= np.asarray(base_mask, dtype=bool)
                        st.session_state.aoi_mask = full_mask
                        st.session_state.rect_aoi_crop = source_image[y1:y2, x1:x2].copy()
                        st.session_state.candidate_positions = None
                        st.success("Area of Interest applied.")
            if st.session_state.get("rect_aoi_crop") is not None:
                st.write("**Selected Area of Interest**")
                st.image(st.session_state.rect_aoi_crop, width=260)
        else:
            if st.session_state.get("aoi_mask") is not None:
                st.session_state.aoi_mask = None
                st.session_state.rect_aoi_crop = None
                st.session_state.candidate_positions = None

    can_find_candidates = (cropping_method == "No Cropping") or has_crop
    if can_find_candidates:
        st.subheader("Colony Candidates")

        candidate_count = st.number_input(
            "Number of candidates",
            min_value=1,
            value=50,
            step=1
        )

        all_candidates = st.checkbox(
            "All candidates"
        )

        if all_candidates:
            candidate_count = None

        use_custom_parameters = st.checkbox(
            "Adjust radius and smoothing"
        )

        candidate_radius = None
        candidate_sigma = None

        if use_custom_parameters:

            col1, col2 = st.columns(2)

            with col1:
                candidate_radius = st.number_input(
                    "Candidate radius",
                    min_value=1,
                    value=10,
                    step=1
                )

            with col2:
                candidate_sigma = st.number_input(
                    "Smoothing (sigma)",
                    min_value=0.1,
                    value=2.0,
                    step=0.1
                )

        button_col, progress_col = st.columns([1, 3])

        with button_col:
            find_clicked = st.button(
                "Find Candidates",
                use_container_width=True
            )

        with progress_col:
            progress_placeholder = st.empty()

        if find_clicked:

            try:
                raw_detection = uploaded_file.name.lower().endswith((".cr2", ".cr3"))
                raw_temp_path = None
                if raw_detection:
                        with tempfile.NamedTemporaryFile(
                            suffix=os.path.splitext(uploaded_file.name)[1],
                            delete=False
                        ) as raw_temp:
                            raw_temp.write(uploaded_file.getvalue())
                            raw_temp_path = raw_temp.name
                        plate = SingleImagePlate(
                            filename=raw_temp_path,
                            name=os.path.splitext(uploaded_file.name)[0]
                        )
                else:
                    plate = SingleImagePlate(
                        filename=image_array,
                        name=os.path.splitext(uploaded_file.name)[0]
                    )

                parameters = {}

                parameters["n"] = None if all_candidates else int(candidate_count)

                if candidate_radius is not None:
                    parameters["radius"] = int(candidate_radius)

                if candidate_sigma is not None:
                    parameters["sigma"] = float(candidate_sigma)

                progress_bar = progress_placeholder.progress(
                    0, text="Finding candidates... 0%"
                )
                last_progress = [-1]

                def update_progress(value):
                    """Update the progress bar in five-percent increments."""
                    value = max(0, min(100, int(value)))
                    if value >= last_progress[0] + 5 or value == 100:
                        progress_bar.progress(
                            value,
                            text=(
                                "Candidates found!" if value == 100
                                else f"Finding candidates... {value}%"
                            ),
                        )
                        last_progress[0] = value

                parameters["progress_callback"] = update_progress

                full_selection = st.session_state.get("aoi_mask")
                if full_selection is None and cropping_method != "No Cropping":
                    full_selection = st.session_state.get("applied_mask")
                active_mask = None
                if full_selection is not None:
                    full_selection = np.asarray(full_selection, dtype=bool)
                    if full_selection.shape != image_array.shape[:2]:
                        raise ValueError("Selection mask and uploaded image dimensions differ.")
                    if raw_detection:
                        with rawpy.imread(raw_temp_path) as raw:
                            half_rgb = raw.postprocess(half_size=True)
                        half_rgb = colony_tools._transpose_if_needed(half_rgb)
                        half_h, half_w = half_rgb.shape[:2]
                        oriented = full_selection
                        if (image_array.shape[0] > image_array.shape[1]
                                and half_h < half_w):
                            oriented = np.flip(oriented.T, axis=0)
                        half_mask = cv2.resize(
                            oriented.astype(np.uint8), (half_w, half_h),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        a, b, c, d = (colony_tools.a, colony_tools.b,
                                      colony_tools.c, colony_tools.d)
                        active_mask = half_mask[b:d, a:c]
                    else:
                        active_mask = full_selection
                    if active_mask.shape != plate.rgb_image.shape[:2]:
                        raise ValueError(
                            "AOI coordinates do not match the RAW detection image. "
                            f"Mask: {active_mask.shape}, image: {plate.rgb_image.shape[:2]}"
                        )
                    if not np.any(active_mask):
                        raise ValueError("The selected AOI is outside the detection image.")
                    plate.selection_mask = active_mask

                plate.find_candidates(**parameters)

                detected = list(plate.candidates)
                detected_intensities = list(plate.candidate_intensities)
                if active_mask is not None:
                    height, width = active_mask.shape
                    valid = [
                        idx for idx, (row, col) in enumerate(detected)
                        if 0 <= int(row) < height and 0 <= int(col) < width
                        and active_mask[int(row), int(col)]
                    ]
                    detected = [detected[idx] for idx in valid]
                    detected_intensities = [detected_intensities[idx] for idx in valid]

                st.session_state.analysis_selection_mask = (active_mask.copy() if active_mask is not None else None)
                st.session_state.analysis_results = {}
                st.session_state.candidate_positions = detected
                st.session_state.candidate_intensities = detected_intensities
                st.session_state.candidate_rgb_image = plate.rgb_image.copy()
                st.session_state.candidates_visible = min(100, len(detected))
                plt.close("all")

                st.success("Candidate detection completed!")

            except Exception as e:
                st.error(
                    f"Candidate detection failed: {e}"
                )
            finally:
                if "raw_temp_path" in locals() and raw_temp_path is not None:
                    try:
                        os.remove(raw_temp_path)
                    except OSError:
                        pass

        if st.session_state.get("candidate_positions") is not None:
            positions = st.session_state.candidate_positions
            total = len(positions)
            visible = min(st.session_state.get("candidates_visible", 100), total)

            st.write("Candidates found:", total)
            if total > 100:
                st.info(
                    f"Showing {visible} of {total} candidates. "
                    "Click 'Load 100 more' to see additional candidates."
                )


            for page_start in range(0, visible, 100):
                fig = plot_candidate_page(
                    st.session_state.candidate_rgb_image,
                    positions,
                    intensities=st.session_state.get("candidate_intensities"),
                    start=page_start,
                    count=min(100, visible - page_start),
                )
                if fig is not None:
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
                    plt.close(fig)
                    buffer.seek(0)
                    st.image(buffer, width=1100)

            if visible < total:
                if st.button("Load 100 more", key="load_more_candidates"):
                    st.session_state.candidates_visible = min(visible + 100, total)
                    st.rerun()

            st.divider()
            st.subheader("Further Analysis")
            tab_cluster, tab_size, tab_color, tab_explore = st.tabs(
                ["Clustering", "Size Analysis", "Color Classification", "Colony Explorer & Export"]
            )
            from types import SimpleNamespace
            from colony_differentiation.tools import Candidates
            import contextlib

            analysis_plate = SimpleNamespace(
                rgb_image=st.session_state.candidate_rgb_image,
                selection_mask=st.session_state.get("analysis_selection_mask"),
                cropped_bounds=None,
            )
            analysis_candidates = Candidates(analysis_plate)
            analysis_candidates.extend(positions)
            results_store = st.session_state.setdefault("analysis_results", {})

            def display_analysis_figure(fig):
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=115, bbox_inches="tight")
                plt.close(fig)
                buffer.seek(0)
                st.image(buffer, width=650)

            def analysis_image_axes(title):
                image = analysis_plate.rgb_image
                h, w = image.shape[:2]
                x1, y1, x2, y2 = 0, 0, w, h
                mask = analysis_plate.selection_mask
                if mask is not None and np.asarray(mask).shape == (h, w) and np.any(mask):
                    yy, xx = np.where(mask)
                    x1, x2 = int(xx.min()), int(xx.max()) + 1
                    y1, y2 = int(yy.min()), int(yy.max()) + 1
                fig, ax = plt.subplots(figsize=(7.2, 4.3))
                ax.imshow(image[y1:y2, x1:x2], extent=(x1, x2, y2, y1))
                ax.set_xlim(x1, x2)
                ax.set_ylim(y2, y1)
                ax.set_title(title)
                ax.set_xlabel("X (pixels)")
                ax.set_ylabel("Y (pixels)")
                return fig, ax

            with tab_cluster:
                st.caption("Groups colonies by similarity of their measured colors. You can name each group after identifying its species.")
                c1, c2, c3 = st.columns(3)
                with c1:
                    n_cluster = st.number_input("Candidates to cluster", min_value=1,
                        max_value=max(1, total), value=min(50, total), key="dbscan_count")
                with c2:
                    eps = st.number_input("DBSCAN eps", min_value=0.1,
                        value=20.0, step=1.0, key="dbscan_eps")
                with c3:
                    min_samples = st.number_input("Minimum samples", min_value=1,
                        value=3, step=1, key="dbscan_min_samples")
                st.caption("**DBSCAN eps:** Maximum RGB-color distance for nearby colonies to be grouped. Smaller values separate colors more strictly; larger values merge more colors.")
                st.caption("**Minimum samples:** Minimum number of nearby colonies needed to form a dense group. Colonies outside groups are marked as outliers.")
                if st.button("Run Clustering", key="run_dbscan", type="primary"):
                    try:
                        details = analysis_candidates.cluster_candidates(
                            n_candidates=int(n_cluster), eps=float(eps),
                            min_samples=int(min_samples), return_details=True,
                            verbose=False)
                        results_store["cluster"] = details
                    except Exception as exc:
                        st.error(f"Clustering failed: {exc}")
                details = results_store.get("cluster")
                if details is not None:
                    st.write(f"**Clusters:** {details['n_clusters']} · "
                             f"**Analyzed colonies:** {details['n_candidates']} · "
                             f"**Outliers:** {details['noise_count']}")
                    if details["cluster_counts"]:
                        st.markdown("**Cluster sizes:** " + " · ".join(
                            f"**Cluster {cid}:** {count}"
                            for cid, count in sorted(details["cluster_counts"].items())))
                    cluster_names = {}
                    for cluster_id in sorted(details["cluster_counts"]):
                        cluster_names[cluster_id] = st.text_input(
                            f"Species name for Cluster {cluster_id}",
                            key=f"species_cluster_{cluster_id}",
                            placeholder="Enter species name (optional)")
                    try:
                        fig, ax = analysis_candidates.plot_clusters(
                            result=details, figsize=(7.2, 4.2),
                            marker_size=65, show_numbers=False, show=False)
                        if cluster_names:
                            from matplotlib.lines import Line2D
                            handles, labels_legend = ax.get_legend_handles_labels()
                            legend = ax.get_legend()
                            if legend is not None:
                                for t in legend.get_texts():
                                    label = t.get_text()
                                    for cid, species in cluster_names.items():
                                        if species.strip() and label.startswith(f"Cluster {cid} ("):
                                            t.set_text(f"Cluster {cid}: {species.strip()} ({details['cluster_counts'][cid]})")
                        display_analysis_figure(fig)
                    except Exception as exc:
                        st.error(f"Cluster plot failed: {exc}")

            with tab_size:
                st.caption("Estimates colony diameter in image pixels and groups colonies into small, medium and large.")
                c1, c2, c3 = st.columns(3)
                with c1:
                    max_radius = st.number_input("Maximum measurement radius (px)",
                        min_value=2, value=15, key="size_radius")
                with c2:
                    small_limit = st.number_input("Small: upper boundary (px)",
                        min_value=1.0, value=10.0, key="size_small")
                with c3:
                    large_limit = st.number_input("Large: lower boundary (px)",
                        min_value=1.0, value=15.0, key="size_large")
                st.caption("**Maximum measurement radius:** Maximum distance searched outward from the colony center to estimate its edge. Increase it for larger colonies; results are diameters in pixels.")
                if st.button("Analyze Sizes", key="run_sizes", type="primary"):
                    if large_limit <= small_limit:
                        st.error("The large threshold must exceed the small threshold.")
                    else:
                        try:
                            with contextlib.redirect_stdout(io.StringIO()):
                                measurements = analysis_candidates.measure_sizes(
                                    max_radius=int(max_radius))
                            results_store["sizes"] = np.asarray(measurements).copy()
                        except Exception as exc:
                            st.error(f"Size analysis failed: {exc}")
                measurements = results_store.get("sizes")
                if measurements is not None:
                    valid = np.isfinite(measurements)
                    small = int(np.sum(valid & (measurements < small_limit)))
                    medium = int(np.sum(valid & (measurements >= small_limit)
                                        & (measurements < large_limit)))
                    large = int(np.sum(valid & (measurements >= large_limit)))
                    st.write(f"**Measured:** {int(valid.sum())} / {total} · "
                             f"**Small:** {small} · **Medium:** {medium} · **Large:** {large}")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.1),
                                                   gridspec_kw={"width_ratios": [2.2, 1]})
                    indices = np.arange(len(measurements))
                    palette = {"Small": "#58b9e6", "Medium": "#f2be4e", "Large": "#f28360"}
                    for label, mask in [
                        ("Small", valid & (measurements < small_limit)),
                        ("Medium", valid & (measurements >= small_limit) & (measurements < large_limit)),
                        ("Large", valid & (measurements >= large_limit))]:
                        ax1.scatter(indices[mask], measurements[mask], s=22,
                                    color=palette[label], label=label)
                    ax1.axhline(small_limit, color="gray", ls="--", alpha=.6)
                    ax1.axhline(large_limit, color="gray", ls="--", alpha=.6)
                    ax1.set(xlabel="Colony index", ylabel="Colony diameter (px)",
                            title="Colony Size Distribution")
                    ax1.grid(alpha=.18)
                    ax1.legend(fontsize=8)
                    counts = [small, medium, large]
                    percentages = [100 * c / max(1, int(valid.sum())) for c in counts]
                    ax2.bar(["Small", "Medium", "Large"], percentages,
                            color=list(palette.values()))
                    for i, (pct, count) in enumerate(zip(percentages, counts)):
                        ax2.text(i, pct + 1, f"{pct:.1f}%\n(n={count})",
                                 ha="center", fontsize=8)
                    ax2.set(ylabel="Percentage (%)", title="Size Classes")
                    ax2.set_ylim(0, max(percentages) + 18)
                    ax2.grid(axis="y", alpha=.18)
                    fig.tight_layout()
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", dpi=135, bbox_inches="tight")
                    plt.close(fig)
                    buffer.seek(0)
                    st.image(buffer, width=850)

            with tab_color:
                if st.button("Classify Colors", key="run_colors", type="primary"):
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            analysis_candidates.classify_colors()
                        results_store["colors"] = dict(analysis_candidates.colors)
                    except Exception as exc:
                        st.error(f"Color classification failed: {exc}")
                color_groups = results_store.get("colors")
                if color_groups is not None:
                    st.write(" · ".join(f"**{name.title()}:** {len(items)}"
                                        for name, items in color_groups.items()
                                        if len(items) > 0 or name.lower() == "unknown"))
                    color_species = {}
                    for name, coords in color_groups.items():
                        if coords:
                            color_species[name] = st.text_input(
                                f"Species name for {name.title()} colonies",
                                key=f"species_color_{name}",
                                placeholder="Enter species name (optional)")
                    fig, ax = analysis_image_axes("Color classification")
                    palette = {"white": "#f4f4f4", "yellow": "#ffce00",
                               "green": "#23d160", "red": "#fa5252",
                               "blue": "#339af0", "unknown": "#9e9e9e"}
                    for name, coords in color_groups.items():
                        if not coords:
                            continue
                        arr = np.asarray(coords)
                        ax.scatter(arr[:, 1], arr[:, 0], s=125, facecolors="none",
                                   edgecolors=palette[name], linewidths=2,
                                   label=f"{name.title()}" + (f" — {color_species[name].strip()}" if color_species.get(name, "").strip() else "") + f" ({len(coords)})")
                    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
                    display_analysis_figure(fig)

            with tab_explore:
                st.caption("Inspect individual detected colonies and export your analysis results.")
                color_groups = results_store.get("colors")
                color_by_position = {}
                if color_groups:
                    for group, coords in color_groups.items():
                        for y, x in coords:
                            color_by_position[(int(y), int(x))] = group

                cluster_by_position = {}
                cluster_details = results_store.get("cluster")
                if cluster_details is not None:
                    for (y, x), cluster_id in zip(cluster_details["positions"], cluster_details["labels"]):
                        cluster_by_position[(int(y), int(x))] = int(cluster_id)

                available_groups = ["All colonies"]
                if color_groups:
                    available_groups += [
                        name.title() for name, coords in color_groups.items() if coords
                    ]
                selected_group = st.selectbox("Show colonies", available_groups, key="explorer_group")
                if selected_group == "All colonies":
                    chosen_indices = list(range(total))
                else:
                    chosen_indices = [
                        idx for idx, pos in enumerate(positions)
                        if color_by_position.get(tuple(map(int, pos))) == selected_group.lower()
                    ]

                c_zoom, c_limit = st.columns(2)
                with c_zoom:
                    crop_radius = st.slider("Preview radius (px)", 12, 100, 35, 5, key="explorer_radius")
                with c_limit:
                    preview_count = st.selectbox("Colonies to display", [12, 24, 48, 96], key="explorer_limit")
                st.caption(f"Showing {min(len(chosen_indices), preview_count)} of {len(chosen_indices)} matching colonies.")
                source_rgb = np.asarray(analysis_plate.rgb_image)
                image_h, image_w = source_rgb.shape[:2]
                for offset in range(0, min(len(chosen_indices), preview_count), 6):
                    cols = st.columns(6)
                    for col, idx in zip(cols, chosen_indices[offset:offset + 6]):
                        y, x = map(int, positions[idx])
                        y0, y1 = max(0, y-crop_radius), min(image_h, y+crop_radius+1)
                        x0, x1 = max(0, x-crop_radius), min(image_w, x+crop_radius+1)
                        patch = source_rgb[y0:y1, x0:x1].copy()
                        if patch.size == 0:
                            continue
                        preview = Image.fromarray(np.asarray(patch, dtype=np.uint8))
                        draw = ImageDraw.Draw(preview)
                        cx, cy = x-x0, y-y0
                        draw.ellipse((cx-4, cy-4, cx+4, cy+4), outline="#e500ff", width=2)
                        with col:
                            st.image(preview, width="stretch")
                            st.caption(f"#{idx+1} · ({x}, {y})")

                st.markdown("**Export results**")
                st.caption("One row per detected candidate. Empty cells indicate analyses that have not been run.")
                sizes_export = results_store.get("sizes")
                rows = []
                for idx, (y, x) in enumerate(positions):
                    key_pos = (int(y), int(x))
                    color_name = color_by_position.get(key_pos, "")
                    cluster_id = cluster_by_position.get(key_pos)
                    diameter = ""
                    size_class = ""
                    if sizes_export is not None and idx < len(sizes_export):
                        value = float(sizes_export[idx])
                        if np.isfinite(value):
                            diameter = round(value, 3)
                            size_class = ("Small" if value < float(st.session_state.get("size_small", 10))
                                          else "Medium" if value < float(st.session_state.get("size_large", 15))
                                          else "Large")
                    species_name = ""
                    if color_name:
                        species_name = st.session_state.get(f"species_color_{color_name}", "").strip()
                    if not species_name and cluster_id is not None and cluster_id >= 0:
                        species_name = st.session_state.get(f"species_cluster_{cluster_id}", "").strip()
                    rows.append({
                        "candidate_id": idx + 1, "x_px": int(x), "y_px": int(y),
                        "diameter_px": diameter, "size_class": size_class,
                        "color_group": color_name.title() if color_name else "",
                        "cluster": ("Outlier" if cluster_id == -1 else f"Cluster {cluster_id}")
                                   if cluster_id is not None else "",
                        "color_species_name": st.session_state.get(f"species_color_{color_name}", "").strip() if color_name else "",
                        "cluster_species_name": st.session_state.get(f"species_cluster_{cluster_id}", "").strip()
                                                if cluster_id is not None and cluster_id >= 0 else "",
                        "species_name": species_name,
                    })
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
                st.download_button("Download colony data (CSV)",
                    data=output.getvalue().encode("utf-8-sig"),
                    file_name="colony_analysis.csv", mime="text/csv",
                    key="download_colony_csv", type="primary")
