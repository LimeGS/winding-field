"""Frozen original-CT planes and physical winding contours, independent of producers."""

import hashlib
import io
import json
import zipfile

import numpy as np

from physical_grid import point_stencils


def plane_coordinates(origin, size, axis, index):
    origin, size = np.asarray(origin), np.asarray(size)
    if (
        origin.shape != (3,)
        or size.shape != (3,)
        or origin.dtype.kind not in "iu"
        or size.dtype.kind not in "iu"
        or (origin < 0).any()
        or (size < 2).any()
        or type(axis) is not int
        or axis not in (0, 1, 2)
        or type(index) is not int
        or not 0 <= index < size[axis]
    ):
        raise ValueError("integer native XYZ plane and positive extent required")
    axes = [k for k in range(3) if k != axis]
    u, v = np.meshgrid(np.arange(size[axes[0]]), np.arange(size[axes[1]]))
    xyz = np.empty((*u.shape, 3), float)
    xyz[..., axis] = origin[axis] + index
    xyz[..., axes[0]] = origin[axes[0]] + u
    xyz[..., axes[1]] = origin[axes[1]] + v
    return xyz, axes


def sample_winding(field, meta, xyz, support=None):
    step = meta["step"]
    if not np.isfinite(step) or step <= 0 or list(field.shape) != meta["shape_zyx"]:
        raise ValueError("invalid field geometry or calibrated step")
    nodes, weights, valid = point_stencils(
        xyz.reshape(-1, 3), meta["origin_xyz"], meta["spacing_xyz"], field.shape, support
    )
    samples = field.ravel()[nodes]
    valid &= np.all((weights == 0) | np.isfinite(samples), axis=1)
    values = np.full(len(valid), np.nan)
    values[valid] = np.sum(np.where(weights[valid] > 0, samples[valid], 0) * weights[valid], axis=1) / step
    return values.reshape(xyz.shape[:-1])


def prepare(spec, case_directory, frame, step):
    if spec.get("schema") != "ct-winding-overlays-v1":
        raise ValueError("unsupported visualization schema")
    manifest_path = (case_directory / spec["archive_manifest"]["path"]).resolve()
    inputs = {}

    def checked_bytes(path, expected, name, limit):
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("visual archive exceeds byte budget: " + name)
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError("visual archive hash mismatch: " + name)
        inputs[name] = dict(path=str(path), sha256=actual, bytes=len(data))
        return data

    manifest = json.loads(
        checked_bytes(manifest_path, spec["archive_manifest"]["sha256"], "visual_manifest", 1024 * 1024)
    )

    def member(name):
        path = manifest_path.parent / name
        return checked_bytes(path, manifest[name], "visual_" + name, 64 * 1024 * 1024)

    receipt = json.loads(member("RECEIPT.json"))
    if type(receipt.get("synthetic", False)) is not bool:
        raise ValueError("synthetic CT flag must be boolean")
    if (
        receipt["frame"] != frame
        or receipt["step"] != step
        or receipt["voxel_um"] != 9.6
        or receipt["ct_level"] != 2
        or receipt["native_l0_scale"] != 4
    ):
        raise ValueError("CT visualization frame, scale or physical sampling differs from case")
    levels = np.asarray(receipt["contour_levels"], float)
    window = np.asarray(receipt["ct_window"], float)
    if (
        levels.ndim != 1
        or not 2 <= len(levels) <= 150
        or not np.isfinite(levels).all()
        or not np.array_equal(levels, np.floor(levels))
        or not (np.diff(levels) == 1).all()
        or window.shape != (2,)
        or not np.isfinite(window).all()
        or not 0 <= window[0] < window[1] <= 255
    ):
        raise ValueError("invalid frozen contour levels or CT window")
    origin, size = np.asarray(receipt["origin_xyz"]), np.asarray(receipt["size_xyz"])
    if size.shape != (3,) or size.dtype.kind not in "iu" or (size < 2).any() or (size > 1024).any():
        raise ValueError("visual planes require integer extents of 2..1024 pixels per axis")
    reference = spec.get("reference_key")
    if reference is not None:
        identity = receipt["field_sha256"].get(reference)
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("visual reference needs a valid source field SHA-256 identity")
    if [p["name"] for p in receipt["planes"]] != ["XY", "XZ", "YZ"]:
        raise ValueError("visual archive must declare ordered XY/XZ/YZ planes")
    planes = []
    for plane, axis in zip(receipt["planes"], (2, 1, 0)):
        if plane["fixed_axis_xyz"] != axis or type(plane["fixed_position_l2"]) is not int:
            raise ValueError("visual plane axis or position mismatch")
        xyz, remaining = plane_coordinates(origin, size, axis, plane["fixed_position_l2"] - int(origin[axis]))
        data = member(plane["name"] + "-samples.npz")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > 64 * 1024 * 1024:
                raise ValueError("visual sample archive exceeds its byte budget")
        with np.load(io.BytesIO(data), allow_pickle=False) as samples:
            ct = samples["ct"]
            if ct.shape != xyz.shape[:-1] or ct.dtype != np.uint8 or ct.size != plane["ct_pixels"]:
                raise ValueError("CT plane shape/dtype differs from physical geometry")
            item = dict(
                name=plane["name"], axis=axis, position=plane["fixed_position_l2"], remaining=remaining, xyz=xyz, ct=ct
            )
            if reference is not None:
                values, valid = samples[reference], samples[reference + "_valid"]
                if (
                    values.shape != ct.shape
                    or values.dtype.kind not in "fiu"
                    or valid.shape != ct.shape
                    or valid.dtype != bool
                    or not np.array_equal(valid, np.isfinite(values))
                ):
                    raise ValueError("invalid frozen reference samples/support")
                item["reference"] = values
            planes.append(item)
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("CT pictures require matplotlib; install the plot extra") from exc
    return dict(
        planes=planes,
        source=receipt,
        levels=levels,
        window=window,
        inputs=inputs,
        reference_key=reference,
        reference_label=str(spec.get("reference_label", "Frozen reference")),
        matplotlib_version=matplotlib.__version__,
    )


def draw_figures(planes, origin, voxel_mm, levels, window, columns, output, prefix=None, synthetic=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    norm, cmap = Normalize(levels[0], levels[-1]), plt.get_cmap("turbo")
    names = [((prefix or "overview") + ".png", planes)]
    names += [((prefix + "-" if prefix else "") + p["name"] + ".png", [p]) for p in planes]
    with matplotlib.rc_context({"text.usetex": False}):
        for filename, selection in names:
            fig, axes = plt.subplots(
                len(selection),
                len(columns),
                figsize=(5.3 * len(columns), 5.2 * len(selection)),
                squeeze=False,
                layout="constrained",
            )
            try:
                for row, plane in zip(axes, selection):
                    x = np.arange(plane["ct"].shape[1]) * voxel_mm
                    y = np.arange(plane["ct"].shape[0]) * voxel_mm
                    extent = (-voxel_mm / 2, x[-1] + voxel_mm / 2, -voxel_mm / 2, y[-1] + voxel_mm / 2)
                    for ax, (key, title) in zip(row, columns):
                        ax.imshow(
                            plane["ct"],
                            origin="lower",
                            cmap="gray",
                            vmin=window[0],
                            vmax=window[1],
                            extent=extent,
                            interpolation="nearest",
                        )
                        if key is not None:
                            data = plane[key]
                            values = data[np.isfinite(data)]
                            message = None
                            if not len(values):
                                message = "No supported prediction"
                            elif values.max() < levels[0] or values.min() > levels[-1]:
                                message = "Outside frozen contour range"
                            elif values.min() == values.max():
                                message = "No contour variation"
                            else:
                                contours = ax.contour(
                                    x,
                                    y,
                                    np.ma.masked_invalid(data),
                                    levels=levels,
                                    cmap=cmap,
                                    norm=norm,
                                    linewidths=0.65,
                                )
                                ax.clabel(contours, contours.levels[::2], inline=True, fontsize=5, fmt="%d")
                            if message:
                                ax.text(
                                    0.5,
                                    0.5,
                                    message,
                                    transform=ax.transAxes,
                                    ha="center",
                                    color="white",
                                    bbox=dict(facecolor="black", alpha=0.7),
                                )
                            title += f" — {len(values) / data.size:.1%} supported"
                        a, b = plane["remaining"]
                        ax.set_xlabel(f"{'XYZ'[a]} displacement (mm), origin {origin[a]} L2")
                        ax.set_ylabel(f"{'XYZ'[b]} displacement (mm), origin {origin[b]} L2")
                        ax.set_aspect("equal")
                        ax.set_title(
                            f"{title}\n{plane['name']} at {'XYZ'[plane['axis']]}={plane['position']} L2", fontsize=10
                        )
                fig.colorbar(
                    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=axes[:, 1:].ravel().tolist(),
                    label="W = phi / fixed step; integer contours, inherited gauge",
                    shrink=0.8,
                )
                fig.suptitle(
                    f"Predicted winding-coordinate contours on {'synthetic CT-like data' if synthetic else 'original CT'}\n"
                    "Matched CT/window and fixed contour levels; physical sheet boundaries are not certified.",
                    fontsize=12,
                )
                fig.savefig(output / filename, dpi=160)
            finally:
                plt.close(fig)
    return [name for name, _ in names]


def render(visual, field, meta, support, output):
    source = visual["source"]
    samples, reports = {}, []
    for plane in visual["planes"]:
        values = sample_winding(field, meta, plane["xyz"], support)
        plane["candidate"] = values
        key = plane["name"]
        samples.update(
            {key + "_ct": plane["ct"], key + "_candidate": values, key + "_candidate_valid": np.isfinite(values)}
        )
        if "reference" in plane:
            samples[key + "_reference"] = plane["reference"]
        known = values[np.isfinite(values)]
        reports.append(
            dict(
                name=key,
                fixed_axis_xyz=plane["axis"],
                fixed_position_l2=plane["position"],
                ct_pixels=values.size,
                candidate_covered=len(known),
                candidate_range=[float(known.min()), float(known.max())] if len(known) else None,
                candidate_outside_contour_range=int(
                    np.sum((known < visual["levels"][0]) | (known > visual["levels"][-1]))
                ),
            )
        )
    np.savez_compressed(output / "ct-winding-samples.npz", **samples)
    synthetic = source.get("synthetic", False)
    columns = [(None, f"{'Synthetic CT-like data' if synthetic else 'Original CT'} — 9.6 µm/pixel")]
    if visual["reference_key"] is not None:
        columns.append(("reference", visual["reference_label"]))
    spacing = np.asarray(meta["spacing_xyz"]) * source["voxel_um"]
    resolution = f"{spacing[0]:g}" if (spacing == spacing[0]).all() else " × ".join(f"{v:g}" for v in spacing)
    columns.append(("candidate", f"Candidate — {resolution} µm field"))
    figures = draw_figures(
        visual["planes"],
        source["origin_xyz"],
        source["voxel_um"] / 1000,
        visual["levels"],
        visual["window"],
        columns,
        output,
        prefix="ct-winding",
        synthetic=synthetic,
    )
    receipt = dict(
        frame=source["frame"],
        synthetic=synthetic,
        step=meta["step"],
        candidate_field_sha256=meta["field_sha256"],
        reference_field_sha256=source["field_sha256"][visual["reference_key"]]
        if visual["reference_key"] is not None
        else None,
        origin_xyz=source["origin_xyz"],
        size_xyz=source["size_xyz"],
        voxel_um=source["voxel_um"],
        contour_levels=visual["levels"].tolist(),
        ct_window=visual["window"].tolist(),
        planes=reports,
        input_ct=source["input_ct"],
        ct_url=source["ct_url"],
        inputs=visual["inputs"],
        matplotlib=visual["matplotlib_version"],
        figures=figures,
        alignment="Physical metadata only; no registration, gauge fitting or per-view rescaling.",
        interpretation="Predicted coordinate visual QC, not certified physical lamina identities.",
    )
    with (output / "VISUALS.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(receipt="VISUALS.json", figures=figures)
