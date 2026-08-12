import polyscope as ps
import numpy as np
import h5py
import polyscope.imgui as psim


# =========================
# LOADERS
# =========================

def load_dfaust_sequence(h5_path, sequence_key):
    with h5py.File(h5_path, "r") as f:
        verts = f[sequence_key][:]
        verts = np.transpose(verts, (2, 0, 1))  # (T, V, 3)
        faces = f["faces"][:]
    return verts, faces


def load_ply_with_time(path, max_points=None):
    points, colors, times = [], [], []

    with open(path, "r") as f:
        header = True
        for line in f:
            if header:
                if line.strip() == "end_header":
                    header = False
                continue

            x, y, z, r, g, b, t = line.split()
            points.append([float(x), float(y), float(z)])
            colors.append([int(r), int(g), int(b)])
            times.append(float(t))

            if max_points is not None and len(points) >= max_points:
                break

    return (
        np.array(points),
        np.array(colors) / 255.0,
        np.clip(np.array(times), 0.0, 1.0),
    )


def compute_time_bins(times, num_bins):
    return np.clip((times * (num_bins - 1)).astype(int), 0, num_bins - 1)


# =========================
# MAIN
# =========================

def main():
    ps.init()

    # ---- CONFIG ----
    point_cloud_num_bins = 1000
    point_cloud_max_points = 5000000  # Set to an int to cap loaded/used points per cloud.

    h5_A = "/home/ben/LRZSyncShare/Thesis/Data/data/DFAUST/registrations_f.hdf5"
    h5_B = "/home/ben/LRZSyncShare/Thesis/Data/data/DFAUST/registrations_m.hdf5"

    seq_A = "50025_shake_arms"
    seq_B = "50009_punching"

    cloud_A = "samples/geofusion_dfaust_50025_shake_arms_50009_punching/shape-A_e0020_n300000_colored.ply"
    cloud_B = "samples/geofusion_dfaust_50025_shake_arms_50009_punching/shape-B_e0020_n300000_colored.ply"

    offset_A = np.array([-1.5, 0, 0])
    offset_B = np.array([ 1.5, 0, 0])
    # -----------------

    # --- load meshes ---
    verts_A, faces = load_dfaust_sequence(h5_A, seq_A)
    verts_B, _     = load_dfaust_sequence(h5_B, seq_B)

    T = len(verts_A)

    verts_A += offset_A
    verts_B += offset_B

    mesh_A = ps.register_surface_mesh("mesh_A", verts_A[0], faces)
    mesh_B = ps.register_surface_mesh("mesh_B", verts_B[0], faces)

    for m in [mesh_A, mesh_B]:
        m.set_edge_width(1.0)
        m.set_transparency(0.4)

    # --- load point clouds ---
    pts_A, col_A, t_A = load_ply_with_time(cloud_A, max_points=point_cloud_max_points)
    pts_B, col_B, t_B = load_ply_with_time(cloud_B, max_points=point_cloud_max_points)

    bins_A = compute_time_bins(t_A, point_cloud_num_bins)
    bins_B = compute_time_bins(t_B, point_cloud_num_bins)

    pts_A += offset_A
    pts_B += offset_B

    # --- precompute bins ---
    pcs_A, pcs_B = {}, {}

    for i in range(point_cloud_num_bins):
        mask_A = bins_A == i
        mask_B = bins_B == i

        if np.any(mask_A):
            pc = ps.register_point_cloud(f"A_{i}", pts_A[mask_A])
            pc.add_color_quantity("color", col_A[mask_A], enabled=True)
            pc.set_enabled(False)
            pcs_A[i] = pc

        if np.any(mask_B):
            pc = ps.register_point_cloud(f"B_{i}", pts_B[mask_B])
            pc.add_color_quantity("color", col_B[mask_B], enabled=True)
            pc.set_enabled(False)
            pcs_B[i] = pc

    # =========================
    # CALLBACK
    # =========================

    current_step = 0
    playback_pos = 0.0
    autoplay = False
    playback_speed = 1.0
    show_points = True
    show_all = False
    show_A = True
    show_B = True
    point_radius = 1e-3

    def callback():
        nonlocal current_step, playback_pos, autoplay, playback_speed, show_points, show_all, show_A, show_B, point_radius

        psim.Text("Controls")
        psim.Separator()

        # --- toggles ---
        _, autoplay = psim.Checkbox("Autoplay", autoplay)
        _, show_points = psim.Checkbox("Show Points", show_points)
        _, show_all = psim.Checkbox("Show All Bins", show_all)

        psim.Separator()
        psim.Text("Shapes")

        _, show_A = psim.Checkbox("Show Shape A", show_A)
        _, show_B = psim.Checkbox("Show Shape B", show_B)

        # --- playback speed ---
        _, playback_speed = psim.SliderFloat("Playback Speed (x original)", playback_speed, 0.1, 4.0)

        # --- timeline step (point-cloud bins) ---
        changed, current_step = psim.SliderInt("Step", current_step, 0, point_cloud_num_bins - 1)
        if changed:
            playback_pos = float(current_step)

        # --- point size ---
        log_r = np.log10(point_radius)
        _, log_r = psim.SliderFloat("log10(Point Size)", log_r, -5, -2)
        point_radius = 10 ** log_r

        if autoplay and not changed:
            bin_step = (point_cloud_num_bins - 1) / max(T - 1, 1)
            playback_pos = (playback_pos + bin_step * playback_speed) % point_cloud_num_bins
            current_step = int(round(playback_pos)) % point_cloud_num_bins

        # Map the point-cloud step timeline to mesh frame timeline.
        mesh_frame = int(round(current_step * (T - 1) / max(point_cloud_num_bins - 1, 1)))

        # --- update meshes ---
        mesh_A.set_enabled(show_A)
        mesh_B.set_enabled(show_B)

        mesh_A.update_vertex_positions(verts_A[mesh_frame])
        mesh_B.update_vertex_positions(verts_B[mesh_frame])

        # --- update point clouds ---
        for i, pc in pcs_A.items():
            visible = show_points and show_A and (show_all or i == current_step)
            pc.set_enabled(visible)
            if visible:
                pc.set_radius(point_radius)

        for i, pc in pcs_B.items():
            visible = show_points and show_B and (show_all or i == current_step)
            pc.set_enabled(visible)
            if visible:
                pc.set_radius(point_radius)

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()


