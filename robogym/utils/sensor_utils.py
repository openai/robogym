OCCLUSION_MARKERS = [
    "robot0:ffocclusion",
    "robot0:mfocclusion",
    "robot0:rfocclusion",
    "robot0:lfocclusion",
    "robot0:thocclusion",
]
OCCLUSION_DIST_CUTOFF = -0.0001  # neg; penetrated.


def occlusion_markers_exist(sim):
    for marker in OCCLUSION_MARKERS:
        if marker not in sim.model.geom_names:
            return False
    return True


def check_occlusion(sim, dist_cutoff=OCCLUSION_DIST_CUTOFF):
    """
    Check whether there is any collision or contact with the finger occlusion detection
    geoms (class = "D_Occlusion").

    Given a finger occlusion geom, if there is a contact and the contact distance is smaller
    than `dist_cutoff`, we consider it as "being occluded".

    Returns: a list of 5 binary, indicating whether a finger (ff, mf, rf, lf, th) is occluded.
    """
    target_geom_ids = [sim.model.geom_name2id(m) for m in OCCLUSION_MARKERS]
    geom_ids_with_contact = set()
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        if contact.dist < dist_cutoff:
            geom1 = contact.geom1
            geom2 = contact.geom2
            geom_ids_with_contact.add(geom1)
            geom_ids_with_contact.add(geom2)

    return [int(g_id in geom_ids_with_contact) for g_id in target_geom_ids]


def recolor_occlusion_geoms(sim, robot_occlusion_data):
    """
    Color the occlusion geoms differently according to whether the simulator and the
    phasespace tracker matches.
    """
    colormap = [
        [0, 0, 0, 0.1],  # transparent grey for both off
        [1, 0, 0, 0.7],  # red for robot not but sim occluded
        [0, 0, 1, 0.7],  # blue for robot occluded but sim not
        [1, 1, 0, 1.0],  # solid yellow for both occluded
    ]
    sim_occlusion_data = check_occlusion(sim)
    geom_ids = [sim.model.geom_name2id(m) for m in OCCLUSION_MARKERS]

    for g_id, robot_occluded, sim_occluded in zip(
        geom_ids, robot_occlusion_data, sim_occlusion_data
    ):
        category = 2 * int(robot_occluded) + int(sim_occluded)
        sim.model.geom_rgba[g_id] = colormap[category]
