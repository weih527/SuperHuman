import numpy as np
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
import elf.segmentation.watershed as ws


# if the compute_affinity features function doesn't work because of windows issues
def affinity_feature_fallback(rag, ws, affs, offsets):
    assert offsets == [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    xy_boundaries = np.maximum(affs[1], affs[2])
    z_boundaries = affs[0]

    # compute in plane features
    features = feats.compute_boundary_mean_and_length(rag, xy_boundaries)
    # compute between plane features
    z_features = feats.compute_boundary_mean_and_length(rag, z_boundaries)

    # over-ride between plane features in features
    z_edges = feats.compute_z_edge_mask(rag, ws)
    features[between_plane_edges] = z_features[z_edges]
    
    return features


def mc_baseline(affs, fragments=None):
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    if fragments is None:
        fragments = np.zeros_like(boundary_input, dtype='uint64')
        offset = 0
        for z in range(fragments.shape[0]):
            wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=.25, sigma_seeds=2.)
            wsz += offset
            offset += max_id
            fragments[z] = wsz
    rag = feats.compute_rag(fragments)
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    # This is only necessary if compute_affinity_features does not work due to issues on windows or mac.
    # costs = affinity_feature_fallback(rag, fragments, affs, offsets)[:, 0]
    costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
    edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    return segmentation


def multicut_multi(affs, offsets=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]], fragments=None):
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    if fragments is None:
        fragments = np.zeros_like(boundary_input, dtype='uint64')
        offset = 0
        for z in range(affs.shape[1]):
            wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=.25, sigma_seeds=2.)
            wsz += offset
            offset += max_id
            fragments[z] = wsz
    rag = feats.compute_rag(fragments)

    # This is only necessary if compute_affinity_features does not work due to issues on windows or mac.
    # costs = affinity_feature_fallback(rag, fragments, affs, offsets)[:, 0]
    costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
    
    edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    return segmentation
