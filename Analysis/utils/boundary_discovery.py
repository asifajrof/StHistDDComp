# binary mask of boundary pixels

from utils.models.cooridnate import Coordinates
from utils.data_loader import DataSet, DataRecord
import numpy as np
from scipy.spatial import cKDTree

class BoundaryMaskDetector:
    @staticmethod
    def detect_boundary_mask(coordinates: Coordinates, data_set: DataSet) -> list[str]:
        # sort barcodes of dataset and coordinates to ensure alignment
        coordinates.points.sort(key=lambda x: x.barcode)
        data_set.records.sort(key=lambda x: x.barcode)

        assert len(coordinates.points) == len(data_set.records), "Coordinates and DataSet must have the same number of points."
        # Extract coordinates and labels from the dataset
        barcodes = np.array([record.barcode for record in coordinates.points])
        coords = np.array([[record.x, record.y] for record in coordinates.points])
        labels = np.array([record.label for record in data_set.records])

        # Detect boundary points
        boundary_indices = BoundaryMaskDetector._get_boundary_points(coords, labels)
        boundary_barcodes = barcodes[boundary_indices]

        return boundary_barcodes

    @staticmethod
    def _get_boundary_points(coords, labels, k=6, threshod=0.3):
        """
        Identifies the indices of points that are on the boundary of a cluster.
        A point is on a boundary if at least one of its k-nearest neighbors 
        has a different label.
        """
        tree = cKDTree(coords)
        # Find k nearest neighbors (k+1 because the point itself is included)
        dists, indices = tree.query(coords, k=k+1)
        
        boundary_indices = []
        
        for i, neighbor_indices in enumerate(indices):
            # Get labels of neighbors (excluding the point itself at index 0)
            neighbor_labels = labels[neighbor_indices[1:]]
            current_label = labels[i]
            
            # If a significant portion of neighbors have a different label, it's a boundary
            different_label_ratio = np.sum(neighbor_labels != current_label) / len(neighbor_labels)
            if different_label_ratio > threshod:  # Threshold can be adjusted to make the rule softer or stricter
                boundary_indices.append(i)
                
        return np.array(boundary_indices)
