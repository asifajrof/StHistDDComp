from utils.models.cooridnate import Coordinates
import numpy as np
import libpysal
from esda.moran import Moran
from matplotlib import pyplot as plt

class MoransIOfBoundaryMask:
    
    @staticmethod
    def calculate_morans_i(boundary_barcodes: str, coordinates: Coordinates) -> float:

        # Extract all barcodes and create a boundary mask
        all_barcodes = [point.barcode for point in coordinates.points]
        boundary_mask = np.array([1 if barcode in boundary_barcodes else 0 for barcode in all_barcodes])
        
        # Extract coordinates as a NumPy array
        coordinates_array = np.array([[point.x, point.y] for point in coordinates.points])

        # Create spatial weights (K-Nearest Neighbors graph)
        w = libpysal.weights.KNN.from_array(coordinates_array, k=6)
        w.transform = 'R'  # Row-standardize the weights
        
        # Calculate Moran's I for the boundary mask
        moran = Moran(boundary_mask, w)

        if np.isnan(moran.I):
            print(f"Len : {len(boundary_barcodes)}, {len(all_barcodes)}, Barcodes: {boundary_barcodes}")
            return 0.0
        
        return moran.I