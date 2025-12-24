import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Rectangle
from matplotlib.gridspec import GridSpec

def plot_pie_overlay(labels, method_names, coordinates_pixels,
                     outpath="pie_overlay.png",
                     pie_radius=6.0,         # adjust for your pixel coordinate scale
                     pie_alpha=0.55,
                     cmap_name="tab20"):
    """
    labels: list of length M, each is an array-like of length N (method label per spot)
    method_names: list of length M
    coordinates_pixels: (N,2) array-like -> x,y pixel coordinates
    """
    # Convert inputs to arrays
    M = len(labels)                   # number of methods / slices per pie
    labels = [np.asarray(l) for l in labels]
    coords = np.asarray(coordinates_pixels)
    N = coords.shape[0]

    if any(len(l) != N for l in labels):
        raise ValueError("All label arrays must have the same length as number of coordinates.")

    if len(method_names) != M:
        raise ValueError("method_names length must equal number of methods (len(labels)).")

    # All unique cluster labels across methods (can be strings or numbers)
    all_labels = np.unique(np.concatenate(labels))
    K = len(all_labels)

    # Map each actual cluster label -> color
    cmap = plt.get_cmap(cmap_name, K)
    label_to_color = {lab: cmap(i) for i, lab in enumerate(all_labels)}

    # Prepare figure with main panel + right legend panel (wider legend panel)
    fig = plt.figure(figsize=(14, 10))  # Increased from 12 to 14 for wider legend
    gs = GridSpec(1, 2, width_ratios=[3, 1.5], wspace=0.25)  # Changed from [4, 1] to [3, 1.5]
    ax_main = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")
    ax_leg.set_aspect('equal')

    # Remove axis elements and make plot tight
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['bottom'].set_visible(False)
    ax_main.spines['left'].set_visible(False)

    ax_main.set_xlim(coordinates_pixels[:, 0].min() - pie_radius, coordinates_pixels[:, 0].max() + pie_radius)
    ax_main.set_ylim(coordinates_pixels[:, 1].min() - pie_radius, coordinates_pixels[:, 1].max() + pie_radius)

    # Helper: draw one pie at x,y for a given spot
    def draw_spot_pie(ax, x, y, labels_for_spot, radius=pie_radius, alpha=pie_alpha):
        starts = np.linspace(0, 360, len(labels_for_spot) + 1)
        for i, lab in enumerate(labels_for_spot):
            if lab in label_to_color:
                color = label_to_color[lab]
            elif str(lab) in label_to_color:
                color = label_to_color[str(lab)]
            else:
                raise ValueError(f"Label '{lab}' not found in label_to_color mapping.")
            # color = label_to_color[lab]
            wedge = Wedge(center=(x, y),
                          r=radius,
                          theta1=starts[i],
                          theta2=starts[i+1],
                          facecolor=(color[0], color[1], color[2], alpha),
                          edgecolor="black",
                          linewidth=0.0)
            ax.add_patch(wedge)
        # outer outline
        circ = Circle((x, y), radius, fill=False, edgecolor="black", linewidth=0.0)
        ax.add_patch(circ)

    # Draw pies for each spot
    for i in range(N):
        labels_here = [labels[m][i] for m in range(M)]  # Remove int() conversion
        x, y = coords[i]
        draw_spot_pie(ax_main, x, y, labels_here)

    ax_main.set_aspect('equal')
    ax_main.invert_yaxis()   # match image coordinate convention if desired
    ax_main.set_xlim(coords[:, 0].min() - pie_radius, coords[:, 0].max() + pie_radius)
    ax_main.set_ylim(coords[:, 1].min() - pie_radius, coords[:, 1].max() + pie_radius)
    ax_main.set_title("Overlay Domain Map: per-spot pies")

    # ---- Legend panel: colorless demo pie + arrows to method names ----
    ax_leg.text(0.05, 1.32, "Method slices", fontsize=14, fontweight='bold', va='top')  # Moved back down slightly

    demo_center = (0.25, 0.85)  # Moved left from 0.35 to 0.25
    demo_radius = 0.12  # Made slightly smaller from 0.15

    # demo pie: colorless slices (transparent facecolor)
    starts = np.linspace(0, 360, M + 1)
    mid_angles = (starts[:-1] + starts[1:]) / 2.0
    for i in range(M):
        w = Wedge(center=demo_center,
                  r=demo_radius,
                  theta1=starts[i],
                  theta2=starts[i+1],
                  facecolor=(1,1,1,0),   # fully transparent in legend pie
                  edgecolor='black',
                  linewidth=1.0)
        ax_leg.add_patch(w)
    # demo outline
    ax_leg.add_patch(Circle(demo_center, demo_radius, fill=False, edgecolor='black', linewidth=1.5))

    # arrows from demo slices to method names
    arrow_radius = 0.25  # Distance from center for text positioning
    for i, mname in enumerate(method_names):
        # Use the same angle as the slice middle for consistent radial layout
        angle_rad = np.deg2rad(mid_angles[i])
        
        # Start point: edge of demo pie slice
        x0 = demo_center[0] + demo_radius * 0.9 * np.cos(angle_rad)
        y0 = demo_center[1] + demo_radius * 0.9 * np.sin(angle_rad)
        
        # Target point: positioned in a circle around the demo pie
        tx = demo_center[0] + arrow_radius * np.cos(angle_rad)
        ty = demo_center[1] + arrow_radius * np.sin(angle_rad)
        
        ax_leg.annotate("", xy=(tx, ty), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color='black'))
        
        # Position text slightly further out from arrow target
        text_x = demo_center[0] + (arrow_radius + 0.05) * np.cos(angle_rad)
        text_y = demo_center[1] + (arrow_radius + 0.05) * np.sin(angle_rad)
        
        # Adjust text alignment based on position relative to center
        ha = 'left' if text_x > demo_center[0] else 'right'
        va = 'bottom' if text_y > demo_center[1] else 'top'
        
        ax_leg.text(text_x, text_y, mname, va=va, ha=ha, fontsize=10)

    # ---- Cluster color legend (moved much further down) ----
    ax_leg.text(0.05, 0.30, "Cluster colors", fontsize=14, fontweight='bold')  # Moved down to 0.35
    box_x = 0.06
    box_w = 0.12
    box_h = 0.03  # Slightly larger boxes
    y0 = 0.20  # Start lower at 0.28 for more separation
    dy = 0.05  # Reduced spacing between boxes to fit more items
    
    # Calculate the minimum y position needed for all labels
    min_y_needed = y0 - (len(all_labels) - 1) * dy - box_h
    
    for idx, lab in enumerate(all_labels):
        yy = y0 - idx * dy
        c = label_to_color[lab]
        ax_leg.add_patch(Rectangle((box_x, yy), box_w, box_h, facecolor=c, edgecolor='black', linewidth=0.8))
        ax_leg.text(box_x + box_w + 0.04, yy + box_h / 2.0, str(lab), va='center', fontsize=11)  # More text spacing

    # Adjust y-axis limits to ensure all legend items are visible
    ax_leg.set_ylim(min(min_y_needed - 0.05, -0.1), 1.5)
    
    plt.tight_layout()
    # check if it is eps
    
    if outpath.endswith('.eps'):
        fig.savefig(outpath, format='eps', dpi=300, bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved pie overlay to: {outpath}")


if __name__ == "__main__":
    # Example synthetic call with string labels:
    N = 60
    M = 10
    coords = np.random.rand(N,2) * 500
    # Mix of string and numeric labels
    label_choices = ['A', 'B', 'C', 'D']
    method_labels = [np.random.choice(label_choices, size=N) for _ in range(M)]
    names = [f"Method {i+1}" for i in range(M)]

    import os

    # Ensure the output directory exists
    os.makedirs("results/plots", exist_ok=True)

    plot_pie_overlay(method_labels, names, coords, outpath="results/plots/pie_overlay_example.png",
                    pie_radius=8.0, pie_alpha=0.9, cmap_name="tab20")
