import pandas as pd
import numpy as np

def calculate_hull_area(points):
    """
    Calculates the Convex Hull Area of a set of 2D points using the Monotone Chain algorithm
    and the Shoelace formula. No scipy required.
    """
    if len(points) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Sort points lexically
    points = sorted(points, key=lambda p: (p[0], p[1]))

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate (exclude last point of each list because it's the same as first of other)
    hull = lower[:-1] + upper[:-1]
    
    # Shoelace Formula for Area
    n = len(hull)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    
    return abs(area) / 2.0

def calculate_footprint(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, skiprows=7, delimiter='\t')
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    fungal_stats = []

    for genet_id, group in df.groupby('GenetID'):
        site = group['Site'].iloc[0]
        unique_trees = group.drop_duplicates(subset=['TreeID'])
        coords = unique_trees[['UTM_X', 'UTM_Y']].values
        degree = len(unique_trees)
        
        max_span = 0.0
        hull_area = 0.0
        
        # Max Span (Pairwise Distance)
        if degree > 1:
            # Numpy broadcasting for pairwise distance
            # (N,1,2) - (1,N,2) -> (N,N,2)
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=-1)
            max_span = np.sqrt(np.max(sq_dist))
            
        # Convex Hull Area
        if degree >= 3:
            # Convert to list of tuples for our custom function
            points_list = [(float(x), float(y)) for x, y in coords]
            hull_area = calculate_hull_area(points_list)
                
        fungal_stats.append({
            'GenetID': genet_id,
            'Site': site,
            'Degree': degree,
            'Max_Span_m': round(max_span, 2),
            'Hull_Area_m2': round(hull_area, 2)
        })

    summary_df = pd.DataFrame(fungal_stats)
    
    print("\n--- Physical Footprint Summary ---")
    print(summary_df.groupby('Site')[['Degree', 'Max_Span_m', 'Hull_Area_m2']].mean())
    
    # Saving to 'physical_summary.csv' to match user's analysis code
    output_file = 'physical_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\nDetailed summary saved to '{output_file}'")
    
    return summary_df

if __name__ == "__main__":
    # Update this path if necessary
    path = 'c:/Users/Hamsika/Documents/Business Analytics/.antigravity/ML/Problem 1/data/mycorrhizal_samples.txt'
    calculate_footprint(path)
