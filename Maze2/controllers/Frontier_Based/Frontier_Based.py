from controller import Robot, Supervisor, Motor, Keyboard
import matplotlib.pyplot as plt
import numpy as np
import math
import heapq # For A* and Dijkstra priority queue
import collections # For deque in BFS-like search
import cv2 # For OpenCV operations (color detection)

# --- CONSTANTS ---
TIME_STEP = 32
MAX_VELOCITY = 26.0
BASE_SPEED = 15
backup_state = False
backup_timer = 0
BACKUP_DURATION = 3

# Mapping parameters
ROBOT_RADIUS_METERS = 0.149
WORLD_SIZE = 10.00
MAP_SIZE = 500  # REDUCED MAP SIZE FOR PERFORMANCE
MAP_RESOLUTION_METERS_PER_CELL = WORLD_SIZE / MAP_SIZE

# Robot inflation
ROBOT_INFLATION_EFFECTIVE_RADIUS = ROBOT_RADIUS_METERS
ROBOT_INFLATION_CELLS = int(ROBOT_INFLATION_EFFECTIVE_RADIUS / MAP_RESOLUTION_METERS_PER_CELL)

# Confidence parameters
CONFIDENCE_INCREMENT_HIT = 0.08
CONFIDENCE_DECREMENT_FREE = 0.05
P_UNKNOWN = 0.5

# Thresholds
OCCUPIED_THRESHOLD = 0.8
INFLATED_OCCUPIED_VALUE = 0.7  # Value for inflated obstacles, aligned with PDF

# Map cell states
FRONTIER_VALUE = 0.25  # constant for frontier cells

# Processing limits
MAX_PROCESSING_RANGE = 5
MIN_DEPTH = 0.1  # Minimum depth for valid camera readings
PLOT_UPDATE_INTERVAL = 1
MAP_CLEAN_INTERVAL = 1

# Navigation parameters
TURN_SPEED_RATIO = 0.3
AUTONOMOUS_INITIAL_MOVEMENT_DURATION = 5  # Keep for initial forward movement
TARGET_REACHED_THRESHOLD = 0.5  # meters (for reaching the final frontier/color target)
WAYPOINT_REACHED_THRESHOLD_CELLS = 5  # cells (for reaching intermediate A* waypoints)

# Path Simplification Parameter
PATH_SIMPLIFICATION_LOOKAHEAD = 15  # Number of waypoints to look ahead for re-planning

# Map update turn detection
MIN_SPEED_FOR_MAP_UPDATE_TURN_DETECTION = 0.3  # meters/s
FRONTIER_EPSILON = 0.005 

robot = Supervisor()
target_robot = robot.getFromDef('ROSBOT_MAIN')
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

# Map initialization
occupancy_map = np.ones((MAP_SIZE, MAP_SIZE)) * P_UNKNOWN
cost_map = np.ones((MAP_SIZE, MAP_SIZE)) * float('inf')  # New cost map

# Plotting variables
img_display = None
cross_artist, arrow_artist = None, None
target_artist = None
path_artist = None

# --- Global State Variables for Autonomous Mode ---
autonomous_mode = True
initial_position_vec, _ = robot.getFromDef('ROSBOT_MAIN').getField('translation').getSFVec3f(), robot.getFromDef('ROSBOT_MAIN').getField('rotation').getSFRotation()
last_robot_position = list(initial_position_vec)
initial_movement_counter = 0
current_target = None
current_path = []  # Stores the A* generated path (list of (row, col) waypoints)

recently_reached_targets = collections.deque(maxlen=5)
blue_target_detected = None
yellow_target_detected = None
blue_reached = False
mission_complete = False

# Constants for color detection and cleaning
MIN_CONTOUR_AREA = 20
MIN_NEIGHBORS = 2

# Mission states
EXPLORE = 0
INITIAL_SCAN = 1
GO_TO_BLUE = 2
GO_TO_YELLOW = 3    
MISSION_COMPLETE = 4
mission_state = INITIAL_SCAN

# Pillar attempt counters
blue_attempt_count = 0
yellow_attempt_count = 0
MAX_PILLAR_ATTEMPTS = 2  # Try max 2 times before exploring

# Add fallback states
FALLBACK_TO_BLUE = 5
FALLBACK_TO_YELLOW = 6

initial_scan_start_heading = 0.0
initial_scan_end_heading = 0.0
initial_scan_progress = 0.0
# Calculate angular velocity (rad/s)
angular_velocity = (BASE_SPEED * TURN_SPEED_RATIO) * 2 / ROBOT_RADIUS_METERS
# Calculate time for full rotation (seconds)
INITIAL_SCAN_DURATION_STEPS = 200
recently_failed_targets = []  # Stores (row, col) of failed targets
FAILED_TARGET_COOLDOWN = 50   # Skip for 50 timesteps after failure

MIN_WALL_DISTANCE = 1.0  # meters
WALL_SCALING_FACTOR = 0.7  # wall size multiplier

def get_translation_rotation():
    if target_robot:
        return target_robot.getField('translation').getSFVec3f(), target_robot.getField('rotation').getSFRotation()
    return [0, 0, 0], [0, 0, 0, 1]

def world_to_map_coordinates(x_world, y_world):
    x_shifted = x_world + (WORLD_SIZE / 2)
    y_shifted = y_world + (WORLD_SIZE / 2)

    col = int((x_shifted / WORLD_SIZE) * MAP_SIZE)
    row = int((y_shifted / WORLD_SIZE) * MAP_SIZE)
    
    row = MAP_SIZE - 1 - row  # Invert row for matplotlib's image origin

    return np.clip(row, 0, MAP_SIZE - 1), np.clip(col, 0, MAP_SIZE - 1)

def map_to_world_coordinates(row, col):
    y_map_inverted = MAP_SIZE - 1 - row
    
    x_shifted = col * MAP_RESOLUTION_METERS_PER_CELL + MAP_RESOLUTION_METERS_PER_CELL / 2
    y_shifted = y_map_inverted * MAP_RESOLUTION_METERS_PER_CELL + MAP_RESOLUTION_METERS_PER_CELL / 2
    
    x_world = x_shifted - (WORLD_SIZE / 2)
    y_world = y_shifted - (WORLD_SIZE / 2)
    return x_world, y_world

def axis_angle(rotation):
    axis = rotation[:3]
    angle = rotation[3]
    return -angle if axis[2] < 0 else angle
def mark_permanent_free_space(occupancy_map, row, col, radius=3):
    """
    Marks a circular area around (row,col) as permanently free space (0.0).
    These cells will be added to fixed_map to prevent future modifications.
    
    Args:
        occupancy_map: The occupancy grid to modify
        row, col: Center point in map coordinates
        radius: Radius in cells to mark as free (default 3 gives 7x7 area)
    """
    # Ensure we don't go out of bounds
    min_row = max(0, row - radius)
    max_row = min(MAP_SIZE-1, row + radius)
    min_col = max(0, col - radius)
    max_col = min(MAP_SIZE-1, col + radius)
    
    for r in range(min_row, max_row+1):
        for c in range(min_col, max_col+1):
            # Use Euclidean distance to create a circular area
            distance_sq = (r - row)**2 + (c - col)**2
            if distance_sq <= radius**2:
                occupancy_map[r, c] = 0.0  # Mark as free
                fixed_map[r, c] = True    # Permanently fix this cell
def plot_robot(ax, row, col, heading, length=10):
    cross_artist = ax.plot(col, row, 'g+', markersize=10, markeredgewidth=2)[0]
    dx = length * math.cos(-heading)  # Adjust for map's y-axis
    dy = length * math.sin(-heading)
    arrow_artist = ax.quiver(col, row, dx, dy, angles='xy', scale_units='xy',
                             scale=1, color='g', width=0.003)
    return cross_artist, arrow_artist

def get_lidar_data_in_world(lidar, robot_position_world, robot_heading_radians):
    range_image = np.array(lidar.getRangeImage())
    num_beams = lidar.getHorizontalResolution()
    fov = lidar.getFov()

    beams_data = []
    cos_heading = math.cos(robot_heading_radians)
    sin_heading = math.sin(robot_heading_radians)

    for i in range(0, num_beams, 2):  # Process every other beam for performance/density
        angle = fov / 2.0 - i * fov / (num_beams - 1.0)  # Angle relative to robot's forward
        distance = range_image[i]

        if distance > MAX_PROCESSING_RANGE or math.isinf(distance) or math.isnan(distance) or distance < 0.01:
            continue

        x_local = distance * math.cos(angle)
        y_local = distance * math.sin(angle)

        x_world = robot_position_world[0] + (x_local * cos_heading - y_local * sin_heading)
        y_world = robot_position_world[1] + (x_local * sin_heading + y_local * cos_heading)

        beams_data.append({
            'x_world': x_world,
            'y_world': y_world,
            'distance': distance,
            'angle': angle
        })

    return beams_data

def bresenham_line(start_col, start_row, end_col, end_row):
    """Generates points on a line using Bresenham's algorithm."""
    x0, y0 = start_col, start_row
    x1, y1 = end_col, end_row
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points.append((x, y))  # Add the end point
    return points

# After map initialization
fixed_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)  # Track fixed cells
# Set starting position as fixed free space
initial_position_vec, _ = get_translation_rotation()
robot_map_r, robot_map_c = world_to_map_coordinates(initial_position_vec[0], initial_position_vec[1])

occupancy_map[robot_map_r, robot_map_c] = 0.0  # Mark as free
fixed_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool) # Permanently fix
# At initialization after map setup
initial_position_vec, _ = get_translation_rotation()
robot_map_r, robot_map_c = world_to_map_coordinates(initial_position_vec[0], initial_position_vec[1])
mark_permanent_free_space(occupancy_map, robot_map_r, robot_map_c)
def update_occupancy_grid(current_map, lidar_beams_data, robot_position_world, robot_heading_radians):
    robot_row, robot_col = world_to_map_coordinates(robot_position_world[0], robot_position_world[1])
    hit_cells = []

    for beam in lidar_beams_data:
        if beam['distance'] > MAX_PROCESSING_RANGE or math.isinf(beam['distance']) or math.isnan(beam['distance']) or beam['distance'] < 0.01:
            continue

        obs_row, obs_col = world_to_map_coordinates(beam['x_world'], beam['y_world'])
        
        # Free space (ray tracing)
        ray_cells = bresenham_line(robot_col, robot_row, obs_col, obs_row)
        for x, y in ray_cells[:-1]:  # Exclude the hit point itself
            col, row = x, y
            if fixed_map[row, col]:
                continue
            if 0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE:
                # Only decrement if not already occupied or inflated
                if current_map[row, col] != 1.0 and current_map[row, col] != INFLATED_OCCUPIED_VALUE:
                    current_map[row, col] = max(0.0, current_map[row, col] - CONFIDENCE_DECREMENT_FREE)
                    
                    # After marking as free, check neighbors for unknown cells to tag as frontiers
                    for dr_n in [-1, 0, 1]:
                        for dc_n in [-1, 0, 1]:
                            if dr_n == 0 and dc_n == 0: continue
                            n_row, n_col = row + dr_n, col + dc_n
                            if 0 <= n_row < MAP_SIZE and 0 <= n_col < MAP_SIZE:
                                if current_map[n_row, n_col] == P_UNKNOWN:
                                    current_map[n_row, n_col] = FRONTIER_VALUE  # Tag as frontier

        # Occupied space (hit point)
        if 0 <= obs_row < MAP_SIZE and 0 <= obs_col < MAP_SIZE:
            hit_cells.append((obs_row, obs_col))

    for obs_row, obs_col in hit_cells:
        if current_map[obs_row, obs_col] != INFLATED_OCCUPIED_VALUE:
            current_map[obs_row, obs_col] = min(1.0, current_map[obs_row, obs_col] + CONFIDENCE_INCREMENT_HIT)
            if current_map[obs_row, obs_col] >= OCCUPIED_THRESHOLD:
                current_map[obs_row, obs_col] = 1.0

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = obs_row + dr, obs_col + dc
            if 0 <= r < MAP_SIZE and 0 <= c < MAP_SIZE:
                if current_map[r, c] < OCCUPIED_THRESHOLD and current_map[r, c] != INFLATED_OCCUPIED_VALUE:
                    current_map[r, c] = min(1.0, current_map[r, c] + CONFIDENCE_INCREMENT_HIT * 0.5)

    temp_map = np.copy(current_map)
    for obs_row, obs_col in hit_cells:
        if current_map[obs_row, obs_col] >= OCCUPIED_THRESHOLD:  # If truly occupied
            for dr in range(-ROBOT_INFLATION_CELLS, ROBOT_INFLATION_CELLS):
                for dc in range(-ROBOT_INFLATION_CELLS, ROBOT_INFLATION_CELLS):
                    if dr*dr + dc*dc <= ROBOT_INFLATION_CELLS*ROBOT_INFLATION_CELLS:
                        ir, ic = obs_row + dr, obs_col + dc
                        if 0 <= ir < MAP_SIZE and 0 <= ic < MAP_SIZE:
                            # ONLY inflate if not frontier and not unknown
                            if (current_map[ir, ic] != FRONTIER_VALUE and 
                                current_map[ir, ic] != P_UNKNOWN and
                                current_map[ir, ic] < OCCUPIED_THRESHOLD):
                                temp_map[ir, ic] = INFLATED_OCCUPIED_VALUE

    current_map[:] = temp_map[:]
    # Clip map values between 0.0 and 1.0 (inclusive)
    np.clip(current_map, 0.0, 1.0, out=current_map)

def clean_isolated_pixels(current_map, occupied_thresh, map_size):
    # Create a padded map to handle boundaries easily
    padded_map = np.pad(current_map, 1, mode='constant', constant_values=0.0)
    
    # Identify cells that are considered obstacles (occupied or inflated)
    obstacle_mask = (current_map >= occupied_thresh) | (current_map == INFLATED_OCCUPIED_VALUE)
    
    # Initialize a count for obstacle neighbors for each cell
    neighbor_count = np.zeros_like(current_map, dtype=int)

    # Iterate through 8-connected neighbors
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:  # Skip the cell itself
                continue
            # Accumulate count of obstacle neighbors
            neighbor_count += (padded_map[1 + dr : map_size + 1 + dr, 1 + dc : map_size + 1 + dc] >= occupied_thresh) | \
                              (padded_map[1 + dr : map_size + 1 + dr, 1 + dc : map_size + 1 + dc] == INFLATED_OCCUPIED_VALUE)

    # Define a minimum number of neighbors to consider a pixel a valid obstacle
    MIN_NEIGHBORS_CLEAN = 5  # Changed name to avoid conflict with color detection MIN_NEIGHBORS
    
    # Identify pixels to clean:
    to_clean = obstacle_mask & \
               (neighbor_count < MIN_NEIGHBORS_CLEAN) & \
               (current_map < OCCUPIED_THRESHOLD) & \
               (current_map != 0.0) & \
               (current_map != INFLATED_OCCUPIED_VALUE)
    
    # Apply fixed_map protection if available
    if fixed_map is not None:
        to_clean = to_clean & (~fixed_map)
    
    # Create a copy to modify and return
    cleaned_map = np.copy(current_map)
    # Reset identified isolated pixels to P_UNKNOWN
    cleaned_map[to_clean] = P_UNKNOWN
    return cleaned_map

def remove_isolated_frontiers(current_map, free_value=0.0, frontier_value=FRONTIER_VALUE, unknown_value=P_UNKNOWN):
    temp_map = np.copy(current_map)
    
    # Find all current frontier cells
    frontier_cells_rows, frontier_cells_cols = np.where((current_map >= (frontier_value - FRONTIER_EPSILON)) &
                                                        (current_map <= (frontier_value + FRONTIER_EPSILON)))

    for r, c in zip(frontier_cells_rows, frontier_cells_cols):
        is_isolated = True
        # Check 8-connected neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                n_row, n_col = r + dr, c + dc
                if 0 <= n_row < MAP_SIZE and 0 <= n_col < MAP_SIZE:
                    if current_map[n_row, n_col] == free_value:
                        is_isolated = False
                        break
            if not is_isolated:
                break
        
        if is_isolated:
            temp_map[r, c] = unknown_value  # Untag as frontier, revert to unknown

    return temp_map

# --- Optimized Costmap Generation Function (Dijkstra from Frontiers) ---
def compute_costs(current_map, cost_map):
    cost_map[:] = float('inf')  # Reset all costs to infinity
    open_set = []  # Priority queue (cost, (row, col))

    # Initialize priority queue with all frontier cells
    frontier_rows, frontier_cols = np.where(current_map == FRONTIER_VALUE)
    for r, c in zip(frontier_rows, frontier_cols):
        cost_map[r, c] = 0
        heapq.heappush(open_set, (0, (r, c)))
    
    # Neighbors offsets (8-connected) with their respective costs
    neighbors_offsets = [
        (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),        # Cardinal
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),      # Diagonal
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))
    ]

    while open_set:
        current_cost, (r, c) = heapq.heappop(open_set)

        # If we found a shorter path to this cell already, skip
        if current_cost > cost_map[r, c]:
            continue

        for dr, dc, move_cost in neighbors_offsets:
            n_r, n_c = r + dr, c + dc
            if not (0 <= n_r < MAP_SIZE and 0 <= n_c < MAP_SIZE):
                continue  # Out of bounds

            neighbor_map_value = current_map[n_r, n_c]

            # Only propagate costs to traversable cells (not occupied or inflated)
            if (neighbor_map_value >= OCCUPIED_THRESHOLD or 
                neighbor_map_value == INFLATED_OCCUPIED_VALUE):
                continue  # Is an obstacle or inflated zone

            new_cost = current_cost + move_cost

            if new_cost < cost_map[n_r, n_c]:
                cost_map[n_r, n_c] = new_cost
                heapq.heappush(open_set, (new_cost, (n_r, n_c)))

# --- Frontier Detection and Target Selection ---
def find_frontiers(current_map, frontier_value=FRONTIER_VALUE):
    frontiers = []
    rows, cols = np.where((current_map >= (FRONTIER_VALUE - FRONTIER_EPSILON)) &
                        (current_map <= (FRONTIER_VALUE + FRONTIER_EPSILON)))
    for r, c in zip(rows, cols):
        frontiers.append((r, c))
    return frontiers

def cluster_frontiers(frontiers):
    clusters = []
    visited = set()

    for r, c in frontiers:
        if (r, c) in visited:
            continue

        current_cluster = []
        q = collections.deque([(r, c)])
        visited.add((r, c))

        while q:
            curr_r, curr_c = q.popleft()
            current_cluster.append((curr_r, curr_c))

            # Check 8-connected neighbors for clustering
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    n_r, n_c = curr_r + dr, curr_c + dc
                    if (n_r, n_c) in frontiers and (n_r, n_c) not in visited:
                        q.append((n_r, n_c))
                        visited.add((n_r, n_c))
        clusters.append(current_cluster)
    return clusters

def set_target(current_map, cost_map, robot_position_world):
    frontiers = find_frontiers(current_map)
    
    if not frontiers:
        return None
    
    clusters = cluster_frontiers(frontiers)
    best_target = None
    min_combined_cost = float('inf')
    robot_r, robot_c = world_to_map_coordinates(robot_position_world[0], robot_position_world[1])
    blue_target, yellow_target = get_world_centers()
    
    # Target attraction points (priority: blue > yellow)
    attraction_points = []
    if blue_target is not None and not blue_reached:
        attraction_points.append(world_to_map_coordinates(blue_target[0], blue_target[1]))
    elif yellow_target is not None and blue_reached:  # Only consider yellow if blue reached
        attraction_points.append(world_to_map_coordinates(yellow_target[0], yellow_target[1]))

    for cluster in clusters:
        cluster_min = float('inf')
        cluster_target = None
        
        for r, c in cluster:
            if (r, c) in recently_reached_targets or \
               any((r, c) == failed_rc for (failed_rc, _) in recently_failed_targets):
                continue
                
            # Base exploration cost (from frontiers)
            base_cost = cost_map[r, c]
            
            # Target attraction cost
            attraction_cost = 0
            if attraction_points:
                min_dist = min(math.sqrt((r-att_r)**2 + (c-att_c)**2) 
                          for att_r, att_c in attraction_points)
                attraction_cost = 0.7 * (min_dist / MAP_SIZE)  # Weighted attraction

            # Unknown area penalty (NEW)
            unknown_penalty = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:  # Check 4-neighbors
                nr, nc = r+dr, c+dc
                if 0 <= nr < MAP_SIZE and 0 <= nc < MAP_SIZE:
                    if current_map[nr, nc] == P_UNKNOWN:
                        unknown_penalty += 0.3  # Penalty per unknown neighbor

            combined_cost = base_cost + attraction_cost + unknown_penalty
            
            if combined_cost < cluster_min:
                cluster_min = combined_cost
                cluster_target = (r, c)

        if cluster_target and cluster_min < min_combined_cost:
            min_combined_cost = cluster_min
            best_target = cluster_target

    if best_target:
        x, y = map_to_world_coordinates(*best_target)
        return {'x': x, 'y': y, 'map_rc': best_target}
    return None

def go_2_target(robot_position_world, robot_heading_radians, target_point_world_x, target_point_world_y):
    # Calculate vector from robot to target
    dx = target_point_world_x - robot_position_world[0]
    dy = target_point_world_y - robot_position_world[1]

    # Calculate distance to target
    distance_to_target = math.sqrt(dx**2 + dy**2)

    # Calculate angle to target relative to robot's forward direction
    angle_to_target = math.atan2(dy, dx)
    
    # Normalize angle to be between -PI and PI
    relative_angle = angle_to_target - robot_heading_radians
    relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))

    # Control logic
    angular_threshold = 0.1  # radians
    
    left_speed = 0.0
    right_speed = 0.0

    if abs(relative_angle) > angular_threshold:
        # Need to rotate
        if relative_angle > 0:  # Turn left
            left_speed = -BASE_SPEED * TURN_SPEED_RATIO
            right_speed = BASE_SPEED * TURN_SPEED_RATIO
        else:  # Turn right
            left_speed = BASE_SPEED * TURN_SPEED_RATIO
            right_speed = -BASE_SPEED * TURN_SPEED_RATIO
    else:
        # Aligned, move forward, speed decreases as we approach target to avoid overshooting
        speed_factor = max(0.2, min(1.0, distance_to_target / 0.5))  # Adjust 0.5 for desired deceleration distance
        left_speed = BASE_SPEED * speed_factor
        right_speed = BASE_SPEED * speed_factor
        
    return left_speed, right_speed, distance_to_target  # Return distance for checking waypoint reached

# --- A* Pathfinding Implementation ---
def heuristic(a, b):
    # Euclidean distance heuristic (straight-line distance)
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def find_path_astar(start_cell, goal_cell, occupancy_map_arg, allow_inflated_start=True, unknown_weight=3.0, emergency_clearance=False):
    rows, cols = occupancy_map_arg.shape
    temp_map = occupancy_map_arg.copy()  # Work on a copy
    
    # Apply emergency clearance if requested
    if emergency_clearance:
        start_row, start_col = start_cell
        clearance_radius = 8  # Cells to clear around robot
        
        for dr in range(-clearance_radius, clearance_radius + 1):
            for dc in range(-clearance_radius, clearance_radius + 1):
                r, c = start_row + dr, start_col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    # Only clear if it's inflated or unknown, not confirmed obstacles
                    if temp_map[r, c] == INFLATED_OCCUPIED_VALUE or temp_map[r, c] == P_UNKNOWN:
                        temp_map[r, c] = 0.0  # Treat as free space temporarily
    
    # Bounds checking
    if not (0 <= start_cell[0] < rows and 0 <= start_cell[1] < cols and
            0 <= goal_cell[0] < rows and 0 <= goal_cell[1] < cols):
        print(f"A* Error: Start or goal cell out of bounds: Start={start_cell}, Goal={goal_cell}")
        return []

    # Start cell validation
    start_is_obstacle = (occupancy_map_arg[start_cell[0], start_cell[1]] >= OCCUPIED_THRESHOLD or 
                        occupancy_map_arg[start_cell[0], start_cell[1]] == INFLATED_OCCUPIED_VALUE)
    if start_is_obstacle and not allow_inflated_start:
        global backup_state, backup_timer
        backup_state = True
        backup_timer = 0
        print(f"Obstacle detected at start position {start_cell}, initiating backup")
        return []  # Return empty path

    # Goal cell validation
    if (occupancy_map_arg[goal_cell[0], goal_cell[1]] >= OCCUPIED_THRESHOLD or 
        occupancy_map_arg[goal_cell[0], goal_cell[1]] == INFLATED_OCCUPIED_VALUE):
        print(f"A* Error: Goal cell is in obstacle/inflated zone: {goal_cell}")
        return []

    # Priority queue: (f_score, g_score, row, col)
    open_set = []
    heapq.heappush(open_set, (0, 0, start_cell[0], start_cell[1]))
    
    g_scores = np.full((rows, cols), float('inf'))
    g_scores[start_cell[0], start_cell[1]] = 0
    
    came_from = {}

    while open_set:
        current_f, current_g, current_r, current_c = heapq.heappop(open_set)

        if (current_r, current_c) == goal_cell:
            # Reconstruct path
            path = []
            while (current_r, current_c) in came_from:
                path.append((current_r, current_c))
                current_r, current_c = came_from[(current_r, current_c)]
            path.append(start_cell)
            return path[::-1]

        # Check all 8 neighbors
        for dr, dc, move_cost in [
            (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),         # Cardinal
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),       # Diagonal
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))
        ]:
            nr, nc = current_r + dr, current_c + dc
            
            # Skip out of bounds
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            cell_value = occupancy_map_arg[nr, nc]
            
            # Skip obstacles and inflated zones
            if cell_value >= OCCUPIED_THRESHOLD or cell_value == INFLATED_OCCUPIED_VALUE:
                continue

            # Calculate cost components
            base_cost = move_cost
            
            # Heavy penalty for unknown cells
            if cell_value == P_UNKNOWN:
                additional_cost = unknown_weight * move_cost  # Scale penalty by move cost
            else:
                additional_cost = get_clearance_cost(nr, nc, occupancy_map_arg, rows)
            
            tentative_g = current_g + base_cost + additional_cost

            if tentative_g < g_scores[nr, nc]:
                came_from[(nr, nc)] = (current_r, current_c)
                g_scores[nr, nc] = tentative_g
                f_score = tentative_g + heuristic((nr, nc), goal_cell)
                heapq.heappush(open_set, (f_score, tentative_g, nr, nc))

    print(f"A* Warning: No path found from {start_cell} to {goal_cell}")
    return []

def get_clearance_cost(r, c, occupancy_map_arg, map_size, clearance_radius_cells=8):
    cost = 0.0
    # Search in a square around the cell for obstacles
    for dr_c in range(-clearance_radius_cells, clearance_radius_cells + 1):
        for dc_c in range(-clearance_radius_cells, clearance_radius_cells + 1):
            nr, nc = r + dr_c, c + dc_c
            if 0 <= nr < map_size and 0 <= nc < map_size:
                cell_value = occupancy_map_arg[nr, nc]
                # If an obstacle (or inflated obstacle) is found
                if cell_value >= OCCUPIED_THRESHOLD or cell_value == INFLATED_OCCUPIED_VALUE:
                    distance_squared = dr_c**2 + dc_c**2
                    if distance_squared == 0:  # The cell itself is an obstacle
                        return float('inf')  # Should have been caught by impassable check, but as a safeguard
                    
                    # Apply a cost that increases as distance decreases
                    cost_factor = 1.5  # Adjust this to make the robot prefer more clearance
                    cost += cost_factor / math.sqrt(distance_squared)
    return cost

COLOR_RANGES = {
    'red': [(np.array([0, 100, 100]), np.array([10, 255, 255]))],
    'yellow': [(np.array([20, 100, 100]), np.array([40, 255, 255]))],
    'blue': [(np.array([100, 100, 100]), np.array([140, 255, 255]))]
}

# Initialize after robot creation
camera_rgb = robot.getDevice("camera rgb")
camera_depth = robot.getDevice("camera depth")
camera_params = {}

if camera_rgb:
    camera_rgb.enable(TIME_STEP)
    rgb_width = camera_rgb.getWidth()
    rgb_height = camera_rgb.getHeight()
    rgb_hfov = camera_rgb.getFov()
    aspect_ratio = rgb_height / rgb_width
    rgb_vfov = 2 * math.atan(math.tan(rgb_hfov / 2) * aspect_ratio)
    fx = rgb_width / (2 * math.tan(rgb_hfov / 2))
    fy = rgb_height / (2 * math.tan(rgb_vfov / 2))
    cx_img = rgb_width / 2
    cy_img = rgb_height / 2
    camera_params.update({
        'rgb_width': rgb_width, 'rgb_height': rgb_height,
        'fx': fx, 'fy': fy,
        'cx_img': cx_img, 'cy_img': cy_img
    })

if camera_depth:
    camera_depth.enable(TIME_STEP)
    camera_params.update({
        'depth_width': camera_depth.getWidth(),
        'depth_height': camera_depth.getHeight()
    })

# Color storage
color_points = {'red': set(), 'yellow': set(), 'blue': set()}
color_clean_counter = 0
color_processing_counter = 0  # New counter for periodic color processing

def create_color_masks(hsv_image):
    masks = {}
    kernel = np.ones((3,3), np.uint8)
    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            color_mask = cv2.inRange(hsv_image, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        masks[color] = mask
    return masks

def calculate_safe_approach_waypoint(pillar_world_coords, robot_world_coords, occupancy_map):
    safe_distance = ROBOT_INFLATION_EFFECTIVE_RADIUS + 0.1  # Extra buffer
    
    # Calculate direction vector
    dx = robot_world_coords[0] - pillar_world_coords[0]
    dy = robot_world_coords[1] - pillar_world_coords[1]
    magnitude = math.sqrt(dx**2 + dy**2)
    
    if magnitude < 0.001:  # Avoid division by zero
        dx, dy = 1.0, 0.0  # Default direction (east)
        magnitude = 1.0
        
    dx_unit = dx / magnitude
    dy_unit = dy / magnitude
    
    # Calculate nominal safe point
    safe_x = pillar_world_coords[0] + dx_unit * safe_distance
    safe_y = pillar_world_coords[1] + dy_unit * safe_distance
    
    # Convert to map coordinates
    safe_row, safe_col = world_to_map_coordinates(safe_x, safe_y)
    
    # Validate and adjust if in obstacle/inflated zone
    if not (0 <= safe_row < MAP_SIZE and 0 <= safe_col < MAP_SIZE):
        return safe_x, safe_y  # Out of bounds, return as-is
    
    cell_value = occupancy_map[safe_row, safe_col]
    if cell_value < OCCUPIED_THRESHOLD and cell_value != INFLATED_OCCUPIED_VALUE:
        return safe_x, safe_y  # Valid position
    
    # Find nearest valid cell if original is blocked
    return find_nearest_valid_cell(safe_row, safe_col, occupancy_map)

def find_nearest_obstacle(occupancy_map, start_row, start_col, max_search_radius=8):
    """Find the nearest obstacle to a given point."""
    if occupancy_map[start_row, start_col] >= OCCUPIED_THRESHOLD:
        return (start_row, start_col)
    
    rows, cols = occupancy_map.shape
    for radius in range(1, max_search_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue
                
                r, c = start_row + dr, start_col + dc
                if 0 <= r < rows and 0 <= c < cols and occupancy_map[r, c] >= OCCUPIED_THRESHOLD:
                    return (r, c)
    return None

def find_nearest_valid_cell(start_row, start_col, occupancy_map_arg, max_radius=20):
    """Find nearest free/unknown cell within specified radius"""
    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = start_row + dr, start_col + dc
                if 0 <= r < MAP_SIZE and 0 <= c < MAP_SIZE:
                    cell_value = occupancy_map_arg[r, c]
                    if cell_value < OCCUPIED_THRESHOLD and cell_value != INFLATED_OCCUPIED_VALUE:
                        return map_to_world_coordinates(r, c)
    # Fallback to original if no valid cell found
    return map_to_world_coordinates(start_row, start_col)

def check_path_for_obstacles(path, occupancy_map, lookahead_steps):
    if not path:
        return False
    lookahead = min(len(path), lookahead_steps)
    for i in range(lookahead):
        r, c = path[i]
        if not (0 <= r < MAP_SIZE and 0 <= c < MAP_SIZE):
            return True
        cell_value = occupancy_map[r, c]
        if cell_value >= OCCUPIED_THRESHOLD or cell_value == INFLATED_OCCUPIED_VALUE:
            return True
    return False

def get_object_world_coordinates(contour, depth_array, camera_params, position, heading):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    depth = depth_array[cy, cx]
    if depth < MIN_DEPTH or depth > MAX_PROCESSING_RANGE or math.isnan(depth):
        return None
    
    # Convert to 3D camera coordinates
    x_c = (cx - camera_params['cx_img']) * depth / camera_params['fx']
    y_c = (cy - camera_params['cy_img']) * depth / camera_params['fy']
    
    # Convert to robot frame
    x_robot = depth
    y_robot = -x_c  # Webots camera x is to the right, robot y is to the left (negative for conversion)
    
    # Convert to global coordinates
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    x_world = position[0] + x_robot * cos_h - y_robot * sin_h
    y_world = position[1] + x_robot * sin_h + y_robot * cos_h
    
    return x_world, y_world

def process_color_detection(camera_rgb, camera_depth, camera_params, position, heading, color_points):
    """Process RGB and depth images for color detection."""
    rgb_image = camera_rgb.getImage()
    depth_array = np.array(camera_depth.getRangeImage(), dtype=np.float32).reshape(
        (camera_params['depth_height'], camera_params['depth_width']))
    
    if not rgb_image or depth_array is None:
        return
    
    # Convert Webots image to OpenCV format
    rgb_array = np.frombuffer(rgb_image, dtype=np.uint8).reshape(
        (camera_params['rgb_height'], camera_params['rgb_width'], 4))
    bgr_array = rgb_array[:, :, :3]
    hsv = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
    
    masks = create_color_masks(hsv)
    
    # Process each color
    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                world_coords = get_object_world_coordinates(
                    contour, depth_array, camera_params, position, heading)
                if world_coords:
                    row, col = world_to_map_coordinates(world_coords[0], world_coords[1])
                    if 0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE:
                        color_points[color].add((row, col))

def clean_isolated_color_points(color_points):
    """Remove isolated color points that don't have nearby neighbors."""
    for color_set in color_points.values():
        if len(color_set) < 2:
            continue
            
        to_remove = set()
        points_list = list(color_set)
        
        for idx, point in enumerate(points_list):
            row, col = point
            neighbor_distances = []
            
            for other_idx, other_point in enumerate(points_list):
                if idx == other_idx:
                    continue
                r, c = other_point
                distance = ((row - r) ** 2 + (col - c) ** 2) ** 0.5
                neighbor_distances.append(distance)
            
            neighbor_distances.sort()
            closest_neighbors = neighbor_distances[:6]
            close_count = sum(1 for d in closest_neighbors if d <= 6.0)
            
            if close_count < 2:
                to_remove.add(point)
        if len(color_set) > 2:
            color_set -= to_remove

def align_color_points_to_obstacles(color_points, occupancy_map):
    """Align color points to nearest obstacles on the map."""
    for color_set in color_points.values():
        to_remove = set()
        to_add = set()
        
        for point in color_set:
            row, col = point
            if occupancy_map[row, col] < 0.1:
                nearest_obstacle = find_nearest_obstacle(occupancy_map, row, col)
                if nearest_obstacle:
                    to_remove.add(point)
                    to_add.add(nearest_obstacle)
        
        color_set -= to_remove
        color_set.update(to_add)

def get_world_center(color):
    if not color_points[color]:
        return None
    world_points = []
    for (row, col) in color_points[color]:
        x, y = map_to_world_coordinates(row, col)
        world_points.append([x, y])
    world_points = np.array(world_points)
    return world_points.mean(axis=0)  # [x_center, y_center]

def get_world_centers():
    blue_center = get_world_center('blue')
    yellow_center = get_world_center('yellow')
    return blue_center, yellow_center

# --- Visualization Setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(7, 7))
ax.axis('off')
plt.tight_layout(pad=0.2)
def update_visualization(img_display, ax, occupancy_map, color_points, position, heading, 
                        cross_artist, arrow_artist):
    """Update the visualization with current map and robot state."""
    # Create RGB display image
    display_img = np.zeros((MAP_SIZE, MAP_SIZE, 3))
    display_img[:, :, 0] = occupancy_map
    display_img[:, :, 1] = occupancy_map
    display_img[:, :, 2] = occupancy_map
    
    # Add colored objects
    color_map = {'red': [1.0, 0.0, 0.0], 'yellow': [1.0, 1.0, 0.0], 'blue': [0.0, 0.0, 1.0], 'green': [0.0, 1.0, 0.0]}
    for color, points in color_points.items():
        for (row, col) in points:
            display_img[row, col] = color_map[color]
    
    # Update display
    img_display.set_data(display_img)
    
    # Update robot position
    row, col = world_to_map_coordinates(position[0], position[1])
    if cross_artist:
        cross_artist.remove()
        arrow_artist.remove()
    cross_artist, arrow_artist = plot_robot(ax, row, col, heading)
    
    plt.draw()
    plt.pause(0.01)
    return cross_artist, arrow_artist

img_display = ax.imshow(occupancy_map, cmap='gray', vmin=0, vmax=1.0, origin='upper')
plt.show(block=False)

plot_counter = 0
clean_counter = 0

# --- Red Wall Avoidance Algorithm ---
def extend_red_to_robot(red_points, robot_position, occupancy_map, max_extension=30, circle_radius=5):

    if not red_points:
        return occupancy_map

    # Convert robot position to map coordinates
    robot_row, robot_col = world_to_map_coordinates(robot_position[0], robot_position[1])
    endpoints = set()

    for (red_row, red_col) in red_points:
        # Use Bresenham's line algorithm (parameters in col,row order!)
        line_cells = bresenham_line(red_col, red_row, robot_col, robot_row)
        
        # Process line cells (skip first point - the red point itself)
        for i, (col, row) in enumerate(line_cells[1:], 1):
            if i > max_extension:
                break
            
            # Boundary check
            if not (0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE):
                continue
                
            # Stop at obstacles
            if occupancy_map[row, col] >= OCCUPIED_THRESHOLD:
                break
                
            # Mark as inflated obstacle
            occupancy_map[row, col] = INFLATED_OCCUPIED_VALUE
            
            # Record endpoints
            if i == len(line_cells[1:])-1 or i == max_extension:
                endpoints.add((row, col))

    # Create clearance circles
    for (center_row, center_col) in endpoints:
        min_row = max(0, center_row - circle_radius)
        max_row = min(MAP_SIZE-1, center_row + circle_radius)
        min_col = max(0, center_col - circle_radius)
        max_col = min(MAP_SIZE-1, center_col + circle_radius)
        
        for row in range(min_row, max_row+1):
            for col in range(min_col, max_col+1):
                dist_sq = (row - center_row)**2 + (col - center_col)**2
                if dist_sq <= circle_radius**2:
                    if occupancy_map[row, col] < OCCUPIED_THRESHOLD:
                        occupancy_map[row, col] = INFLATED_OCCUPIED_VALUE

    return occupancy_map

# Devices
motor_names = {
    "fl": "fl_wheel_joint", "fr": "fr_wheel_joint",
    "rl": "rl_wheel_joint", "rr": "rr_wheel_joint"
}
motors = {key: robot.getDevice(name) for key, name in motor_names.items()}
for motor in motors.values():
    if motor:
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)

lidar = robot.getDevice("laser")
if lidar:
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud()

# --- Main loop ---
while robot.step(TIME_STEP) != -1:
    position, rotation = get_translation_rotation()
    heading = axis_angle(rotation)
    robot_world_x, robot_world_y = position[0], position[1]
    robot_map_r, robot_map_c = world_to_map_coordinates(robot_world_x, robot_world_y)
    current_time = robot.getTime()
    
    # Clean up failed targets
    recently_failed_targets = [
        (rc, timestamp) for (rc, timestamp) in recently_failed_targets
        if current_time - timestamp < FAILED_TARGET_COOLDOWN
    ]
    
    # --- Keyboard Input and Mode Toggle ---
    key = keyboard.getKey()
    if key == ord('A'):
        autonomous_mode = not autonomous_mode
        if autonomous_mode:
            print("Switched to autonomous mode.")
            # Reset all state variables
            blue_target_detected = None
            yellow_target_detected = None
            blue_reached = False
            mission_complete = False
            initial_movement_counter = 0
            current_target = None 
            current_path = [] 
            last_robot_position = list(position)
            recently_failed_targets.clear()
            mission_state = INITIAL_SCAN
            initial_scan_progress = 0.0
            print("Starting initial 360-degree scan for color detection.")
        else:
            print("Switched to manual control mode.")
            for motor in motors.values():
                if motor:
                    motor.setVelocity(0.0)
         
    # --- Lidar Data Processing ---
    lidar_beams_data = []
    if lidar:
        lidar_beams_data = get_lidar_data_in_world(lidar, position, heading)

    # --- Control Logic ---
    left_speed = 0.0
    right_speed = 0.0

    if autonomous_mode:
        # Periodic color detection
        color_processing_counter += 1
        if color_processing_counter >= 10:
            process_color_detection(camera_rgb, camera_depth, camera_params, 
                                   position, heading, color_points)
            blue_target_detected, yellow_target_detected = get_world_centers()
            color_processing_counter = 0

            # Extend red walls towards the robot to avoid dead ends
            # if 'red' in color_points and color_points['red']:
            #     occupancy_map = extend_red_to_robot(
            #         color_points['red'],
            #         position,
            #         occupancy_map,
            #         max_extension=10,
            #         circle_radius=8
            #     )

        # Clean color points periodically
        color_clean_counter += 1
        if color_clean_counter >= 100:
            clean_isolated_color_points(color_points)
            if color_clean_counter >= 500:
                align_color_points_to_obstacles(color_points, occupancy_map)
                color_clean_counter = 0

        # Mission State Machine
        if mission_state == INITIAL_SCAN:
            if initial_scan_progress < INITIAL_SCAN_DURATION_STEPS:
                left_speed = -BASE_SPEED * TURN_SPEED_RATIO
                right_speed = BASE_SPEED * TURN_SPEED_RATIO
                initial_scan_progress += 1
            else:
                mission_state = EXPLORE
                print("Initial scan complete. Starting exploration.")
                
        elif mission_state == EXPLORE:
            # Prioritize blue pillar if detected and not reached
            if blue_target_detected is not None and not blue_reached and blue_attempt_count < MAX_PILLAR_ATTEMPTS:
                # Calculate safe approach point
                safe_x, safe_y = calculate_safe_approach_waypoint(
                    blue_target_detected, 
                    (position[0], position[1]), occupancy_map)
                safe_rc = world_to_map_coordinates(safe_x, safe_y)
                
                # Check if safe point is valid
                if (0 <= safe_rc[0] < MAP_SIZE and 0 <= safe_rc[1] < MAP_SIZE and
                    occupancy_map[safe_rc[0], safe_rc[1]] < OCCUPIED_THRESHOLD and
                    occupancy_map[safe_rc[0], safe_rc[1]] != INFLATED_OCCUPIED_VALUE):
                    
                    # Plan path to blue
                    current_path = find_path_astar(
                        (robot_map_r, robot_map_c),
                        safe_rc,
                        occupancy_map,
                        allow_inflated_start=False,
                        unknown_weight=3.0,
                        emergency_clearance=True
                    )
                    
                    if current_path:
                        print("Direct path to blue pillar found! Attempting immediate path")
                        current_target = {'x': safe_x, 'y': safe_y, 'map_rc': safe_rc}
                        mission_state = GO_TO_BLUE
                        blue_attempt_count = 0
                    else:
                        print("No direct path to blue pillar. Staying in exploration.")
                        blue_attempt_count += 1
                else:
                    print("Invalid safe point for blue pillar. Skipping.")
                    blue_attempt_count += 1
            
            # After blue reached, prioritize yellow pillar
            elif blue_reached and yellow_target_detected is not None and yellow_attempt_count < MAX_PILLAR_ATTEMPTS:
                # Calculate safe approach point
                safe_x, safe_y = calculate_safe_approach_waypoint(
                    yellow_target_detected,
                    (position[0], position[1]), occupancy_map)
                safe_rc = world_to_map_coordinates(safe_x, safe_y)
                
                # Check if safe point is valid
                if (0 <= safe_rc[0] < MAP_SIZE and 0 <= safe_rc[1] < MAP_SIZE and
                    occupancy_map[safe_rc[0], safe_rc[1]] < OCCUPIED_THRESHOLD and
                    occupancy_map[safe_rc[0], safe_rc[1]] != INFLATED_OCCUPIED_VALUE):
                    
                    # Plan path to yellow
                    current_path = find_path_astar(
                        (robot_map_r, robot_map_c),
                        safe_rc,
                        occupancy_map,
                        allow_inflated_start=False,
                        unknown_weight=3.0,
                        emergency_clearance=True
                    )
                    
                    if current_path:
                        print("Direct path to yellow pillar found! Attempting immediate path")
                        current_target = {'x': safe_x, 'y': safe_y, 'map_rc': safe_rc}
                        mission_state = GO_TO_YELLOW
                        yellow_attempt_count = 0
                    else:
                        print("No direct path to yellow pillar. Staying in exploration.")
                        yellow_attempt_count += 1
                else:
                    print("Invalid safe point for yellow pillar. Skipping.")
                    yellow_attempt_count += 1
            
            # If no pillars to approach, do normal exploration
            if current_target is None or not current_path:
                compute_costs(occupancy_map, cost_map)
                current_target = set_target(occupancy_map, cost_map, position)
                
                if current_target:
                    print(f"New target selected: ({current_target['x']:.2f}, {current_target['y']:.2f})")
                    start_cell = (robot_map_r, robot_map_c)
                    goal_cell = current_target['map_rc']
                    
                    current_path = find_path_astar(
                        start_cell, 
                        goal_cell, 
                        occupancy_map,
                        allow_inflated_start=False,
                        unknown_weight=3.0,
                        emergency_clearance=True
                    )
                    
                    # if not current_path:
                    #     print("Path planning failed. Blacklisting target.")
                    #     recently_failed_targets.append((current_target['map_rc'], current_time))
                    #     current_target = None

            # Follow path if valid
            if current_target and current_path:
                # Check for obstacles in path
                obstacle_detected = False
                lookahead = min(len(current_path), PATH_SIMPLIFICATION_LOOKAHEAD)
                for i in range(lookahead):
                    r, c = current_path[i]
                    if not (0 <= r < MAP_SIZE and 0 <= c < MAP_SIZE):
                        obstacle_detected = True
                        break
                    cell_value = occupancy_map[r, c]
                    if cell_value >= OCCUPIED_THRESHOLD or cell_value == INFLATED_OCCUPIED_VALUE:
                        obstacle_detected = True
                        break
                
                if obstacle_detected:
                    print("Obstacle detected in path. Re-planning.")
                    current_path = []
                else:
                    # Follow the path
                    next_waypoint_map_r, next_waypoint_map_c = current_path[0]
                    next_waypoint_world_x, next_waypoint_world_y = map_to_world_coordinates(
                        next_waypoint_map_r, next_waypoint_map_c)

                    left_speed, right_speed, dist_to_waypoint = go_2_target(
                        position, heading, next_waypoint_world_x, next_waypoint_world_y)

                    dist_pixels = math.sqrt(
                        (next_waypoint_map_r - robot_map_r)**2 + 
                        (next_waypoint_map_c - robot_map_c)**2)
                    
                    if dist_pixels < WAYPOINT_REACHED_THRESHOLD_CELLS:
                        current_path.pop(0)

                    # Check if reached final target
                    final_target_map_r, final_target_map_c = current_target['map_rc']
                    dist_to_final_target_pixels = math.sqrt(
                        (final_target_map_r - robot_map_r)**2 + 
                        (final_target_map_c - robot_map_c)**2)
                    
                    if dist_to_final_target_pixels < WAYPOINT_REACHED_THRESHOLD_CELLS:
                        recently_reached_targets.append(current_target['map_rc'])
                        current_target = None
                        current_path = []
        
        elif mission_state == GO_TO_BLUE:
            if blue_target_detected is None:
                print("Blue target lost. Reverting to exploration.")
                mission_state = EXPLORE
                current_target = None
                current_path = []
            else:
                # Replan if needed
                if current_target is None or not current_path:
                    safe_x, safe_y = calculate_safe_approach_waypoint(
                        blue_target_detected, 
                        (position[0], position[1]), occupancy_map)
                    safe_rc = world_to_map_coordinates(safe_x, safe_y)
                    current_target = {'x': safe_x, 'y': safe_y, 'map_rc': safe_rc}
                    
                    current_path = find_path_astar(
                        (robot_map_r, robot_map_c),
                        current_target['map_rc'],
                        occupancy_map,
                        allow_inflated_start=False,
                        unknown_weight=3.0,
                        emergency_clearance=True
                    )
                    
                    if not current_path:
                        print("Path to blue target blocked. Reverting to exploration.")
                        blue_attempt_count += 1
                        mission_state = EXPLORE
                
                # Follow path if valid
                if current_target and current_path:
                    # Check for obstacles in path
                    if check_path_for_obstacles(current_path, occupancy_map, PATH_SIMPLIFICATION_LOOKAHEAD):
                        print("Obstacle detected in path to blue. Re-planning.")
                        current_path = []
                    else:
                        next_waypoint_map_r, next_waypoint_map_c = current_path[0]
                        next_waypoint_world_x, next_waypoint_world_y = map_to_world_coordinates(
                            next_waypoint_map_r, next_waypoint_map_c)

                        left_speed, right_speed, dist_to_waypoint = go_2_target(
                            position, heading, next_waypoint_world_x, next_waypoint_world_y)

                        dist_pixels = math.sqrt(
                            (next_waypoint_map_r - robot_map_r)**2 + 
                            (next_waypoint_map_c - robot_map_c)**2)
                        
                        if dist_pixels < WAYPOINT_REACHED_THRESHOLD_CELLS:
                            current_path.pop(0)

                        # Check if reached blue target
                        dist_to_target = math.sqrt(
                            (position[0] - blue_target_detected[0])**2 + 
                            (position[1] - blue_target_detected[1])**2)
                        
                        if dist_to_target < TARGET_REACHED_THRESHOLD:
                            print("Reached blue target!")
                            blue_reached = True
                            mission_state = EXPLORE
                            print("Blue pillar reached. Now looking for yellow pillar.")
                            current_target = None
                            current_path = []

        elif mission_state == GO_TO_YELLOW:
            if yellow_target_detected is None:
                print("Yellow target lost. Reverting to exploration.")
                mission_state = EXPLORE
                current_target = None
                current_path = []
            else:
                # Replan if needed
                if current_target is None or not current_path:
                    safe_x, safe_y = calculate_safe_approach_waypoint(
                        yellow_target_detected, 
                        (position[0], position[1]), occupancy_map)
                    safe_rc = world_to_map_coordinates(safe_x, safe_y)
                    current_target = {'x': safe_x, 'y': safe_y, 'map_rc': safe_rc}
                    
                    current_path = find_path_astar(
                        (robot_map_r, robot_map_c),
                        current_target['map_rc'],
                        occupancy_map,
                        allow_inflated_start=False,
                        unknown_weight=3.0,
                        emergency_clearance=True
                    )
                    
                    if not current_path:
                        print("Path to yellow target blocked. Reverting to exploration.")
                        yellow_attempt_count += 1
                        mission_state = EXPLORE
                
                # Follow path if valid
                if current_target and current_path:
                    # Check for obstacles in path
                    if check_path_for_obstacles(current_path, occupancy_map, PATH_SIMPLIFICATION_LOOKAHEAD):
                        print("Obstacle detected in path to yellow. Re-planning.")
                        current_path = []
                    else:
                        next_waypoint_map_r, next_waypoint_map_c = current_path[0]
                        next_waypoint_world_x, next_waypoint_world_y = map_to_world_coordinates(
                            next_waypoint_map_r, next_waypoint_map_c)

                        left_speed, right_speed, dist_to_waypoint = go_2_target(
                            position, heading, next_waypoint_world_x, next_waypoint_world_y)

                        dist_pixels = math.sqrt(
                            (next_waypoint_map_r - robot_map_r)**2 + 
                            (next_waypoint_map_c - robot_map_c)**2)
                        
                        if dist_pixels < WAYPOINT_REACHED_THRESHOLD_CELLS:
                            current_path.pop(0)

                        # Check if reached yellow target
                        dist_to_target = math.sqrt(
                            (position[0] - yellow_target_detected[0])**2 + 
                            (position[1] - yellow_target_detected[1])**2)
                        
                        if dist_to_target < TARGET_REACHED_THRESHOLD:
                            print("Reached yellow target! Mission complete.")
                            mission_state = MISSION_COMPLETE
                            mission_complete = True

        elif mission_state == MISSION_COMPLETE:
            left_speed, right_speed = 0.0, 0.0
            if not mission_complete:
                print("Mission complete. Robot is idle.")
                mission_complete = True

    else:  # Manual control mode
        if key == Keyboard.UP:
            left_speed = BASE_SPEED
            right_speed = BASE_SPEED
        elif key == Keyboard.DOWN:
            left_speed = -BASE_SPEED
            right_speed = -BASE_SPEED
        elif key == Keyboard.LEFT:
            left_speed = -BASE_SPEED * TURN_SPEED_RATIO
            right_speed = BASE_SPEED * TURN_SPEED_RATIO
        elif key == Keyboard.RIGHT:
            left_speed = BASE_SPEED * TURN_SPEED_RATIO
            right_speed = -BASE_SPEED * TURN_SPEED_RATIO
        else:  # Stop if no key pressed
            left_speed = 0.0
            right_speed = 0.0

    # --- Apply Motor Commands ---
    left_speed = np.clip(left_speed, -MAX_VELOCITY, MAX_VELOCITY)
    right_speed = np.clip(right_speed, -MAX_VELOCITY, MAX_VELOCITY)

    if motors["fl"]: motors["fl"].setVelocity(left_speed)
    if motors["rl"]: motors["rl"].setVelocity(left_speed)
    if motors["fr"]: motors["fr"].setVelocity(right_speed)
    if motors["rr"]: motors["rr"].setVelocity(right_speed)

    # --- Update Occupancy Grid ---
    if lidar_beams_data:
        update_occupancy_grid(occupancy_map, lidar_beams_data, position, heading)
        
        # Apply map cleaning
        clean_counter += 1
        if clean_counter >= MAP_CLEAN_INTERVAL:
            occupancy_map = clean_isolated_pixels(occupancy_map, OCCUPIED_THRESHOLD, MAP_SIZE)
            occupancy_map = remove_isolated_frontiers(occupancy_map)
            clean_counter = 0
    mark_permanent_free_space(occupancy_map, robot_map_r, robot_map_c)
    # --- Visualization ---
    plot_counter += 1
    if plot_counter >= PLOT_UPDATE_INTERVAL:
        # Update robot position
        row, col = world_to_map_coordinates(position[0], position[1])
        if cross_artist:
            cross_artist.remove()
            arrow_artist.remove()
        cross_artist, arrow_artist = plot_robot(ax, row, col, heading)

        # Plot current target
        if target_artist:
            target_artist.remove() 
        if current_target:
            target_row, target_col = current_target['map_rc']
            target_artist = ax.plot(target_col, target_row, 'ro', markersize=8, markeredgewidth=2)[0] 
        else:
            target_artist = None

        # Clear previous path
        if path_artist:
            path_artist.remove() 
        if current_path:
            path_cols = [robot_map_c] + [p[1] for p in current_path]
            path_rows = [robot_map_r] + [p[0] for p in current_path]
            path_artist = ax.plot(path_cols, path_rows, 'w--', linewidth=1, alpha=0.7)[0] 
        else:
            path_artist = None

        # Create RGB display image
        display_img = np.zeros((MAP_SIZE, MAP_SIZE, 3))
        display_img[:, :, 0] = occupancy_map
        display_img[:, :, 1] = occupancy_map
        display_img[:, :, 2] = occupancy_map
        if backup_state:
    # Apply backup speeds
            left_speed = right_speed = -2  # Slightly slower in reverse
                        # Apply motor commands and skip rest of control logic

            # Update timer and check duration
            backup_timer += 1
            if backup_timer >= BACKUP_DURATION:
                backup_state = False
                print("Backup maneuver completed")
                continue
            if motors["fl"]: motors["fl"].setVelocity(left_speed)
            if motors["rl"]: motors["rl"].setVelocity(left_speed)
            if motors["fr"]: motors["fr"].setVelocity(right_speed)
            if motors["rr"]: motors["rr"].setVelocity(right_speed)
            continue
            
  # Skip the rest of the control logic for this timestep
        # Add colored objects
        color_map = {'red': [1.0, 0.0, 0.0], 'yellow': [1.0, 1.0, 0.0], 'blue': [0.0, 0.0, 1.0], 'green': [0.0, 1.0, 0.0]}
        for color, points in color_points.items():
            for (row, col) in points:
                display_img[row, col] = color_map[color]
        
        # Update display
        img_display.set_data(display_img)
        
        plt.draw()
        plt.pause(0.01)
        plot_counter = 0