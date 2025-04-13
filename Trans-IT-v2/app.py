# Import required libraries for mapping, web requests, and Flask application
import folium
from folium import Marker
from folium.plugins import LocateControl, Fullscreen, MeasureControl
import requests
import polyline
import json
import os
from flask import Flask, render_template_string, send_from_directory, request
import logging
from datetime import datetime, timedelta
import hashlib
import math

# Initialize Flask application
app = Flask(__name__)

# Configure logging to debug level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define cache directory for storing route data
CACHE_DIR = 'route_cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Define restricted areas as [min_lat, min_lon, max_lat, max_lon]
RESTRICTED_AREAS = [
    [51.5000, -0.1300, 51.5200, -0.1100],  # Example near London
]

def validate_waypoints(waypoints_list):
    validated = []
    error_msg = None
    if not isinstance(waypoints_list, list):
        return [], "Invalid waypoints format"
    for wp in waypoints_list:
        if not isinstance(wp, dict) or 'lat' not in wp or 'lng' not in wp:
            error_msg = "Waypoint format incorrect"
            continue
        point = validate_coordinates(wp.get('lat'), wp.get('lng'))
        if point:
            validated.append(point)
        else:
            error_msg = "Invalid coordinates found"
    if len(validated) > 2:
        validated = validated[:2]
        error_msg = "Only first two waypoints used"
    return validated, error_msg

def validate_coordinates(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        return (lat, lon)
    except (ValueError, TypeError):
        return None

def haversine_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def point_to_line_distance(point, line):
    point_lat, point_lon = point
    start_lat, start_lon = line[0]
    end_lat, end_lon = line[1]
    km_per_lat = 111.0
    km_per_lon = 111.0 * math.cos(math.radians((start_lat + end_lat) / 2))
    x1, y1 = start_lon * km_per_lon, start_lat * km_per_lat
    x2, y2 = end_lon * km_per_lon, end_lat * km_per_lat
    x0, y0 = point_lon * km_per_lon, point_lat * km_per_lat
    line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
    if line_length_sq == 0:
        return haversine_distance(point, line[0])
    t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / line_length_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    proj_lon = proj_x / km_per_lon
    proj_lat = proj_y / km_per_lat
    return haversine_distance(point, (proj_lat, proj_lon))

def get_route_osrm(coordinates, alternatives=3, mode="driving"):
    if mode not in ["driving", "walking", "cycling"]:
        mode = "driving"
    base_url = f"http://router.project-osrm.org/route/v1/{mode}/"
    coordinate_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
    url = f"{base_url}{coordinate_str}?overview=full&geometries=polyline&steps=true&annotations=true&alternatives={alternatives}"
    try:
        logger.debug(f"Fetching {mode} route from OSRM: {url}")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        routes = []
        if 'routes' in data and data['routes']:
            for route in data['routes']:
                decoded_route = polyline.decode(route['geometry'])
                distance = route['distance'] / 1000
                duration = route['duration'] / 60
                steps = route['legs'][0]['steps'] if 'legs' in route and route['legs'] else []
                annotations = route.get('annotations', {})
                routes.append((decoded_route, distance, duration, steps, annotations))
        logger.debug(f"Found {len(routes)} valid {mode} routes")
        return routes if routes else None
    except requests.RequestException as e:
        logger.error(f"Error fetching route: {str(e)}")
        return None

def estimate_traffic_density(route_data):
    try:
        route_coords, distance, duration, steps, annotations = route_data
        if 'speed' in annotations and annotations['speed']:
            speeds = annotations['speed']
            avg_speed = sum(speeds) / len(speeds) if speeds else 30
            now = datetime.now()
            hour = now.hour
            weekday = now.weekday()
            time_multiplier = 1.0
            if weekday < 5:
                if 7 <= hour < 9 or 16 <= hour < 18:
                    time_multiplier = 0.7
                elif 22 <= hour < 6:
                    time_multiplier = 1.3
            else:
                if 10 <= hour < 18:
                    time_multiplier = 0.8
                else:
                    time_multiplier = 1.2
            traffic_efficiency = min(1.0, max(0.2, (avg_speed / 50) * time_multiplier))
            segments = []
            segment_length = max(1, len(route_coords) // 10)
            import random
            random.seed(sum([hash(str(coord)) for coord in route_coords[:5]]))
            prev_efficiency = traffic_efficiency
            for i in range(0, len(route_coords), segment_length):
                segment_end = min(i + segment_length, len(route_coords))
                variation = random.uniform(-0.15, 0.15)
                inertia = 0.7
                segment_traffic = max(0.1, min(1.0, 
                                             inertia * prev_efficiency + 
                                             (1-inertia) * (traffic_efficiency + variation)))
                segments.append((i, segment_end, segment_traffic))
                prev_efficiency = segment_traffic
            return segments
        else:
            expected_speed = distance / (duration / 60) if duration > 0 else 30
            traffic_efficiency = min(1.0, max(0.2, expected_speed / 40))
            segments = []
            segment_length = max(1, len(route_coords) // 8)
            import random
            seed_value = int(hash(f"{route_coords[0][0]}{route_coords[-1][0]}") % 10000)
            random.seed(seed_value)
            for i in range(0, len(route_coords), segment_length):
                segment_end = min(i + segment_length, len(route_coords))
                variation = random.uniform(-0.15, 0.15)
                segment_traffic = max(0.1, min(1.0, traffic_efficiency + variation))
                segments.append((i, segment_end, segment_traffic))
            return segments
    except Exception as e:
        logger.error(f"Error estimating traffic: {str(e)}")
        return None

def traffic_color(efficiency):
    if efficiency >= 0.7:
        return '#4CAF50'  # Green
    elif efficiency >= 0.4:
        return '#FFC107'  # Yellow
    else:
        return '#F44336'  # Red

def is_route_restricted(route_coords):
    for lat, lon in route_coords:
        for area in RESTRICTED_AREAS:
            min_lat, min_lon, max_lat, max_lon = area
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                logger.debug(f"Route rejected: passes through restricted area {area}")
                return True
    return False

def is_route_within_bounds(route_coords, bounds, tolerance=0.9):
    min_lat, min_lon, max_lat, max_lon = bounds
    within_bounds_count = 0
    total_points = len(route_coords)
    for lat, lon in route_coords:
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            within_bounds_count += 1
    if total_points == 0:
        return False
    return (within_bounds_count / total_points) >= tolerance

def deduplicate_routes(routes, threshold=0.7, sample_points=20):
    if not routes:
        return []
    unique_routes = []
    for route in routes:
        route_geometry = route[0]
        step = max(1, len(route_geometry) // sample_points)
        sampled_geometry = route_geometry[::step]
        is_unique = True
        for unique_route in unique_routes:
            unique_geometry = unique_route[0][::step]
            if len(sampled_geometry) > 0 and len(unique_geometry) > 0:
                min_length = min(len(sampled_geometry), len(unique_geometry))
                matches = 0
                points_to_check = [0, min_length//2, min_length-1] if min_length >= 3 else list(range(min_length))
                for i in points_to_check:
                    if (abs(sampled_geometry[i][0] - unique_geometry[i][0]) < 0.001 and
                        abs(sampled_geometry[i][1] - unique_geometry[i][1]) < 0.001):
                        matches += 1
                similarity = matches / len(points_to_check)
                if similarity > threshold:
                    is_unique = False
                    if route[1] < unique_route[1]:
                        unique_routes.remove(unique_route)
                        unique_routes.append(route)
                    break
        if is_unique:
            unique_routes.append(route)
    return unique_routes

def get_all_routes_osrm(waypoints, mode="driving"):
    if len(waypoints) != 2:
        logger.error("Exactly 2 waypoints (start and end) are required")
        return [], None
    routes = []
    start = waypoints[0]
    end = waypoints[1]
    distance_km = haversine_distance(start, end)
    if distance_km < 0.2:
        direct_line = [start, end]
        route_data = get_route_osrm(waypoints, alternatives=1, mode=mode)
        if route_data:
            return route_data, [
                min(start[0], end[0]) - 0.01, 
                min(start[1], end[1]) - 0.01,
                max(start[0], end[0]) + 0.01, 
                max(start[1], end[1]) + 0.01
            ]
        return [], None
    avg_lat = (start[0] + end[0]) / 2
    buffer_percentage = 0.25 if distance_km < 5 else (0.15 if distance_km < 20 else 0.10)
    buffer_km = max(1.0, min(5, distance_km * buffer_percentage))
    buffer_lat = buffer_km / 111.0
    buffer_lon = buffer_km / (111.0 * math.cos(math.radians(avg_lat)))
    bounds = [
        min(start[0], end[0]) - buffer_lat,
        min(start[1], end[1]) - buffer_lon,
        max(start[0], end[0]) + buffer_lat,
        max(start[1], end[1]) + buffer_lon
    ]
    tolerance = 0.9 if distance_km < 3 else (0.85 if distance_km < 10 else 0.8)
    direct_routes = get_route_osrm(waypoints, alternatives=3, mode=mode)
    if direct_routes:
        routes.extend(direct_routes)
    direction_lat = end[0] - start[0]
    direction_lon = end[1] - start[1]
    magnitude = math.sqrt(direction_lat**2 + direction_lon**2)
    if magnitude > 0:
        unit_lat = direction_lat / magnitude
        unit_lon = direction_lon / magnitude
        perp_lat = -unit_lon
        perp_lon = unit_lat
        offset_scale = min(buffer_lat * 0.5, 0.001)
        third_point = (start[0] + direction_lat * 1/3, start[1] + direction_lon * 1/3)
        two_thirds_point = (start[0] + direction_lat * 2/3, start[1] + direction_lon * 2/3)
        offsets = [
            (third_point[0] + perp_lat * offset_scale, third_point[1] + perp_lon * offset_scale),
            (third_point[0] - perp_lat * offset_scale, third_point[1] - perp_lon * offset_scale),
            (two_thirds_point[0] + perp_lat * offset_scale, two_thirds_point[1] + perp_lon * offset_scale),
            (two_thirds_point[0] - perp_lat * offset_scale, two_thirds_point[1] - perp_lon * offset_scale),
        ]
        for alt_waypoint in offsets:
            if bounds[0] <= alt_waypoint[0] <= bounds[2] and bounds[1] <= alt_waypoint[1] <= bounds[3]:
                test_waypoints = [start, alt_waypoint, end]
                alt_routes = get_route_osrm(test_waypoints, alternatives=1, mode=mode)
                if alt_routes:
                    routes.extend(alt_routes)
    unique_routes = deduplicate_routes(routes)
    valid_routes = []
    for route in unique_routes:
        if is_route_restricted(route[0]):
            continue
        if is_route_within_bounds(route[0], bounds, tolerance):
            max_distance = 0
            direct_line = [start, end]
            sample_step = max(1, len(route[0]) // 20)
            for i in range(0, len(route[0]), sample_step):
                point = route[0][i]
                point_distance = point_to_line_distance(point, direct_line)
                max_distance = max(max_distance, point_distance)
            max_allowed_distance = max(1.5, distance_km * buffer_percentage * 3.0)
            if max_distance <= max_allowed_distance:
                valid_routes.append(route)
    valid_routes.sort(key=lambda x: (x[2], x[1]))
    return valid_routes[:5], bounds  # Limit to 5 routes for performance

def load_or_fetch_routes(waypoints, mode="driving"):
    if len(waypoints) != 2:
        return [], None
    waypoint_str = "-".join([f"{lat:.6f},{lon:.6f}" for lat, lon in waypoints])
    mode_key = f"{waypoint_str}_{mode}"
    hash_key = hashlib.md5(mode_key.encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{hash_key}.json")
    now = datetime.now()
    current_hour = now.hour
    time_period = "rush" if (7 <= current_hour <= 9 or 16 <= current_hour <= 18) else ("night" if 22 <= current_hour <= 5 else "normal")
    time_sensitive_cache = os.path.join(CACHE_DIR, f"{hash_key}_{time_period}.json")
    if os.path.exists(time_sensitive_cache):
        file_age = now.timestamp() - os.path.getmtime(time_sensitive_cache)
        max_age = 10800 if time_period == "rush" else 43200
        if file_age < max_age:
            try:
                with open(time_sensitive_cache, 'r') as f:
                    return json.load(f), None
            except Exception as e:
                logger.warning(f"Time-sensitive cache read error: {str(e)}")
    if os.path.exists(cache_file):
        file_age = now.timestamp() - os.path.getmtime(cache_file)
        if file_age < 86400:
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f), None
            except Exception as e:
                logger.warning(f"Cache read error: {str(e)}")
    routes, bounds = get_all_routes_osrm(waypoints, mode)
    serialized_routes = []
    for route_data in routes:
        coords, distance, duration, steps, annotations = route_data
        simplified_annotations = {}
        if annotations and isinstance(annotations, dict):
            for key, values in annotations.items():
                if isinstance(values, list) and len(values) > 100:
                    step = max(1, len(values) // 100)
                    simplified_annotations[key] = values[::step]
                else:
                    simplified_annotations[key] = values
        traffic_segments = estimate_traffic_density(route_data)
        serialized_route = {
            "coords": coords,
            "distance": distance,
            "duration": duration,
            "steps": steps[:50] if steps else [],
            "annotations": simplified_annotations,
            "traffic": traffic_segments,
            "timestamp": now.timestamp(),
            "time_period": time_period
        }
        serialized_routes.append(serialized_route)
    try:
        with open(cache_file, 'w') as f:
            json.dump(serialized_routes, f)
        with open(time_sensitive_cache, 'w') as f:
            json.dump(serialized_routes, f)
    except Exception as e:
        logger.warning(f"Cache write error: {str(e)}")
    return serialized_routes, bounds

@app.route('/', methods=['GET', 'POST'])
def show_map():
    waypoints = []
    mode = "driving"
    error_message = None

    if request.method == 'POST':
        waypoints_data = request.form.get('waypoints', '')
        mode = request.form.get('mode', 'driving')
        logger.debug(f"Received POST with waypoints: {waypoints_data}, mode: {mode}")
        try:
            waypoints_list = json.loads(waypoints_data)
            waypoints, error_message = validate_waypoints(waypoints_list)
            logger.debug(f"Validated waypoints: {waypoints}")
        except json.JSONDecodeError:
            logger.error("Failed to parse waypoints JSON")
            error_message = "Invalid waypoint data format"

    center_lat = 51.5074 if not waypoints else sum(wp[0] for wp in waypoints) / len(waypoints)
    center_lng = -0.1278 if not waypoints else sum(wp[1] for wp in waypoints) / len(waypoints)
    m = folium.Map(location=[center_lat, center_lng], zoom_start=14, tiles=None)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='TransIT',
        name='OSM'
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr='TransIT',
        name='Topo'
    ).add_to(m)

    waypoint_icons = {
        'start': folium.CustomIcon(
            icon_image='https://www.transit.com.eg/assets/LOGO.svg',
            icon_size=[36, 36],
            icon_anchor=[18, 36],
            popup_anchor=[0, -36]
        ),
        'end': folium.CustomIcon(
            icon_image='https://www.transit.com.eg/assets/LOGO.svg',
            icon_size=[28, 28],
            icon_anchor=[14, 28],
            popup_anchor=[0, -28]
        )
    }

    if waypoints:
        for i, wp in enumerate(waypoints):
            icon = waypoint_icons['start'] if i == 0 else waypoint_icons['end']
            name = f"{'Start' if i == 0 else 'End'} ({wp[0]:.6f}, {wp[1]:.6f})"
            popup_content = f"<b>{'START' if i == 0 else 'END'}</b><br>{wp[0]:.6f}, {wp[1]:.6f}"
            Marker(
                wp,
                popup=folium.Popup(popup_content, max_width=200),
                tooltip=name,
                icon=icon
            ).add_to(m)

    routes, bounds = load_or_fetch_routes(waypoints, mode=mode) if len(waypoints) == 2 else ([], None)
    route_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    route_details = []
    route_layer_ids = {}

    if bounds:
        min_lat, min_lon, max_lat, max_lon = bounds
        folium.Rectangle(
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            color='blue',
            weight=2,
            opacity=0,
            fill=True,
            fill_opacity=0,
            tooltip="Route Bounding Box"
        ).add_to(m)

    if routes:
        for i, route_data in enumerate(routes):
            coords = route_data.get("coords", [])
            dist = route_data.get("distance", 0)
            dur = route_data.get("duration", 0)
            steps = route_data.get("steps", [])
            traffic = route_data.get("traffic", [])
            color = route_colors[i % len(route_colors)]
            layer_name = f"Route {i+1}{' (Best)' if i == 0 else ''}"
            layer = folium.FeatureGroup(name=layer_name)
            
            # Calculate average traffic efficiency
            avg_traffic = sum([eff for _, _, eff in traffic]) / len(traffic) if traffic else 0.5
            now = datetime.now()
            arrival = now + timedelta(minutes=dur)
            
            # Build detailed tooltip
            tooltip_html = f"""
                <b>{layer_name}</b><br>
                Distance: {dist:.1f} km<br>
                Duration: {dur:.1f} min<br>
                ETA: {arrival.strftime('%H:%M')}<br>
                Avg. Traffic: {int(avg_traffic * 100)}% ({'Good' if avg_traffic >= 0.7 else 'Moderate' if avg_traffic >= 0.4 else 'Heavy'})<br>
                Steps:<ul style='margin: 5px 0 0 15px; padding: 0;'>{''.join([f'<li>{step["maneuver"]["instruction"]}</li>' for step in steps[:10] if "maneuver" in step and "instruction" in step["maneuver"]])}</ul>
            """
            
            if traffic:
                for start_idx, end_idx, efficiency in traffic:
                    segment_coords = coords[start_idx:end_idx]
                    if len(segment_coords) > 1:
                        folium.PolyLine(
                            segment_coords,
                            color=traffic_color(efficiency),
                            weight=10 if i == 0 else 6,
                            opacity=1 if i == 0 else 0.7,
                            dash_array=None if i == 0 else '5, 10',
                            tooltip=folium.Tooltip(tooltip_html, sticky=True),
                            popup=folium.Popup(tooltip_html, max_width=300)
                        ).add_to(layer)
            else:
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=10 if i == 0 else 6,
                    opacity=1 if i == 0 else 0.7,
                    dash_array=None if i == 0 else '5, 10',
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    popup=folium.Popup(tooltip_html, max_width=300)
                ).add_to(layer)
            layer.add_to(m)  # All layers are added to map by default
            route_layer_ids[f"route{i+1}"] = layer_name
            route_steps = [f"<li>{step['maneuver']['instruction']}</li>" for step in steps if 'maneuver' in step and 'instruction' in step['maneuver']]
            route_details.append({
                "id": f"route{i+1}",
                "title": layer_name,
                "color": color,
                "distance": f"{dist:.1f} km",
                "duration": f"{dur:.1f} min",
                "arrival": arrival.strftime("%H:%M"),
                "steps": "".join(route_steps[:10]),
                "traffic": avg_traffic
            })

        # Fit map to show all routes
        if routes:
            all_coords = [coord for route in routes for coord in route.get("coords", [])]
            if all_coords:
                bounds = [
                    [min(c[0] for c in all_coords), min(c[1] for c in all_coords)],
                    [max(c[0] for c in all_coords), max(c[1] for c in all_coords)]
                ]
                m.fit_bounds(bounds)
    else:
        route_details.append({"message": "Add exactly 2 waypoints to see routes"})

    Fullscreen().add_to(m)
    LocateControl(options={'locateOptions': {'enableHighAccuracy': True}}).add_to(m)
    MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    waypoints_json = json.dumps([{"lat": wp[0], "lng": wp[1]} for wp in waypoints])

    custom_html = f"""
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .control-panel {{
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 1000;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            width: 330px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .control-panel h3 {{
            margin: 0 0 15px 0;
            color: #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .input-group {{
            margin-bottom: 15px;
        }}
        .input-group label {{
            display: block;
            margin-bottom: 5px;
            color: #666;
            font-weight: 500;
        }}
        .input-group input {{
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }}
        .waypoint-list {{
            margin-bottom: 20px;
            max-height: 300px;
            overflow-y: auto;
        }}
        .waypoint-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px;
            margin-bottom: 8px;
            background: #f9f9f9;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        .waypoint-item.start {{ border-color: #2ca02c; }}
        .waypoint-item.end {{ border-color: #d62728; }}
        .waypoint-remove {{
            background: none;
            border: none;
            cursor: pointer;
            color: #666;
            font-size: 1rem;
            padding: 2px 5px;
        }}
        .waypoint-remove:hover {{
            color: #000;
        }}
        .route-selector {{
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            width: 300px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .route-item {{
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            border-left: 4px solid;
            cursor: pointer;
            transition: all 0.3s;
            background: #f9f9f9;
        }}
        .route-item:hover {{
            background: #f0f0f0;
        }}
        .route-item.active {{
            background: #e8f0fe;
        }}
        .route-item h4 {{
            margin: 0 0 8px 0;
        }}
        .route-meta {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        .route-meta span {{
            font-size: 0.85rem;
            color: #666;
        }}
        .arrival-time {{
            font-weight: bold;
            color: #333;
        }}
        .traffic-legend {{
            display: flex;
            align-items: center;
            gap: 5px;
            margin-top: 5px;
            font-size: 0.8rem;
        }}
        .traffic-color {{
            width: 25px;
            height: 6px;
            border-radius: 3px;
            display: inline-block;
        }}
        button {{
            width: 100%;
            padding: 10px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.3s;
            font-weight: 500;
        }}
        .button-secondary {{
            background: #6c757d;
        }}
        .button-secondary:hover {{
            background: #5a6268;
        }}
        .button-group {{
            display: flex;
            gap: 10px;
        }}
        .button-group button {{
            flex: 1;
        }}
        .minimize-btn {{
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1.2rem;
            color: #666;
            padding: 0;
            width: auto;
        }}
        .minimized {{
            width: auto;
            height: auto;
            padding: 10px;
        }}
        .minimized .panel-content,
        .minimized .panel-title {{
            display: none;
        }}
        .instructions {{
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 15px;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .travel-mode-selector {{
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
        }}
        .travel-mode-selector button {{
            flex: 1;
            padding: 8px;
            font-size: 0.85rem;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .travel-mode-selector button.active {{
            background: #4CAF50;
        }}
        .coordinates-display {{
            position: fixed;
            bottom: 10px;
            right: 10px;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.9);
            padding: 8px 12px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 0.9rem;
            color: #333;
            border: 1px solid #ddd;
        }}
    </style>
    <div class="control-panel" id="controlPanel">
        <h3>
            <span class="panel-title">Route Planner</span>
            <button class="minimize-btn" onclick="togglePanel('controlPanel')">▢</button>
        </h3>
        <div class="panel-content">
            <div class="instructions">
                Click on the map to add start and end points (exactly 2 required). Hover over routes for details.
                {'<div style="color: red; margin-top: 5px">' + error_message + '</div>' if error_message else ''}
            </div>
            <div class="travel-mode-selector">
                <button class="mode-btn {'active' if mode == 'driving' else ''}" data-mode="driving">🚗 Driving</button>
                <button class="mode-btn {'active' if mode == 'walking' else ''}" data-mode="walking">🚶 Walking</button>
                <button class="mode-btn {'active' if mode == 'cycling' else ''}" data-mode="cycling">🚴 Cycling</button>
            </div>
            <div class="waypoint-list" id="waypointList">
            </div>
            <div class="input-group">
                <label>New Waypoint</label>
                <div style="display: flex; gap: 5px;">
                    <input type="number" step="any" id="new_lat" placeholder="Latitude" required>
                    <input type="number" step="any" id="new_lon" placeholder="Longitude" required>
                </div>
            </div>
            <div class="button-group">
                <button type="button" onclick="addWaypoint()">Add Waypoint</button>
                <button type="button" class="button-secondary" onclick="clearWaypoints()">Clear All</button>
            </div>
            <div style="margin-top: 15px;">
                <button type="button" onclick="generateRoutes()">Generate Routes</button>
            </div>
        </div>
    </div>
    <div class="route-selector" id="routeSelector">
        <h3>
            <span class="panel-title">Available Routes ({len(routes)})</span>
            <button class="minimize-btn" onclick="togglePanel('routeSelector')">▢</button>
        </h3>
        <div class="panel-content">
            {''.join([f'''
            <div class="route-item" id="{r["id"]}" style="border-left-color: {r["color"]}" 
                onclick="highlightRoute('{r["id"]}')">
                <h4>{r["title"]}</h4>
                <div class="route-meta">
                    <span>{r["distance"]}</span>
                    <span>{r["duration"]}</span>
                </div>
                <p class="arrival-time">ETA: {r["arrival"]}</p>
                <p>Traffic: {int(r["traffic"] * 100)}% ({'Good' if r["traffic"] >= 0.7 else 'Moderate' if r["traffic"] >= 0.4 else 'Heavy'})</p>
                <ul style="font-size: 0.85em; margin: 10px 0 0 10px; color: #555;">{r["steps"]}</ul>
                <div class="traffic-legend">
                    Traffic: 
                    <span class="traffic-color" style="background: #4CAF50"></span>Good
                    <span class="traffic-color" style="background: #FFC107"></span>Moderate
                    <span class="traffic-color" style="background: #F44336"></span>Heavy
                </div>
            </div>
            ''' for r in route_details if "title" in r]) or '<p>Add exactly 2 waypoints to see routes</p>'}
        </div>
    </div>
    <div class="coordinates-display" id="coordinatesDisplay">
        Lat: 0.000000, Lng: 0.000000
    </div>
    <script>
        let waypoints = {waypoints_json};
        let activeRoute = null;
        const routeLayers = {json.dumps(route_layer_ids)};
        
        map.on('click', function(e) {{
            if (waypoints.length >= 2) {{
                alert('Only 2 waypoints (start and end) are allowed. Clear existing waypoints to add new ones.');
                return;
            }}
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;
            waypoints.push({{ lat: lat, lng: lng }});
            updateWaypointsList();
            generateRoutes();
        }});
        
        map.on('mousemove', function(e) {{
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;
            document.getElementById('coordinatesDisplay').innerText = 
                `Lat: ${{lat.toFixed(6)}}, Lng: ${{lng.toFixed(6)}}`;
        }});
        
        function updateWaypointsList() {{
            const list = document.getElementById('waypointList');
            list.innerHTML = waypoints.length === 0 ? 
                '<div class="click-instructions">Click on the map to add start and end points</div>' : '';
            waypoints.forEach((wp, index) => {{
                let itemClass = index === 0 ? 'start' : 'end';
                const item = document.createElement('div');
                item.className = `waypoint-item ${{itemClass}}`;
                item.innerHTML = `
                    <div>
                        ${{index === 0 ? 'Start' : 'End'}}
                        <div style="font-size: 0.8em; color: #777;">
                            ${{wp.lat.toFixed(6)}}, ${{wp.lng.toFixed(6)}}
                        </div>
                    </div>
                    <div class="waypoint-actions">
                        <button class="waypoint-remove" onclick="removeWaypoint(${{index}})">×</button>
                    </div>
                `;
                list.appendChild(item);
            }});
        }}
        
        function addWaypoint() {{
            if (waypoints.length >= 2) {{
                alert('Only 2 waypoints (start and end) are allowed. Clear existing waypoints to add new ones.');
                return;
            }}
            const lat = parseFloat(document.getElementById('new_lat').value);
            const lng = parseFloat(document.getElementById('new_lon').value);
            if (isNaN(lat) || isNaN(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {{
                alert('Please enter valid coordinates');
                return;
            }}
            waypoints.push({{ lat: lat, lng: lng }});
            document.getElementById('new_lat').value = '';
            document.getElementById('new_lon').value = '';
            updateWaypointsList();
            generateRoutes();
        }}
        
        function removeWaypoint(index) {{
            waypoints.splice(index, 1);
            updateWaypointsList();
            generateRoutes();
        }}
        
        function clearWaypoints() {{
            waypoints = [];
            updateWaypointsList();
            generateRoutes();
        }}
        
        function generateRoutes() {{
            document.getElementById('routeSelector').innerHTML = '<h3>Calculating Routes...</h3>';
            const form = document.createElement('form');
            form.method = 'POST';
            form.action = '/';
            const waypointsInput = document.createElement('input');
            waypointsInput.type = 'hidden';
            waypointsInput.name = 'waypoints';
            waypointsInput.value = JSON.stringify(waypoints);
            const modeInput = document.createElement('input');
            modeInput.type = 'hidden';
            modeInput.name = 'mode';
            modeInput.value = document.querySelector('.mode-btn.active').dataset.mode;
            form.appendChild(waypointsInput);
            form.appendChild(modeInput);
            document.body.appendChild(form);
            form.submit();
        }}
        
        function highlightRoute(routeId) {{
            if (activeRoute) {{
                document.getElementById(activeRoute).classList.remove('active');
                map.eachLayer(function(layer) {{
                    if (layer.options && layer.options.name === routeLayers[activeRoute]) {{
                        layer.setStyle({{ weight: activeRoute === 'route1' ? 10 : 6, opacity: activeRoute === 'route1' ? 1 : 0.7 }});
                    }}
                }});
            }}
            activeRoute = routeId;
            document.getElementById(routeId).classList.add('active');
            map.eachLayer(function(layer) {{
                if (layer.options && layer.options.name === routeLayers[routeId]) {{
                    layer.setStyle({{ weight: 12, opacity: 1 }});
                    if (layer.getBounds) {{
                        map.fitBounds(layer.getBounds());
                    }}
                }}
            }});
        }}
        
        function togglePanel(panelId) {{
            const panel = document.getElementById(panelId);
            panel.classList.toggle('minimized');
            const btn = panel.querySelector('.minimize-btn');
            btn.textContent = panel.classList.contains('minimized') ? '◱' : '▢';
        }}
        
        document.querySelectorAll('.mode-btn').forEach(btn => {{
            btn.addEventListener('click', function() {{
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                if (waypoints.length === 2) generateRoutes();
            }});
        }});
        
        updateWaypointsList();
    </script>
    """

    m.get_root().html.add_child(folium.Element(custom_html))
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Route Planner</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
        <link rel="icon" href="/static/favicon.ico">
        {{ map_html|safe }}
    </head>
    <body>
        <div id="map" style="width:100vw; height:100vh;"></div>
    </body>
    </html>
    ''', map_html=m._repr_html_())

@app.route('/favicon.ico')
def favicon():
    favicon_path = os.path.join(app.root_path, 'static', 'favicon.ico')
    if os.path.exists(favicon_path):
        return send_from_directory(os.path.join(app.root_path, 'static'),
                                 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    else:
        logger.warning("Favicon not found at static/favicon.ico. Returning 204 to suppress 404 error.")
        return '', 204

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, host='0.0.0.0', port=5000)