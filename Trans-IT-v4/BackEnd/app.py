import os
import json
import math
import random
import logging
import hashlib
import requests
import polyline
from datetime import datetime, timedelta
from functools import lru_cache
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from dotenv import load_dotenv
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Custom filter to add request_id to log records
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        try:
            from flask import request
            record.request_id = getattr(request, 'request_id', 'none')
        except (RuntimeError, ImportError, AttributeError):
            record.request_id = 'none'
        return True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(request_id)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('route_planner.log'),
        logging.StreamHandler()
    ]
)

root_logger = logging.getLogger()
root_logger.addFilter(RequestIdFilter())

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.INFO)
werkzeug_logger.handlers = []
werkzeug_logger.propagate = True

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
load_dotenv()

logger = logging.getLogger(__name__)

@app.before_request
def add_request_id():
    request.request_id = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Configuration
MAPS_API_KEY = os.getenv('MAPS_API_KEY')
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_BASE_URL = os.getenv('OSRM_BASE_URL', "http://router.project-osrm.org/route/v1")
CACHE_DIR = 'route_cache'
RESTRICTED_AREAS = json.loads(os.getenv('RESTRICTED_AREAS', '[[51.5100, -0.1250, 51.5120, -0.1230]]'))
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TYPE = 'file'
REQUEST_TIMEOUT = 15
NOMINATIM_RATE_LIMIT = 1
SUPPORTED_MODES = {'driving', 'walking', 'cycling'}
TRAFFIC_COLORS = {
    'good': '#2ecc71',
    'moderate': '#f39c12',
    'heavy': '#e74c3c'
}
MAX_WAYPOINTS = 2

@lru_cache(maxsize=256)
def geocode_address(address):
    cache_key = f"geocode:{hashlib.md5(address.encode()).hexdigest()}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

        headers = {'User-Agent': 'AdvancedRoutePlanner/1.0 (contact: your-email@example.com)'}
        params = {'format': 'json', 'q': address, 'limit': 1}
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            logger.warning(f"No geocoding results for '{address}'")
            return None

        result = {
            'lat': float(data[0]['lat']),
            'lng': float(data[0]['lon']),
            'display_name': data[0]['display_name']
        }

        with open(cache_file, 'w') as f:
            json.dump(result, f)

        time.sleep(NOMINATIM_RATE_LIMIT)
        return result
    except requests.RequestException as e:
        logger.error(f"Geocoding error for '{address}': {e}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Corrupted cache for '{address}'")
        return None
    except Exception as e:
        logger.error(f"Unexpected geocoding error: {e}")
        return None

def validate_coordinates(lat, lon):
    try:
        lat, lon = float(lat), float(lon)
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
        logger.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
        return None
    except (ValueError, TypeError):
        logger.warning(f"Invalid coordinate types: lat={lat}, lon={lon}")
        return None

def validate_waypoints(waypoints_list):
    validated = []
    error_msg = None

    if not isinstance(waypoints_list, list):
        return [], "Waypoints must be a list"

    if len(waypoints_list) != MAX_WAYPOINTS:
        return [], f"Exactly {MAX_WAYPOINTS} waypoints required (received {len(waypoints_list)})"

    for i, wp in enumerate(waypoints_list):
        if not isinstance(wp, dict):
            return [], f"Waypoint {i} must be a dictionary"

        if 'lat' in wp and 'lng' in wp:
            coords = validate_coordinates(wp['lat'], wp['lng'])
            if not coords:
                return [], f"Invalid coordinates in waypoint {i}: lat={wp.get('lat')}, lng={wp.get('lng')}"
            validated.append({'lat': coords[0], 'lng': coords[1], 'name': wp.get('name', '')})
        elif 'address' in wp:
            result = geocode_address(wp['address'])
            if not result:
                return [], f"Could not geocode address in waypoint {i}: {wp.get('address')}"
            validated.append({
                'lat': result['lat'],
                'lng': result['lng'],
                'name': result['display_name']
            })
        else:
            return [], f"Waypoint {i} must have 'lat'/'lng' or 'address'"

    return validated, error_msg

def haversine_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def point_to_line_distance(point, line_start, line_end):
    point_lat, point_lon = point
    start_lat, start_lon = line_start
    end_lat, end_lon = line_end

    km_per_lat = 111.0
    km_per_lon = 111.0 * math.cos(math.radians((start_lat + end_lat) / 2))

    x1, y1 = start_lon * km_per_lon, start_lat * km_per_lat
    x2, y2 = end_lon * km_per_lon, end_lat * km_per_lat
    x0, y0 = point_lon * km_per_lon, point_lat * km_per_lat

    line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
    if line_length_sq == 0:
        return haversine_distance(point, line_start)

    t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / line_length_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    proj_lon = proj_x / km_per_lon
    proj_lat = proj_y / km_per_lat
    return haversine_distance(point, (proj_lat, proj_lon))

def get_route_osrm(coordinates, mode="driving", alternatives=3, preferences=None):
    """Fetch routes from OSRM with retries and detailed logging."""
    mode = mode.lower() if mode.lower() in SUPPORTED_MODES else "driving"
    base_url = f"{OSRM_BASE_URL}/{mode}/"
    coordinate_str = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
    params = {
        'overview': 'full',
        'geometries': 'polyline',
        'steps': 'true',
        'annotations': 'true',
        'alternatives': str(alternatives)
    }
    if preferences:
        exclude = []
        if preferences.get('avoid_tolls'):
            exclude.append('toll')
        if preferences.get('avoid_highways'):
            exclude.append('motorway')
        if preferences.get('avoid_ferries'):
            exclude.append('ferry')
        if exclude:
            params['exclude'] = ','.join(exclude)

    url = f"{base_url}{coordinate_str}"
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        logger.debug(f"Fetching {mode} route: {url} with params {params}")
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != "Ok":
            logger.error(f"OSRM error: {data.get('message', 'Unknown error')}")
            return None

        routes = []
        for route in data.get('routes', []):
            geometry = polyline.decode(route.get('geometry', '')) if route.get('geometry') else []
            distance = route.get('distance', 0) / 1000
            duration = route.get('duration', 0) / 60
            steps = route.get('legs', [{}])[0].get('steps', [])
            annotations = route.get('annotations', {})
            if geometry and distance > 0 and duration > 0:
                routes.append((geometry, distance, duration, steps, annotations))
                logger.debug(f"Valid route: distance={distance:.2f}km, duration={duration:.1f}min")
            else:
                logger.warning(f"Skipping invalid route: geometry={len(geometry)}, distance={distance}, duration={duration}")

        logger.info(f"Found {len(routes)} {mode} routes")
        return routes if routes else None

    except requests.ConnectionError as e:
        logger.error(f"OSRM connection failed: {e}")
        return None
    except requests.Timeout as e:
        logger.error(f"OSRM request timed out after {REQUEST_TIMEOUT}s: {e}")
        return None
    except requests.HTTPError as e:
        logger.error(f"OSRM HTTP error: {e}, response: {response.text[:200] if 'response' in locals() else 'N/A'}")
        return None
    except requests.RequestException as e:
        logger.error(f"OSRM request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected OSRM error: {e}")
        return None
    finally:
        session.close()

def estimate_traffic_density(route_data):
    try:
        route_coords, distance, duration, steps, annotations = route_data
        traffic_segments = []
        segment_length = max(1, len(route_coords) // 10)

        avg_speed = (distance / (duration / 60)) if duration > 0 else 30
        if annotations.get('speed'):
            speeds = annotations['speed']
            avg_speed = sum(speeds) / len(speeds) if speeds else avg_speed

        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        time_multiplier = 1.0
        if weekday < 5:
            if 7 <= hour < 9 or 16 <= hour < 18:
                time_multiplier = 0.7
            elif 22 <= hour or hour < 6:
                time_multiplier = 1.3
        else:
            if 10 <= hour < 18:
                time_multiplier = 0.9
            else:
                time_multiplier = 1.2

        base_efficiency = min(1.0, max(0.2, (avg_speed / 50) * time_multiplier))
        random.seed(hashlib.md5(str(route_coords[:5]).encode()).hexdigest())
        prev_efficiency = base_efficiency

        for i in range(0, len(route_coords), segment_length):
            segment_end = min(i + segment_length, len(route_coords))
            variation = random.uniform(-0.15, 0.15)
            inertia = 0.8
            efficiency = max(0.1, min(1.0, 
                inertia * prev_efficiency + (1 - inertia) * (base_efficiency + variation)))
            traffic_segments.append((i, segment_end, efficiency))
            prev_efficiency = efficiency

        return traffic_segments
    except Exception as e:
        logger.error(f"Traffic estimation failed: {e}")
        return [(0, len(route_coords), 0.5)]

def traffic_color(efficiency):
    if efficiency >= 0.7:
        return TRAFFIC_COLORS['good']
    elif efficiency >= 0.4:
        return TRAFFIC_COLORS['moderate']
    return TRAFFIC_COLORS['heavy']

def is_route_restricted(route_coords):
    for lat, lon in route_coords:
        for area in RESTRICTED_AREAS:
            min_lat, min_lon, max_lat, max_lon = area
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                logger.debug(f"Route filtered out due to restricted area: {area}")
                return True
    return False

def is_route_within_bounds(route_coords, bounds, tolerance=0.9):
    min_lat, min_lon, max_lat, max_lon = bounds
    within_count = sum(1 for lat, lon in route_coords 
                       if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon)
    fraction_within = within_count / len(route_coords) if route_coords else 0
    logger.debug(f"Route within bounds fraction: {fraction_within}, required: {tolerance}")
    return fraction_within >= tolerance

def deduplicate_routes(routes, threshold=0.7, sample_points=10):
    unique_routes = []
    for route in routes:
        geometry = route[0]
        step = max(1, len(geometry) // sample_points)
        sampled = geometry[::step]
        is_unique = True
        for i, uroute in enumerate(unique_routes):
            ugeometry = uroute[0][::step]
            min_length = min(len(sampled), len(ugeometry))
            if min_length < 2:
                continue
            matches = sum(1 for j in range(min_length) 
                         if abs(sampled[j][0] - ugeometry[j][0]) < 0.002 and 
                            abs(sampled[j][1] - ugeometry[j][1]) < 0.002)
            if matches / min_length > threshold:
                is_unique = False
                if route[1] < uroute[1]:
                    unique_routes[i] = route
                break
        if is_unique:
            unique_routes.append(route)
    return unique_routes

def get_all_routes_osrm(waypoints, mode="driving", preferences=None):
    """Fetch and filter OSRM routes with relaxed constraints."""
    if len(waypoints) != MAX_WAYPOINTS:
        logger.warning(f"Invalid number of waypoints: {len(waypoints)}")
        return [], None

    start, end = waypoints
    distance_km = haversine_distance(start, end)

    if distance_km < 0.2:
        direct_line = [start, end]
        routes = get_route_osrm([start, end], mode=mode, alternatives=1, preferences=preferences)
        bounds = [
            min(start[0], end[0]) - 0.01,
            min(start[1], end[1]) - 0.01,
            max(start[0], end[0]) + 0.01,
            max(start[1], end[1]) + 0.01
        ]
        return routes or [], bounds

    avg_lat = (start[0] + end[0]) / 2
    buffer_percentage = 0.25 if distance_km < 5 else (0.20 if distance_km < 20 else 0.15)  # Relaxed
    buffer_km = max(2.0, min(10, distance_km * buffer_percentage))  # Wider buffer
    buffer_lat = buffer_km / 111.0
    buffer_lon = buffer_km / (111.0 * math.cos(math.radians(avg_lat)))
    bounds = [
        min(start[0], end[0]) - buffer_lat,
        min(start[1], end[1]) - buffer_lon,
        max(start[0], end[0]) + buffer_lat,
        max(start[1], end[1]) + buffer_lon
    ]
    tolerance = 0.6 if distance_km < 3 else (0.5 if distance_km < 10 else 0.4)  # Relaxed tolerance

    routes = []
    direct_routes = get_route_osrm([start, end], mode=mode, alternatives=3, preferences=preferences)
    if direct_routes:
        routes.extend(direct_routes)
        logger.debug(f"Direct routes found: {len(direct_routes)}")
    else:
        logger.warning("No direct routes found")

    direction_lat = end[0] - start[0]
    direction_lon = end[1] - start[1]
    magnitude = math.sqrt(direction_lat**2 + direction_lon**2)
    if magnitude > 0:
        unit_lat = direction_lat / magnitude
        unit_lon = direction_lon / magnitude
        perp_lat, perp_lon = -unit_lon, unit_lat
        offset_scale = min(buffer_lat * 0.75, 0.1)  # Increased offset
        mid_points = [
            (start[0] + direction_lat * i/3, start[1] + direction_lon * i/3)
            for i in range(1, 3)
        ]
        offsets = [
            (mp[0] + perp_lat * offset_scale, mp[1] + perp_lon * offset_scale)
            for mp in mid_points
        ] + [
            (mp[0] - perp_lat * offset_scale, mp[1] - perp_lon * offset_scale)
            for mp in mid_points
        ]

        for alt_wp in offsets:
            if bounds[0] <= alt_wp[0] <= bounds[2] and bounds[1] <= alt_wp[1] <= bounds[3]:
                alt_routes = get_route_osrm(
                    [start, alt_wp, end],
                    mode=mode,
                    alternatives=1,
                    preferences=preferences
                )
                if alt_routes:
                    routes.extend(alt_routes)
                    logger.debug(f"Alternative routes found for waypoint {alt_wp}: {len(alt_routes)}")

    unique_routes = deduplicate_routes(routes)
    logger.debug(f"Unique routes after deduplication: {len(unique_routes)}")
    max_deviation = max(5.0, distance_km * buffer_percentage * 6.0)  # Relaxed deviation
    valid_routes = []
    for r in unique_routes:
        route_coords = r[0]
        is_restricted = is_route_restricted(route_coords)
        within_bounds = is_route_within_bounds(route_coords, bounds, tolerance)
        max_dev = max([point_to_line_distance(p, start, end) for p in route_coords[::max(1, len(route_coords)//20)]]) if route_coords else float('inf')
        deviation_ok = max_dev <= max_deviation

        logger.debug(f"Route filter: restricted={is_restricted}, within_bounds={within_bounds}, max_dev={max_dev:.2f}/{max_deviation:.2f}")

        if not is_restricted and within_bounds and deviation_ok:
            valid_routes.append(r)

    valid_routes.sort(key=lambda x: (x[2], x[1]))
    valid_routes = valid_routes[:3]

    # Fallback: Return a straight-line route if no valid routes
    if not valid_routes:
        logger.warning("No valid routes after filtering; returning straight-line fallback")
        geometry = [start, end]
        distance = distance_km
        duration = distance / 50 * 60  # Assume 50 km/h average speed
        valid_routes = [(geometry, distance, duration, [], {})]

    return valid_routes, bounds

def load_or_fetch_routes(waypoints, mode="driving", preferences=None):
    """Fetch routes, using cache if available, invalidate on failure."""
    waypoint_str = "-".join([f"{lat:.6f},{lon:.6f}" for lat, lon in waypoints])
    pref_str = json.dumps(preferences, sort_keys=True) if preferences else ""
    cache_key = f"route:{hashlib.md5(f'{waypoint_str}_{mode}_{pref_str}'.encode()).hexdigest()}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    now = datetime.now()

    try:
        if os.path.exists(cache_file):
            file_age = now.timestamp() - os.path.getmtime(cache_file)
            max_age = 3600
            if file_age < max_age:
                with open(cache_file, 'r') as f:
                    cached_routes = json.load(f)
                    if cached_routes:  # Only return cache if non-empty
                        logger.debug(f"Loaded {len(cached_routes)} routes from cache")
                        return cached_routes, None
                    logger.debug("Empty cache detected; fetching new routes")
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Cache read error for {cache_key}: {e}")

    routes, bounds = get_all_routes_osrm(waypoints, mode, preferences)
    serialized_routes = []
    for i, (coords, distance, duration, steps, annotations) in enumerate(routes):
        traffic_segments = estimate_traffic_density((coords, distance, duration, steps, annotations))
        serialized_routes.append({
            "coords": coords,
            "distance": distance,
            "duration": duration,
            "steps": steps[:50],
            "annotations": {
                k: v[::max(1, len(v)//100)] if isinstance(v, list) else v
                for k, v in annotations.items()
            },
            "traffic": traffic_segments,
            "timestamp": now.isoformat()
        })

    # Only cache non-empty results
    if serialized_routes:
        try:
            with open(cache_file, 'w') as f:
                json.dump(serialized_routes, f)
            logger.debug(f"Cached {len(serialized_routes)} routes")
        except Exception as e:
            logger.warning(f"Cache write error for {cache_key}: {e}")
    else:
        # Remove stale cache
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                logger.debug(f"Removed empty cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")

    return serialized_routes, bounds

@app.route('/api/routes', methods=['POST'])
@limiter.limit("10 per minute")
def get_routes():
    """Generate routes between waypoints."""
    data = request.get_json()
    if not data or 'waypoints' not in data:
        logger.warning("Missing waypoints in request")
        return jsonify({"error": "Missing waypoints"}), 400

    waypoints, error_msg = validate_waypoints(data['waypoints'])
    if error_msg:
        logger.warning(f"Validation error: {error_msg}")
        return jsonify({"error": error_msg}), 400

    mode = data.get('mode', 'driving').lower()
    if mode not in SUPPORTED_MODES:
        logger.warning(f"Invalid mode: {mode}, defaulting to driving")
        mode = 'driving'

    preferences = data.get('preferences', {})
    departure_time = data.get('departureTime', 'now')

    routes, bounds = load_or_fetch_routes(
        [(wp['lat'], wp['lng']) for wp in waypoints],
        mode,
        preferences
    )

    if not routes:
        logger.error("No routes generated, even with fallback")
        return jsonify({"error": "No routes found", "details": "Failed to generate routes, even with fallback"}), 500

    now = datetime.now()
    if departure_time != 'now':
        try:
            departure_dt = datetime.fromisoformat(departure_time.replace('Z', '+00:00'))
        except ValueError:
            logger.warning(f"Invalid departure time format: {departure_time}")
            departure_dt = now
    else:
        departure_dt = now

    response = {
        "routes": [
            {
                "id": f"route{i+1}",
                "title": f"Route {i+1}{' (Fastest)' if i == 0 else ''}",
                "coordinates": [[lat, lng] for lat, lng in route["coords"]],
                "distance_km": round(route["distance"], 2),
                "duration_min": round(route["duration"], 1),
                "eta": (departure_dt + timedelta(minutes=route["duration"])).strftime('%H:%M'),
                "steps": [
                    {"instruction": step.get("maneuver", {}).get("instruction", "Unknown step")}
                    for step in route["steps"]
                    if step.get("maneuver", {}).get("instruction")
                ][:10],
                "traffic_segments": [
                    {"start_idx": s, "end_idx": e, "efficiency": round(eff, 2)}
                    for s, e, eff in route["traffic"]
                ],
                "traffic_colors": [
                    {"start_idx": s, "end_idx": e, "color": traffic_color(eff)}
                    for s, e, eff in route["traffic"]
                ]
            }
            for i, route in enumerate(routes)
        ],
        "bounds": bounds
    }
    logger.info(f"Generated {len(routes)} routes for waypoints: {waypoints}")
    return jsonify(response), 200

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if not path or path == "":
        return render_template('index.html')
    try:
        return send_from_directory(app.static_folder, path)
    except FileNotFoundError:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)