"""
Single source of truth for per-machine focus range ("this laptop drives different
scope bodies with different focus ranges") — replaces the old scattered defaults in
run.py, config/init_focus_range.txt, and the General Acquisition panel.

Profiles are stored in config/focus_profiles.json as {name: {start_mm, end_mm,
search_range_mm}}, plus which one is "active" (used on next launch). Reads/writes
go through this module only; callers (SharedConfig, ui.py, run.py) never touch the
file directly.
"""

import json
import os

PROFILES_PATH = os.path.join('config', 'focus_profiles.json')

_DEFAULT_PROFILES = {
    "active_profile": "Octopi-demo-encl",
    "profiles": {
        "Octopi-demo-encl": {"start_mm": 6.25, "end_mm": 6.45, "search_range_mm": 0.1},
        "Octopi-PrakashLab": {"start_mm": 4.86, "end_mm": 5.06, "search_range_mm": 0.1},
    },
}


def _load_raw():
    if not os.path.exists(PROFILES_PATH):
        data = json.loads(json.dumps(_DEFAULT_PROFILES))
        _save_raw(data)
        return data
    try:
        with open(PROFILES_PATH, 'r') as f:
            data = json.load(f)
        if not data.get('profiles'):
            raise ValueError('no profiles in focus_profiles.json')
        return data
    except Exception as e:
        print(f"[focus_profiles] Failed to read {PROFILES_PATH} ({e}); restoring defaults")
        data = json.loads(json.dumps(_DEFAULT_PROFILES))
        _save_raw(data)
        return data


def _save_raw(data):
    os.makedirs(os.path.dirname(PROFILES_PATH), exist_ok=True)
    tmp_path = PROFILES_PATH + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, PROFILES_PATH)


def list_profiles():
    """{name: {start_mm, end_mm, search_range_mm}} for every saved profile."""
    return dict(_load_raw()['profiles'])


def get_profile(name):
    return _load_raw()['profiles'].get(name)


def get_active_profile():
    """(name, start_mm, end_mm, search_range_mm) for the profile marked active on disk."""
    data = _load_raw()
    profiles = data['profiles']
    name = data.get('active_profile')
    if name not in profiles:
        name = next(iter(profiles))
    p = profiles[name]
    return name, float(p['start_mm']), float(p['end_mm']), float(p.get('search_range_mm', 0.1))


def set_active_profile(name):
    """Persist `name` as the default profile to load on next launch."""
    data = _load_raw()
    if name not in data['profiles']:
        raise KeyError(f"Unknown focus profile: {name}")
    data['active_profile'] = name
    _save_raw(data)


def upsert_profile(name, start_mm, end_mm, search_range_mm=0.1, make_active=True):
    """Create or overwrite a named profile. make_active also persists it as the default."""
    name = (name or '').strip()
    if not name:
        raise ValueError("Profile name cannot be empty")
    data = _load_raw()
    data['profiles'][name] = {
        'start_mm': float(start_mm),
        'end_mm': float(end_mm),
        'search_range_mm': float(search_range_mm),
    }
    if make_active:
        data['active_profile'] = name
    _save_raw(data)
