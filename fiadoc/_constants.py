# -*- coding: utf-8 -*-
"""Constants and configs. that need manual updating"""
# Manually map driver and team names to their ID in jolpica database
TEAMS = {
    2023: {
        'Red Bull Racing Honda RBPT': 'red_bull',
        'Ferrari': 'ferrari',
        'Mercedes': 'mercedes',
        'Alpine Renault': 'alpine',
        'McLaren Mercedes': 'mclaren',
        'Alfa Romeo Ferrari': 'alfa',
        'Aston Martin Aramco Mercedes': 'aston_martin',
        'Haas Ferrari': 'haas',
        'AlphaTauri Honda RBPT': 'alphatauri',
        'Williams Mercedes': 'williams'
    },
    2024: {
        'Red Bull Racing Honda RBPT': 'red_bull',
        'Mercedes': 'mercedes',
        'Ferrari': 'ferrari',
        'McLaren Mercedes': 'mclaren',
        'Aston Martin Aramco Mercedes': 'aston_martin',
        'Alpine Renault': 'alpine',
        'Williams Mercedes': 'williams',
        'RB Honda RBPT': 'rb',
        'Kick Sauber Ferrari': 'sauber',
        'Haas Ferrari': 'haas'
    },
    2025: {
        'McLaren Mercedes': 'mclaren',
        'Ferrari': 'ferrari',
        'Red Bull Racing Honda RBPT': 'red_bull',
        'Mercedes': 'mercedes',
        'Aston Martin Aramco Mercedes': 'aston_martin',
        'Alpine Renault': 'alpine',
        'Haas Ferrari': 'haas',
        'Racing Bulls Honda RBPT': 'rb',
        'Williams Mercedes': 'williams',
        'Kick Sauber Ferrari': 'sauber'
    },
    2026: {
        'McLaren Mercedes': 'mclaren',
        'Mercedes': 'mercedes',
        'Red Bull Racing Red Bull Ford': 'red_bull',
        'Ferrari': 'ferrari',
        'Atlassian Williams Mercedes': 'williams',
        'Racing Bulls Red Bull Ford': 'rb',
        'Aston Martin Aramco Honda': 'aston_martin',
        'Haas Ferrari': 'haas',
        'Audi': 'audi',
        'Alpine Mercedes': 'alpine',
        'Cadillac Ferrari': 'cadillac'
    }
}

# How many drivers in each quali. session
QUALI_DRIVERS = {
    2023: {
        1: float('inf'),
        2: 15,
        3: 10
    },
    2024: {
        1: float('inf'),
        2: 15,
        3: 10
    },
    2025: {
        1: float('inf'),
        2: 15,
        3: 10
    },
    2026: {
        1: float('inf'),
        2: 16,
        3: 10
    }
}

# Regular drivers in each season
REGULAR_DRIVERS = {
    2023: {
        'max verstappen': 'max_verstappen',
        'sergio perez': 'perez',
        'charles leclerc': 'leclerc',
        'carlos sainz': 'sainz',
        'george russell': 'russell',
        'lewis hamilton': 'hamilton',
        'esteban ocon': 'ocon',
        'pierre gasly': 'gasly',
        'oscar piastri': 'piastri',
        'lando norris': 'norris',
        'valtteri bottas': 'bottas',
        'zhou guanyu': 'zhou',
        'lance stroll': 'stroll',
        'fernando alonso': 'alonso',
        'kevin magnussen': 'kevin_magnussen',
        'nico hulkenberg': 'hulkenberg',
        'daniel ricciardo': 'ricciardo',
        'yuki tsunoda': 'tsunoda',
        'alexander albon': 'albon',
        'logan sargeant': 'sargeant',
        'nyck de vries': 'de_vries',
        'liam lawson': 'lawson'
    },
    2024: {
        'max verstappen': 'max_verstappen',
        'sergio perez': 'perez',
        'george russell': 'russell',
        'lewis hamilton': 'hamilton',
        'charles leclerc': 'leclerc',
        'carlos sainz': 'sainz',
        'oscar piastri': 'piastri',
        'lando norris': 'norris',
        'lance stroll': 'stroll',
        'fernando alonso': 'alonso',
        'esteban ocon': 'ocon',
        'pierre gasly': 'gasly',
        'alexander albon': 'albon',
        'logan sargeant': 'sargeant',
        'daniel ricciardo': 'ricciardo',
        'yuki tsunoda': 'tsunoda',
        'valtteri bottas': 'bottas',
        'zhou guanyu': 'zhou',
        'kevin magnussen': 'kevin_magnussen',
        'nico hulkenberg': 'hulkenberg',
        'oliver bearman': 'bearman',
        'franco colapinto': 'colapinto',
        'jack doohan': 'doohan',
        'liam lawson': 'lawson',
        'andrea kimi antonelli': 'antonelli'
    },
    2025: {
        'oscar piastri': 'piastri',
        'lando norris': 'norris',
        'charles leclerc': 'leclerc',
        'lewis hamilton': 'hamilton',
        'max verstappen': 'max_verstappen',
        'liam lawson': 'lawson',
        'george russell': 'russell',
        'andrea kimi antonelli': 'antonelli',
        'kimi antonelli': 'antonelli',
        'lance stroll': 'stroll',
        'fernando alonso': 'alonso',
        'pierre gasly': 'gasly',
        'jack doohan': 'doohan',
        'esteban ocon': 'ocon',
        'oliver bearman': 'bearman',
        'isack hadjar': 'hadjar',
        'yuki tsunoda': 'tsunoda',
        'alexander albon': 'albon',
        'carlos sainz': 'sainz',
        'nico hulkenberg': 'hulkenberg',
        'gabriel bortoleto': 'bortoleto',
        'garbiel bortoleto': 'bortoleto',  # typo in entry list in 2025 australian
        'franco colapinto': 'colapinto'
    },
    2026: {
        'oscar piastri': 'piastri',
        'lando norris': 'norris',
        'george russell': 'russell',
        'andrea kimi antonelli': 'antonelli',
        'kimi antonelli': 'antonelli',
        'max verstappen': 'max_verstappen',
        'isack hadjar': 'hadjar',
        'charles leclerc': 'leclerc',
        'lewis hamilton': 'hamilton',
        'alexander albon': 'albon',
        'carlos sainz': 'sainz',
        'arvid lindblad': 'arvid_lindblad',
        'liam lawson': 'lawson',
        'lance stroll': 'stroll',
        'fernando alonso': 'alonso',
        'esteban ocon': 'ocon',
        'oliver bearman': 'bearman',
        'nico hulkenberg': 'hulkenberg',
        'gabriel bortoleto': 'bortoleto',
        'garbiel bortoleto': 'bortoleto',  # Typo in entry list in 2025 australian
        'pierre gasly': 'gasly',
        'franco colapinto': 'colapinto',
        'sergio perez': 'perez',
        'valtteri bottas': 'bottas'
    }
}


# Best DPI for PDF parsing
DPI = 600

# Expected cols. in the PDFs
"""
* required: must have cols. If any of these cols. are missing, then means the PDF/table
            layout changes, and thus we can't parse
* optional: we cover many years' PDFs, and some cols. are added while others are removed between
            years. These col. changes don't affect the parsing, so we flag them as optional
* to_parse: cols. we need to parse. Some required cols. can be skipped to save time, e.g. the
            country flag col. in classification PDFs. This is different from required cols., as
            we need all required cols. to make sure the table layout is what we expect, and among
            these required cols., we only need to parse a subset of them and skip the rest
"""
EXPECTED_COLS: dict[str, dict[str, set | list]] = {
    'entry_list': {
        'required': {'no.', 'constructor'},
        'optional': {'tla', 'driver', 'nat', 'team'}
    },
    'fp_classification': {
        'required': {'no', 'driver', 'nat', 'entrant', 'time', 'laps', 'gap', 'int', 'km/h',
                     'time of day'},
        'to_parse': {'no', 'time', 'laps', 'gap', 'int', 'km/h', 'time of day'}
    },
    'fp_lap_times': {
        'required': {'no', 'time'},
        'to_check_strikeout': {'time'}
    },
    'quali_lap_times': {
        'required': {'no', 'time'},
        'to_check_strikeout': {'time'}
    },
    'quali_classification': {
        'required': {'no', 'driver', 'nat', 'entrant', 'q1', 'q1_laps', 'q1_time', 'q2', 'q2_laps',
                     'q2_time', 'q3', 'q3_laps', 'q3_time'},
        'to_parse': {'no', 'q1', 'q1_laps', 'q1_time', 'q2', 'q2_laps', 'q2_time', 'q3', 'q3_laps',
                     'q3_time'}
    },
    'race_classification': {
        'required': {'no', 'driver', 'nat', 'entrant', 'laps', 'time', 'gap', 'int', 'km/h',
                     'fastest', 'on', 'pts'},
        'to_parse': {'no', 'laps', 'time', 'gap', 'fastest', 'on', 'pts'}
    },
    'race_lap_times': {
        'required': {'no', 'time'},
        'to_check_strikeout': {'time'}
    },
    'race_sector_analysis': {
        'required': ['lap', 'time', 'km/h', 'time', 'km/h', 'time', 'km/h', 'time'],
        'to_check_strikeout': {'time'}
    }
}


if __name__ == '__main__':
    pass
