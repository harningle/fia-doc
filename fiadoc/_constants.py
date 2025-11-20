# -*- coding: utf-8 -*-
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
    }
}


if __name__ == '__name__':
    pass
