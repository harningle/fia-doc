# -*- coding: utf-8 -*-
"""Manually map driver and team names to their ID in jolpica database"""

DRIVERS = {
    2023: {
        'Max Verstappen': 'max_verstappen',
        'Sergio Perez': 'perez',
        'Charles Leclerc': 'leclerc',
        'Carlos Sainz': 'sainz',
        'George Russell': 'russell',
        'Lewis Hamilton': 'hamilton',
        'Esteban Ocon': 'ocon',
        'Pierre Gasly': 'gasly',
        'Oscar Piastri': 'piastri',
        'Lando Norris': 'norris',
        'Valtteri Bottas': 'bottas',
        'Zhou Guanyu': 'zhou',
        'Lance Stroll': 'stroll',
        'Fernando Alonso': 'alonso',
        'Kevin Magnussen': 'kevin_magnussen',
        'Nico Hulkenberg': 'hulkenberg',
        'Daniel Ricciardo': 'ricciardo',
        'Yuki Tsunoda': 'tsunoda',
        'Alexander Albon': 'albon',
        'Logan Sargeant': 'sargeant',
        'Nyck de Vries': 'de_vries',
        'Liam Lawson': 'lawson'
    },
    2024: {
        'Max Verstappen': 'max_verstappen',
        'Sergio Perez': 'perez',
        'George Russell': 'russell',
        'Lewis Hamilton': 'hamilton',
        'Charles Leclerc': 'leclerc',
        'Carlos Sainz': 'sainz',
        'Oscar Piastri': 'piastri',
        'Lando Norris': 'norris',
        'Lance Stroll': 'stroll',
        'Fernando Alonso': 'alonso',
        'Esteban Ocon': 'ocon',
        'Pierre Gasly': 'gasly',
        'Alexander Albon': 'albon',
        'Logan Sargeant': 'sargeant',
        'Daniel Ricciardo': 'ricciardo',
        'Yuki Tsunoda': 'tsunoda',
        'Valtteri Bottas': 'bottas',
        'Zhou Guanyu': 'zhou',
        'Kevin Magnussen': 'kevin_magnussen',
        'Nico Hulkenberg': 'hulkenberg',
        'Oliver Bearman': 'bearman',
        'Franco Colapinto': 'colapinto',
        'Jack Doohan': 'doohan',
        'Patricio O’Ward': 'oward',         # TODO: inconsistent with jolpica
        'Felipe Drugovich': 'drugovich',    # TODO: inconsistent with jolpica
        'Robert Shwartzman': 'shwartzman',  # TODO: inconsistent with jolpica
        'Liam Lawson': 'lawson',
        'Andrea Kimi Antonelli': 'antonelli'
    },
    2025: {
        'Oscar Piastri': 'piastri',
        'Lando Norris': 'norris',
        'Charles Leclerc': 'leclerc',
        'Lewis Hamilton': 'hamilton',
        'Max Verstappen': 'max_verstappen',
        'Liam Lawson': 'lawson',
        'George Russell': 'russell',
        'Andrea Kimi Antonelli': 'antonelli',
        'Lance Stroll': 'stroll',
        'Fernando Alonso': 'alonso',
        'Pierre Gasly': 'gasly',
        'Jack Doohan': 'doohan',
        'Esteban Ocon': 'ocon',
        'Oliver Bearman': 'bearman',
        'Isack Hadjar': 'hadjar',
        'Yuki Tsunoda': 'tsunoda',
        'Alexander Albon': 'albon',
        'Carlos Sainz': 'sainz',
        'Nico Hulkenberg': 'hulkenberg',
        'Gabriel Bortoleto': 'borotoleto'
    }
}

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
    2025: {}
}


if __name__ == '__name__':
    pass
