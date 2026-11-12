"""Babylon: climate tipping elements and their coupling structure.

Coupling strengths derived from:
- Wunderling et al. (2021) Earth System Dynamics — interacting tipping elements
- Armstrong McKay et al. (2022) Science — exceeding 1.5C triggers multiple tipping points
- Lenton et al. (2008, 2019) PNAS — tipping elements in Earth's climate system
- Kriegler et al. (2009) PNAS — expert elicitation of tipping point interactions

Couplings are directional in reality but symmetrized for the coupling analysis
(average of both directions). Values on 0-1 scale represent the strength of
influence one tipping element has on another's likelihood of tipping.
"""

ELEMENTS = {
    "GIS":  {"name": "Greenland Ice Sheet",     "threshold_C": 1.5, "category": "Cryosphere"},
    "WAIS": {"name": "West Antarctic Ice Sheet", "threshold_C": 1.5, "category": "Cryosphere"},
    "EAIS": {"name": "East Antarctic subglacial", "threshold_C": 3.0, "category": "Cryosphere"},
    "ASIC": {"name": "Arctic summer sea ice",   "threshold_C": 1.0, "category": "Cryosphere"},
    "ALGL": {"name": "Mountain glaciers",        "threshold_C": 1.5, "category": "Cryosphere"},
    "PFST": {"name": "Permafrost collapse",      "threshold_C": 1.5, "category": "Carbon cycle"},
    "AMOC": {"name": "Atlantic circulation",     "threshold_C": 1.5, "category": "Circulation"},
    "AMAZ": {"name": "Amazon rainforest",        "threshold_C": 2.0, "category": "Biosphere"},
    "BORF": {"name": "Boreal forest",            "threshold_C": 3.0, "category": "Biosphere"},
    "REEF": {"name": "Coral reef die-off",       "threshold_C": 1.0, "category": "Biosphere"},
    "ENSO": {"name": "El Nino intensification",  "threshold_C": 2.0, "category": "Circulation"},
    "ISM":  {"name": "Indian summer monsoon",    "threshold_C": 2.0, "category": "Circulation"},
    "WAM":  {"name": "West African monsoon",     "threshold_C": 2.5, "category": "Circulation"},
    "SAHA": {"name": "Sahel/Sahara greening",    "threshold_C": 2.5, "category": "Biosphere"},
    "TBET": {"name": "Tibetan plateau glaciers", "threshold_C": 1.5, "category": "Cryosphere"},
}

ELEMENT_NAMES = list(ELEMENTS.keys())

# Directional couplings: (source, target, strength)
# Strength is how much tipping of source increases likelihood of target tipping.
# Derived from physical mechanisms documented in the literature above.
DIRECTIONAL_COUPLINGS = [
    # Cryosphere internal
    ("GIS",  "AMOC", 0.80),  # massive freshwater input weakens AMOC
    ("GIS",  "WAIS", 0.30),  # sea level + ocean warming destabilizes WAIS
    ("GIS",  "ASIC", 0.30),  # freshwater/albedo changes
    ("WAIS", "AMOC", 0.30),  # freshwater input to Southern Ocean
    ("WAIS", "EAIS", 0.40),  # marine ice sheet instability propagates
    ("ASIC", "GIS",  0.70),  # Arctic amplification accelerates Greenland melt
    ("ASIC", "PFST", 0.80),  # Arctic warming is primary permafrost driver
    ("ASIC", "BORF", 0.40),  # Arctic warming stresses boreal forests
    ("ASIC", "TBET", 0.30),  # amplified high-latitude warming
    ("ALGL", "TBET", 0.20),  # regional warming/albedo

    # Carbon cycle feedbacks
    ("PFST", "ASIC", 0.60),  # methane release amplifies Arctic warming
    ("PFST", "BORF", 0.50),  # warming + soil changes stress boreal forest
    ("PFST", "GIS",  0.30),  # global warming contribution
    ("AMAZ", "PFST", 0.40),  # CO2 release → global warming → permafrost
    ("AMAZ", "AMOC", 0.30),  # CO2 release → warming → circulation changes
    ("BORF", "PFST", 0.40),  # fire CO2 + albedo → warming → permafrost
    ("BORF", "ASIC", 0.20),  # albedo/CO2 feedback to Arctic

    # Circulation interactions
    ("AMOC", "GIS",  0.40),  # mixed: shutdown cools N Atlantic but shifts heat
    ("AMOC", "AMAZ", 0.50),  # ITCZ shift reduces Amazon rainfall
    ("AMOC", "ISM",  0.50),  # disrupts monsoon dynamics
    ("AMOC", "WAM",  0.50),  # disrupts Sahel rainfall
    ("AMOC", "REEF", 0.30),  # ocean temperature redistribution

    # Tropical interactions
    ("ENSO", "AMAZ", 0.50),  # El Nino causes severe Amazon drought
    ("ENSO", "REEF", 0.70),  # El Nino causes mass coral bleaching
    ("ENSO", "ISM",  0.30),  # El Nino disrupts Indian monsoon
    ("AMAZ", "ENSO", 0.30),  # Amazon dieback affects tropical circulation
    ("ISM",  "AMAZ", 0.20),  # monsoon changes affect tropical rainfall
    ("ISM",  "TBET", 0.20),  # monsoon changes affect Tibetan glaciers
    ("TBET", "ISM",  0.30),  # glacier loss affects plateau heating → monsoon

    # Regional
    ("SAHA", "WAM",  0.30),  # Sahel vegetation affects West African monsoon
    ("WAM",  "SAHA", 0.40),  # monsoon rainfall drives Sahel greening
    ("REEF", "AMAZ", 0.10),  # marine ecosystem collapse — minor terrestrial effect
    ("GIS",  "REEF", 0.20),  # sea level + temperature
]


def build_coupling_matrix(temperature_C=1.5):
    """Build the coupling matrix at a given warming level.

    Coupling strengths scale with proximity to tipping thresholds:
    - Elements below their threshold: coupling at baseline
    - Elements near/above threshold: coupling increases (more sensitive)
    - The scaling represents that an element near its tipping point both
      responds more strongly to perturbation AND affects others more strongly
      when it starts tipping.
    """
    import numpy as np
    n = len(ELEMENT_NAMES)
    idx = {name: i for i, name in enumerate(ELEMENT_NAMES)}
    C = np.zeros((n, n))

    def _proximity_scale(element_name):
        thresh = ELEMENTS[element_name]["threshold_C"]
        if temperature_C >= thresh:
            return 1.0
        elif temperature_C >= thresh - 0.5:
            return 0.7 + 0.3 * (temperature_C - (thresh - 0.5)) / 0.5
        return 0.5

    for src, tgt, strength in DIRECTIONAL_COUPLINGS:
        si, ti = idx[src], idx[tgt]
        scale = np.sqrt(_proximity_scale(src) * _proximity_scale(tgt))
        C[si, ti] = strength * scale

    # Symmetrize: take the stronger direction (not mean — mean halves one-way couplings)
    C = np.maximum(C, C.T)
    np.fill_diagonal(C, 0.0)

    return C, ELEMENT_NAMES
