"""Stimuli of the size and the length control."""

DOMINANCE_CONTROL_PROMPTS = [
    {"prompt": "Ein riesiger Elefant und ein winziger Vogel stehen auf einer Wiese.",
     "big": "Elefant", "small": "Vogel"},
    {"prompt": "Ein großer Lastwagen und ein kleines Fahrrad stehen auf der Straße.",
     "big": "Lastwagen", "small": "Fahrrad"},
    {"prompt": "Ein riesiges Haus und eine winzige Blume stehen auf einer Wiese.",
     "big": "Haus", "small": "Blume"},
    {"prompt": "Ein riesiger Baum und ein winziger Pilz stehen auf einer Wiese.",
     "big": "Baum", "small": "Pilz"},
    {"prompt": "Ein riesiger Bus und ein winziger Roller stehen auf der Straße.",
     "big": "Bus", "small": "Roller"},
    {"prompt": "Ein riesiges Schiff und eine winzige Ente sind auf dem Wasser.",
     "big": "Schiff", "small": "Ente"},
]
DOMINANCE_CONTROL_SEEDS = [0, 1000, 2000, 3000, 4000]

LENGTH_CONTROL_PAIRS = [
    {"short": "Eimer", "long": "Feuerlöscher", "ground": "auf dem Boden"},
    {"short": "Vase", "long": "Kaffeemaschine", "ground": "auf einem Tisch"},
    {"short": "Koffer", "long": "Werkzeugkasten", "ground": "auf dem Boden"},
]
LENGTH_CONTROL_ARTICLES = {"Eimer": "Ein", "Vase": "Eine", "Koffer": "Ein"}
LENGTH_CONTROL_FRAME = "{art_a} {a} und {art_b} {b} stehen nebeneinander {ground}."
LENGTH_CONTROL_ORDERS = ("short_first", "long_first")
LENGTH_CONTROL_SEEDS = [0, 1000, 2000, 3000, 4000]  # 3 pairs x 2 orders x 5 seeds = 30 images
