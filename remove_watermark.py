from src.removal_attack.fine_tuning import fine_tuning_attack
from src.removal_attack.overwriting import overwriting_attack

# fine_tuning_attack("VGG", 32*32, 10, 100, 1e-3, "80_5_FHE_1716550963.4475272")

overwriting_attack("VGG", 32*32, 10, 100, 1e-3, "80_5_FHE_1716550963.4475272")