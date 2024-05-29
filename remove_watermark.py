from src.removal_attack.fine_tuning import fine_tuning_attack
from src.removal_attack.overwriting import overwriting_attack
from src.removal_attack.pruning import pruning_attack

fine_tuning_attack("VGG", 32*32, 10, 100, 1e-3, "80_5_FHE_1716902969.942029")

overwriting_attack("VGG", 32*32, 10, 100, 1e-3, "80_5_FHE_1716902969.942029")

# pruning_attack(["80_5_FHE_1716902969.942029", "80_5_FHE_1716904563.9590387",
#                 "80_5_FHE_1716906179.9192486", "80_5_FHE_1716907845.7598705",
#                 "80_5_FHE_1716909447.061131"])