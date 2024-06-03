from src.removal_attack.fine_tuning import fine_tuning_attack
from src.removal_attack.overwriting import overwriting_attack
from src.removal_attack.pruning import pruning_attack

# fine_tuning_attack("VGG", 32*32, 10, 100, 1e-3, "80_5_FHE_1716902969.942029")
#
# overwriting_attack("VGG", 32*32, 10, 100, 1e-3, "80_5_FHE_1716902969.942029")

# pruning_attack(["80_5_FHE_1716902969.942029", "80_5_FHE_1716904563.9590387",
#                 "80_5_FHE_1716906179.9192486", "80_5_FHE_1716907845.7598705",
#                 "80_5_FHE_1716909447.061131"])


# overwriting_attack("ConvMixer", 32*32, 10, 100, 1e-3, "80_5_FHE_1717085618.2459195")


# fine_tuning_attack("ResNet", 32*32, 10, 100, 1e-4, "80_5_FHE_1717167491.7001247")

# overwriting_attack("ResNet", 32*32, 10, 100, 1e-4, "80_5_FHE_1717167491.7001247")