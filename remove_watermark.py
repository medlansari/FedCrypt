from src.removal_attack.fine_tuning import fine_tuning_attack
from src.removal_attack.overwriting import overwriting_attack
from src.removal_attack.pruning import pruning_attack

# fine_tuning_attack("VGG", 32*32, 10, 100, 1e-3, "80_5_FHE_1716902969.942029")
#
overwriting_attack("VGG", 32*32, 10, 100, 1e-3, "80_5_FHE_1716902969.942029")

# pruning_attack(["80_5_FHE_1716902969.942029", "80_5_FHE_1716904563.9590387",
#                 "80_5_FHE_1716906179.9192486", "80_5_FHE_1716907845.7598705",
#                 "80_5_FHE_1716909447.061131"])


# overwriting_attack("ConvMixer", 32*32, 10, 100, 1e-3, "80_5_FHE_1717085618.2459195")


# fine_tuning_attack("ResNet", 32*32, 10, 100, 1e-4, "80_5_FHE_1717167491.7001247")

# overwriting_attack("ResNet", 32*32, 10, 100, 1e-4, "80_5_FHE_1717167491.7001247")

#pruning_attack(["ResNet_80_5_FHE_1717429028.870698", "ResNet_80_5_FHE_1717433158.0431964",
#                 "ResNet_80_5_FHE_1717437364.5714872", "ResNet_80_5_FHE_1717441652.54302",
#                 "ResNet_80_5_FHE_1717445921.7719965"])

# pruning_attack(["ConvMixer_80_5_FHE_1717443803.288811", "ConvMixer_80_5_FHE_1717431083.352223",
#                 "ConvMixer_80_5_FHE_1717448108.1316755", "ConvMixer_80_5_FHE_1717435277.642867",
#                 "ConvMixer_80_5_FHE_1717439538.2617962"])

