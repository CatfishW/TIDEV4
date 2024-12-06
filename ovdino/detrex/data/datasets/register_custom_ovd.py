import os

from .custom_ovd import register_custom_ovd_instances
from.coco_ovd import register_coco_ovd_instances
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta

CUSTOM_CATEGORIES = [
 {
  "name": "Accordion",
  "id": 0
 },
 {
  "name": "Adhesive tape",
  "id": 1
 },
 {
  "name": "Aircraft",
  "id": 2
 },
 {
  "name": "Airplane",
  "id": 3
 },
 {
  "name": "Alarm clock",
  "id": 4
 },
 {
  "name": "Alpaca",
  "id": 5
 },
 {
  "name": "Ambulance",
  "id": 6
 },
 {
  "name": "Animal",
  "id": 7
 },
 {
  "name": "Ant",
  "id": 8
 },
 {
  "name": "Antelope",
  "id": 9
 },
 {
  "name": "Apple",
  "id": 10
 },
 {
  "name": "Armadillo",
  "id": 11
 },
 {
  "name": "Artichoke",
  "id": 12
 },
 {
  "name": "Auto part",
  "id": 13
 },
 {
  "name": "Axe",
  "id": 14
 },
 {
  "name": "Backpack",
  "id": 15
 },
 {
  "name": "Bagel",
  "id": 16
 },
 {
  "name": "Baked goods",
  "id": 17
 },
 {
  "name": "Balance beam",
  "id": 18
 },
 {
  "name": "Ball",
  "id": 19
 },
 {
  "name": "Balloon",
  "id": 20
 },
 {
  "name": "Banana",
  "id": 21
 },
 {
  "name": "Band-aid",
  "id": 22
 },
 {
  "name": "Banjo",
  "id": 23
 },
 {
  "name": "Barge",
  "id": 24
 },
 {
  "name": "Barrel",
  "id": 25
 },
 {
  "name": "Baseball bat",
  "id": 26
 },
 {
  "name": "Baseball glove",
  "id": 27
 },
 {
  "name": "Bat (Animal)",
  "id": 28
 },
 {
  "name": "Bathroom accessory",
  "id": 29
 },
 {
  "name": "Bathroom cabinet",
  "id": 30
 },
 {
  "name": "Bathtub",
  "id": 31
 },
 {
  "name": "Beaker",
  "id": 32
 },
 {
  "name": "Bear",
  "id": 33
 },
 {
  "name": "Bed",
  "id": 34
 },
 {
  "name": "Bee",
  "id": 35
 },
 {
  "name": "Beehive",
  "id": 36
 },
 {
  "name": "Beer",
  "id": 37
 },
 {
  "name": "Beetle",
  "id": 38
 },
 {
  "name": "Bell pepper",
  "id": 39
 },
 {
  "name": "Belt",
  "id": 40
 },
 {
  "name": "Bench",
  "id": 41
 },
 {
  "name": "Bicycle",
  "id": 42
 },
 {
  "name": "Bicycle helmet",
  "id": 43
 },
 {
  "name": "Bicycle wheel",
  "id": 44
 },
 {
  "name": "Bidet",
  "id": 45
 },
 {
  "name": "Billboard",
  "id": 46
 },
 {
  "name": "Billiard table",
  "id": 47
 },
 {
  "name": "Binoculars",
  "id": 48
 },
 {
  "name": "Bird",
  "id": 49
 },
 {
  "name": "Blender",
  "id": 50
 },
 {
  "name": "Blue jay",
  "id": 51
 },
 {
  "name": "Boat",
  "id": 52
 },
 {
  "name": "Bomb",
  "id": 53
 },
 {
  "name": "Book",
  "id": 54
 },
 {
  "name": "Bookcase",
  "id": 55
 },
 {
  "name": "Boot",
  "id": 56
 },
 {
  "name": "Bottle",
  "id": 57
 },
 {
  "name": "Bottle opener",
  "id": 58
 },
 {
  "name": "Bow and arrow",
  "id": 59
 },
 {
  "name": "Bowl",
  "id": 60
 },
 {
  "name": "Bowling equipment",
  "id": 61
 },
 {
  "name": "Box",
  "id": 62
 },
 {
  "name": "Boy",
  "id": 63
 },
 {
  "name": "Brassiere",
  "id": 64
 },
 {
  "name": "Bread",
  "id": 65
 },
 {
  "name": "Briefcase",
  "id": 66
 },
 {
  "name": "Broccoli",
  "id": 67
 },
 {
  "name": "Bronze sculpture",
  "id": 68
 },
 {
  "name": "Brown bear",
  "id": 69
 },
 {
  "name": "Building",
  "id": 70
 },
 {
  "name": "Bull",
  "id": 71
 },
 {
  "name": "Burrito",
  "id": 72
 },
 {
  "name": "Bus",
  "id": 73
 },
 {
  "name": "Bust",
  "id": 74
 },
 {
  "name": "Butterfly",
  "id": 75
 },
 {
  "name": "Cabbage",
  "id": 76
 },
 {
  "name": "Cabinetry",
  "id": 77
 },
 {
  "name": "Cake",
  "id": 78
 },
 {
  "name": "Cake stand",
  "id": 79
 },
 {
  "name": "Calculator",
  "id": 80
 },
 {
  "name": "Camel",
  "id": 81
 },
 {
  "name": "Camera",
  "id": 82
 },
 {
  "name": "Can opener",
  "id": 83
 },
 {
  "name": "Canary",
  "id": 84
 },
 {
  "name": "Candle",
  "id": 85
 },
 {
  "name": "Candy",
  "id": 86
 },
 {
  "name": "Cannon",
  "id": 87
 },
 {
  "name": "Canoe",
  "id": 88
 },
 {
  "name": "Cantaloupe",
  "id": 89
 },
 {
  "name": "Car",
  "id": 90
 },
 {
  "name": "Carnivore",
  "id": 91
 },
 {
  "name": "Carrot",
  "id": 92
 },
 {
  "name": "Cart",
  "id": 93
 },
 {
  "name": "Cassette deck",
  "id": 94
 },
 {
  "name": "Castle",
  "id": 95
 },
 {
  "name": "Cat",
  "id": 96
 },
 {
  "name": "Cat furniture",
  "id": 97
 },
 {
  "name": "Caterpillar",
  "id": 98
 },
 {
  "name": "Cattle",
  "id": 99
 },
 {
  "name": "Ceiling fan",
  "id": 100
 },
 {
  "name": "Cello",
  "id": 101
 },
 {
  "name": "Centipede",
  "id": 102
 },
 {
  "name": "Chainsaw",
  "id": 103
 },
 {
  "name": "Chair",
  "id": 104
 },
 {
  "name": "Cheese",
  "id": 105
 },
 {
  "name": "Cheetah",
  "id": 106
 },
 {
  "name": "Chest of drawers",
  "id": 107
 },
 {
  "name": "Chicken",
  "id": 108
 },
 {
  "name": "Chime",
  "id": 109
 },
 {
  "name": "Chisel",
  "id": 110
 },
 {
  "name": "Chopsticks",
  "id": 111
 },
 {
  "name": "Christmas tree",
  "id": 112
 },
 {
  "name": "Clock",
  "id": 113
 },
 {
  "name": "Closet",
  "id": 114
 },
 {
  "name": "Clothing",
  "id": 115
 },
 {
  "name": "Coat",
  "id": 116
 },
 {
  "name": "Cocktail",
  "id": 117
 },
 {
  "name": "Cocktail shaker",
  "id": 118
 },
 {
  "name": "Coconut",
  "id": 119
 },
 {
  "name": "Coffee",
  "id": 120
 },
 {
  "name": "Coffee cup",
  "id": 121
 },
 {
  "name": "Coffee table",
  "id": 122
 },
 {
  "name": "Coffeemaker",
  "id": 123
 },
 {
  "name": "Coin",
  "id": 124
 },
 {
  "name": "Common fig",
  "id": 125
 },
 {
  "name": "Common sunflower",
  "id": 126
 },
 {
  "name": "Computer keyboard",
  "id": 127
 },
 {
  "name": "Computer monitor",
  "id": 128
 },
 {
  "name": "Computer mouse",
  "id": 129
 },
 {
  "name": "Convenience store",
  "id": 130
 },
 {
  "name": "Cookie",
  "id": 131
 },
 {
  "name": "Cooking spray",
  "id": 132
 },
 {
  "name": "Corded phone",
  "id": 133
 },
 {
  "name": "Cosmetics",
  "id": 134
 },
 {
  "name": "Couch",
  "id": 135
 },
 {
  "name": "Countertop",
  "id": 136
 },
 {
  "name": "Cowboy hat",
  "id": 137
 },
 {
  "name": "Crab",
  "id": 138
 },
 {
  "name": "Cream",
  "id": 139
 },
 {
  "name": "Cricket ball",
  "id": 140
 },
 {
  "name": "Crocodile",
  "id": 141
 },
 {
  "name": "Croissant",
  "id": 142
 },
 {
  "name": "Crown",
  "id": 143
 },
 {
  "name": "Crutch",
  "id": 144
 },
 {
  "name": "Cucumber",
  "id": 145
 },
 {
  "name": "Cupboard",
  "id": 146
 },
 {
  "name": "Curtain",
  "id": 147
 },
 {
  "name": "Cutting board",
  "id": 148
 },
 {
  "name": "Dagger",
  "id": 149
 },
 {
  "name": "Dairy Product",
  "id": 150
 },
 {
  "name": "Deer",
  "id": 151
 },
 {
  "name": "Desk",
  "id": 152
 },
 {
  "name": "Dessert",
  "id": 153
 },
 {
  "name": "Diaper",
  "id": 154
 },
 {
  "name": "Dice",
  "id": 155
 },
 {
  "name": "Digital clock",
  "id": 156
 },
 {
  "name": "Dinosaur",
  "id": 157
 },
 {
  "name": "Dishwasher",
  "id": 158
 },
 {
  "name": "Dog",
  "id": 159
 },
 {
  "name": "Dog bed",
  "id": 160
 },
 {
  "name": "Doll",
  "id": 161
 },
 {
  "name": "Dolphin",
  "id": 162
 },
 {
  "name": "Door",
  "id": 163
 },
 {
  "name": "Door handle",
  "id": 164
 },
 {
  "name": "Doughnut",
  "id": 165
 },
 {
  "name": "Dragonfly",
  "id": 166
 },
 {
  "name": "Drawer",
  "id": 167
 },
 {
  "name": "Dress",
  "id": 168
 },
 {
  "name": "Drill (Tool)",
  "id": 169
 },
 {
  "name": "Drink",
  "id": 170
 },
 {
  "name": "Drinking straw",
  "id": 171
 },
 {
  "name": "Drum",
  "id": 172
 },
 {
  "name": "Duck",
  "id": 173
 },
 {
  "name": "Dumbbell",
  "id": 174
 },
 {
  "name": "Eagle",
  "id": 175
 },
 {
  "name": "Earrings",
  "id": 176
 },
 {
  "name": "Egg (Food)",
  "id": 177
 },
 {
  "name": "Elephant",
  "id": 178
 },
 {
  "name": "Envelope",
  "id": 179
 },
 {
  "name": "Eraser",
  "id": 180
 },
 {
  "name": "Face powder",
  "id": 181
 },
 {
  "name": "Facial tissue holder",
  "id": 182
 },
 {
  "name": "Falcon",
  "id": 183
 },
 {
  "name": "Fashion accessory",
  "id": 184
 },
 {
  "name": "Fast food",
  "id": 185
 },
 {
  "name": "Fax",
  "id": 186
 },
 {
  "name": "Fedora",
  "id": 187
 },
 {
  "name": "Filing cabinet",
  "id": 188
 },
 {
  "name": "Fire hydrant",
  "id": 189
 },
 {
  "name": "Fireplace",
  "id": 190
 },
 {
  "name": "Fish",
  "id": 191
 },
 {
  "name": "Flag",
  "id": 192
 },
 {
  "name": "Flashlight",
  "id": 193
 },
 {
  "name": "Flower",
  "id": 194
 },
 {
  "name": "Flowerpot",
  "id": 195
 },
 {
  "name": "Flute",
  "id": 196
 },
 {
  "name": "Flying disc",
  "id": 197
 },
 {
  "name": "Food",
  "id": 198
 },
 {
  "name": "Food processor",
  "id": 199
 },
 {
  "name": "Football",
  "id": 200
 },
 {
  "name": "Football helmet",
  "id": 201
 },
 {
  "name": "Footwear",
  "id": 202
 },
 {
  "name": "Fork",
  "id": 203
 },
 {
  "name": "Fountain",
  "id": 204
 },
 {
  "name": "Fox",
  "id": 205
 },
 {
  "name": "French fries",
  "id": 206
 },
 {
  "name": "French horn",
  "id": 207
 },
 {
  "name": "Frog",
  "id": 208
 },
 {
  "name": "Fruit",
  "id": 209
 },
 {
  "name": "Frying pan",
  "id": 210
 },
 {
  "name": "Furniture",
  "id": 211
 },
 {
  "name": "Garden Asparagus",
  "id": 212
 },
 {
  "name": "Gas stove",
  "id": 213
 },
 {
  "name": "Giraffe",
  "id": 214
 },
 {
  "name": "Girl",
  "id": 215
 },
 {
  "name": "Glasses",
  "id": 216
 },
 {
  "name": "Glove",
  "id": 217
 },
 {
  "name": "Goat",
  "id": 218
 },
 {
  "name": "Goggles",
  "id": 219
 },
 {
  "name": "Goldfish",
  "id": 220
 },
 {
  "name": "Golf ball",
  "id": 221
 },
 {
  "name": "Golf cart",
  "id": 222
 },
 {
  "name": "Gondola",
  "id": 223
 },
 {
  "name": "Goose",
  "id": 224
 },
 {
  "name": "Grape",
  "id": 225
 },
 {
  "name": "Grapefruit",
  "id": 226
 },
 {
  "name": "Grinder",
  "id": 227
 },
 {
  "name": "Guacamole",
  "id": 228
 },
 {
  "name": "Guitar",
  "id": 229
 },
 {
  "name": "Hair dryer",
  "id": 230
 },
 {
  "name": "Hair spray",
  "id": 231
 },
 {
  "name": "Hamburger",
  "id": 232
 },
 {
  "name": "Hammer",
  "id": 233
 },
 {
  "name": "Hamster",
  "id": 234
 },
 {
  "name": "Hand dryer",
  "id": 235
 },
 {
  "name": "Handbag",
  "id": 236
 },
 {
  "name": "Handgun",
  "id": 237
 },
 {
  "name": "Harbor seal",
  "id": 238
 },
 {
  "name": "Harmonica",
  "id": 239
 },
 {
  "name": "Harp",
  "id": 240
 },
 {
  "name": "Harpsichord",
  "id": 241
 },
 {
  "name": "Hat",
  "id": 242
 },
 {
  "name": "Headphones",
  "id": 243
 },
 {
  "name": "Heater",
  "id": 244
 },
 {
  "name": "Hedgehog",
  "id": 245
 },
 {
  "name": "Helicopter",
  "id": 246
 },
 {
  "name": "Helmet",
  "id": 247
 },
 {
  "name": "High heels",
  "id": 248
 },
 {
  "name": "Hiking equipment",
  "id": 249
 },
 {
  "name": "Hippopotamus",
  "id": 250
 },
 {
  "name": "Home appliance",
  "id": 251
 },
 {
  "name": "Honeycomb",
  "id": 252
 },
 {
  "name": "Horizontal bar",
  "id": 253
 },
 {
  "name": "Horse",
  "id": 254
 },
 {
  "name": "Hot dog",
  "id": 255
 },
 {
  "name": "House",
  "id": 256
 },
 {
  "name": "Houseplant",
  "id": 257
 },
 {
  "name": "Human arm",
  "id": 258
 },
 {
  "name": "Human beard",
  "id": 259
 },
 {
  "name": "Human body",
  "id": 260
 },
 {
  "name": "Human ear",
  "id": 261
 },
 {
  "name": "Human eye",
  "id": 262
 },
 {
  "name": "Human face",
  "id": 263
 },
 {
  "name": "Human foot",
  "id": 264
 },
 {
  "name": "Human hair",
  "id": 265
 },
 {
  "name": "Human hand",
  "id": 266
 },
 {
  "name": "Human head",
  "id": 267
 },
 {
  "name": "Human leg",
  "id": 268
 },
 {
  "name": "Human mouth",
  "id": 269
 },
 {
  "name": "Human nose",
  "id": 270
 },
 {
  "name": "Humidifier",
  "id": 271
 },
 {
  "name": "Ice cream",
  "id": 272
 },
 {
  "name": "Indoor rower",
  "id": 273
 },
 {
  "name": "Infant bed",
  "id": 274
 },
 {
  "name": "Insect",
  "id": 275
 },
 {
  "name": "Invertebrate",
  "id": 276
 },
 {
  "name": "Ipod",
  "id": 277
 },
 {
  "name": "Isopod",
  "id": 278
 },
 {
  "name": "Jacket",
  "id": 279
 },
 {
  "name": "Jacuzzi",
  "id": 280
 },
 {
  "name": "Jaguar (Animal)",
  "id": 281
 },
 {
  "name": "Jeans",
  "id": 282
 },
 {
  "name": "Jellyfish",
  "id": 283
 },
 {
  "name": "Jet ski",
  "id": 284
 },
 {
  "name": "Jug",
  "id": 285
 },
 {
  "name": "Juice",
  "id": 286
 },
 {
  "name": "Kangaroo",
  "id": 287
 },
 {
  "name": "Kettle",
  "id": 288
 },
 {
  "name": "Kitchen & dining room table",
  "id": 289
 },
 {
  "name": "Kitchen appliance",
  "id": 290
 },
 {
  "name": "Kitchen knife",
  "id": 291
 },
 {
  "name": "Kitchen utensil",
  "id": 292
 },
 {
  "name": "Kite",
  "id": 293
 },
 {
  "name": "Knife",
  "id": 294
 },
 {
  "name": "Koala",
  "id": 295
 },
 {
  "name": "Ladder",
  "id": 296
 },
 {
  "name": "Ladle",
  "id": 297
 },
 {
  "name": "Ladybug",
  "id": 298
 },
 {
  "name": "Lamp",
  "id": 299
 },
 {
  "name": "Land vehicle",
  "id": 300
 },
 {
  "name": "Lantern",
  "id": 301
 },
 {
  "name": "Laptop",
  "id": 302
 },
 {
  "name": "Lavender (Plant)",
  "id": 303
 },
 {
  "name": "Lemon",
  "id": 304
 },
 {
  "name": "Leopard",
  "id": 305
 },
 {
  "name": "Light bulb",
  "id": 306
 },
 {
  "name": "Light switch",
  "id": 307
 },
 {
  "name": "Lighthouse",
  "id": 308
 },
 {
  "name": "Lily",
  "id": 309
 },
 {
  "name": "Limousine",
  "id": 310
 },
 {
  "name": "Lion",
  "id": 311
 },
 {
  "name": "Lipstick",
  "id": 312
 },
 {
  "name": "Lizard",
  "id": 313
 },
 {
  "name": "Lobster",
  "id": 314
 },
 {
  "name": "Loveseat",
  "id": 315
 },
 {
  "name": "Luggage and bags",
  "id": 316
 },
 {
  "name": "Lynx",
  "id": 317
 },
 {
  "name": "Magpie",
  "id": 318
 },
 {
  "name": "Mammal",
  "id": 319
 },
 {
  "name": "Man",
  "id": 320
 },
 {
  "name": "Mango",
  "id": 321
 },
 {
  "name": "Maple",
  "id": 322
 },
 {
  "name": "Maracas",
  "id": 323
 },
 {
  "name": "Marine invertebrates",
  "id": 324
 },
 {
  "name": "Marine mammal",
  "id": 325
 },
 {
  "name": "Measuring cup",
  "id": 326
 },
 {
  "name": "Mechanical fan",
  "id": 327
 },
 {
  "name": "Medical equipment",
  "id": 328
 },
 {
  "name": "Microphone",
  "id": 329
 },
 {
  "name": "Microwave oven",
  "id": 330
 },
 {
  "name": "Milk",
  "id": 331
 },
 {
  "name": "Miniskirt",
  "id": 332
 },
 {
  "name": "Mirror",
  "id": 333
 },
 {
  "name": "Missile",
  "id": 334
 },
 {
  "name": "Mixer",
  "id": 335
 },
 {
  "name": "Mixing bowl",
  "id": 336
 },
 {
  "name": "Mobile phone",
  "id": 337
 },
 {
  "name": "Monkey",
  "id": 338
 },
 {
  "name": "Moths and butterflies",
  "id": 339
 },
 {
  "name": "Motorcycle",
  "id": 340
 },
 {
  "name": "Mouse",
  "id": 341
 },
 {
  "name": "Muffin",
  "id": 342
 },
 {
  "name": "Mug",
  "id": 343
 },
 {
  "name": "Mule",
  "id": 344
 },
 {
  "name": "Mushroom",
  "id": 345
 },
 {
  "name": "Musical instrument",
  "id": 346
 },
 {
  "name": "Musical keyboard",
  "id": 347
 },
 {
  "name": "Nail (Construction)",
  "id": 348
 },
 {
  "name": "Necklace",
  "id": 349
 },
 {
  "name": "Nightstand",
  "id": 350
 },
 {
  "name": "Oboe",
  "id": 351
 },
 {
  "name": "Office building",
  "id": 352
 },
 {
  "name": "Office supplies",
  "id": 353
 },
 {
  "name": "Orange",
  "id": 354
 },
 {
  "name": "Organ (Musical Instrument)",
  "id": 355
 },
 {
  "name": "Ostrich",
  "id": 356
 },
 {
  "name": "Otter",
  "id": 357
 },
 {
  "name": "Oven",
  "id": 358
 },
 {
  "name": "Owl",
  "id": 359
 },
 {
  "name": "Oyster",
  "id": 360
 },
 {
  "name": "Paddle",
  "id": 361
 },
 {
  "name": "Palm tree",
  "id": 362
 },
 {
  "name": "Pancake",
  "id": 363
 },
 {
  "name": "Panda",
  "id": 364
 },
 {
  "name": "Paper cutter",
  "id": 365
 },
 {
  "name": "Paper towel",
  "id": 366
 },
 {
  "name": "Parachute",
  "id": 367
 },
 {
  "name": "Parking meter",
  "id": 368
 },
 {
  "name": "Parrot",
  "id": 369
 },
 {
  "name": "Pasta",
  "id": 370
 },
 {
  "name": "Pastry",
  "id": 371
 },
 {
  "name": "Peach",
  "id": 372
 },
 {
  "name": "Pear",
  "id": 373
 },
 {
  "name": "Pen",
  "id": 374
 },
 {
  "name": "Pencil case",
  "id": 375
 },
 {
  "name": "Pencil sharpener",
  "id": 376
 },
 {
  "name": "Penguin",
  "id": 377
 },
 {
  "name": "Perfume",
  "id": 378
 },
 {
  "name": "Person",
  "id": 379
 },
 {
  "name": "Personal care",
  "id": 380
 },
 {
  "name": "Personal flotation device",
  "id": 381
 },
 {
  "name": "Piano",
  "id": 382
 },
 {
  "name": "Picnic basket",
  "id": 383
 },
 {
  "name": "Picture frame",
  "id": 384
 },
 {
  "name": "Pig",
  "id": 385
 },
 {
  "name": "Pillow",
  "id": 386
 },
 {
  "name": "Pineapple",
  "id": 387
 },
 {
  "name": "Pitcher (Container)",
  "id": 388
 },
 {
  "name": "Pizza",
  "id": 389
 },
 {
  "name": "Pizza cutter",
  "id": 390
 },
 {
  "name": "Plant",
  "id": 391
 },
 {
  "name": "Plastic bag",
  "id": 392
 },
 {
  "name": "Plate",
  "id": 393
 },
 {
  "name": "Platter",
  "id": 394
 },
 {
  "name": "Plumbing fixture",
  "id": 395
 },
 {
  "name": "Polar bear",
  "id": 396
 },
 {
  "name": "Pomegranate",
  "id": 397
 },
 {
  "name": "Popcorn",
  "id": 398
 },
 {
  "name": "Porch",
  "id": 399
 },
 {
  "name": "Porcupine",
  "id": 400
 },
 {
  "name": "Poster",
  "id": 401
 },
 {
  "name": "Potato",
  "id": 402
 },
 {
  "name": "Power plugs and sockets",
  "id": 403
 },
 {
  "name": "Pressure cooker",
  "id": 404
 },
 {
  "name": "Pretzel",
  "id": 405
 },
 {
  "name": "Printer",
  "id": 406
 },
 {
  "name": "Pumpkin",
  "id": 407
 },
 {
  "name": "Punching bag",
  "id": 408
 },
 {
  "name": "Rabbit",
  "id": 409
 },
 {
  "name": "Raccoon",
  "id": 410
 },
 {
  "name": "Racket",
  "id": 411
 },
 {
  "name": "Radish",
  "id": 412
 },
 {
  "name": "Ratchet (Device)",
  "id": 413
 },
 {
  "name": "Raven",
  "id": 414
 },
 {
  "name": "Rays and skates",
  "id": 415
 },
 {
  "name": "Red panda",
  "id": 416
 },
 {
  "name": "Refrigerator",
  "id": 417
 },
 {
  "name": "Remote control",
  "id": 418
 },
 {
  "name": "Reptile",
  "id": 419
 },
 {
  "name": "Rhinoceros",
  "id": 420
 },
 {
  "name": "Rifle",
  "id": 421
 },
 {
  "name": "Ring binder",
  "id": 422
 },
 {
  "name": "Rocket",
  "id": 423
 },
 {
  "name": "Roller skates",
  "id": 424
 },
 {
  "name": "Rose",
  "id": 425
 },
 {
  "name": "Rugby ball",
  "id": 426
 },
 {
  "name": "Ruler",
  "id": 427
 },
 {
  "name": "Salad",
  "id": 428
 },
 {
  "name": "Salt and pepper shakers",
  "id": 429
 },
 {
  "name": "Sandal",
  "id": 430
 },
 {
  "name": "Sandwich",
  "id": 431
 },
 {
  "name": "Saucer",
  "id": 432
 },
 {
  "name": "Saxophone",
  "id": 433
 },
 {
  "name": "Scale",
  "id": 434
 },
 {
  "name": "Scarf",
  "id": 435
 },
 {
  "name": "Scissors",
  "id": 436
 },
 {
  "name": "Scoreboard",
  "id": 437
 },
 {
  "name": "Scorpion",
  "id": 438
 },
 {
  "name": "Screwdriver",
  "id": 439
 },
 {
  "name": "Sculpture",
  "id": 440
 },
 {
  "name": "Sea lion",
  "id": 441
 },
 {
  "name": "Sea turtle",
  "id": 442
 },
 {
  "name": "Seafood",
  "id": 443
 },
 {
  "name": "Seahorse",
  "id": 444
 },
 {
  "name": "Seat belt",
  "id": 445
 },
 {
  "name": "Segway",
  "id": 446
 },
 {
  "name": "Serving tray",
  "id": 447
 },
 {
  "name": "Sewing machine",
  "id": 448
 },
 {
  "name": "Shark",
  "id": 449
 },
 {
  "name": "Sheep",
  "id": 450
 },
 {
  "name": "Shelf",
  "id": 451
 },
 {
  "name": "Shellfish",
  "id": 452
 },
 {
  "name": "Shirt",
  "id": 453
 },
 {
  "name": "Shorts",
  "id": 454
 },
 {
  "name": "Shotgun",
  "id": 455
 },
 {
  "name": "Shower",
  "id": 456
 },
 {
  "name": "Shrimp",
  "id": 457
 },
 {
  "name": "Sink",
  "id": 458
 },
 {
  "name": "Skateboard",
  "id": 459
 },
 {
  "name": "Ski",
  "id": 460
 },
 {
  "name": "Skirt",
  "id": 461
 },
 {
  "name": "Skull",
  "id": 462
 },
 {
  "name": "Skunk",
  "id": 463
 },
 {
  "name": "Skyscraper",
  "id": 464
 },
 {
  "name": "Slow cooker",
  "id": 465
 },
 {
  "name": "Snack",
  "id": 466
 },
 {
  "name": "Snail",
  "id": 467
 },
 {
  "name": "Snake",
  "id": 468
 },
 {
  "name": "Snowboard",
  "id": 469
 },
 {
  "name": "Snowman",
  "id": 470
 },
 {
  "name": "Snowmobile",
  "id": 471
 },
 {
  "name": "Snowplow",
  "id": 472
 },
 {
  "name": "Soap dispenser",
  "id": 473
 },
 {
  "name": "Sock",
  "id": 474
 },
 {
  "name": "Sofa bed",
  "id": 475
 },
 {
  "name": "Sombrero",
  "id": 476
 },
 {
  "name": "Sparrow",
  "id": 477
 },
 {
  "name": "Spatula",
  "id": 478
 },
 {
  "name": "Spice rack",
  "id": 479
 },
 {
  "name": "Spider",
  "id": 480
 },
 {
  "name": "Spoon",
  "id": 481
 },
 {
  "name": "Sports equipment",
  "id": 482
 },
 {
  "name": "Sports uniform",
  "id": 483
 },
 {
  "name": "Squash (Plant)",
  "id": 484
 },
 {
  "name": "Squid",
  "id": 485
 },
 {
  "name": "Squirrel",
  "id": 486
 },
 {
  "name": "Stairs",
  "id": 487
 },
 {
  "name": "Stapler",
  "id": 488
 },
 {
  "name": "Starfish",
  "id": 489
 },
 {
  "name": "Stationary bicycle",
  "id": 490
 },
 {
  "name": "Stethoscope",
  "id": 491
 },
 {
  "name": "Stool",
  "id": 492
 },
 {
  "name": "Stop sign",
  "id": 493
 },
 {
  "name": "Strawberry",
  "id": 494
 },
 {
  "name": "Street light",
  "id": 495
 },
 {
  "name": "Stretcher",
  "id": 496
 },
 {
  "name": "Studio couch",
  "id": 497
 },
 {
  "name": "Submarine",
  "id": 498
 },
 {
  "name": "Submarine sandwich",
  "id": 499
 },
 {
  "name": "Suit",
  "id": 500
 },
 {
  "name": "Suitcase",
  "id": 501
 },
 {
  "name": "Sun hat",
  "id": 502
 },
 {
  "name": "Sunglasses",
  "id": 503
 },
 {
  "name": "Surfboard",
  "id": 504
 },
 {
  "name": "Sushi",
  "id": 505
 },
 {
  "name": "Swan",
  "id": 506
 },
 {
  "name": "Swim cap",
  "id": 507
 },
 {
  "name": "Swimming pool",
  "id": 508
 },
 {
  "name": "Swimwear",
  "id": 509
 },
 {
  "name": "Sword",
  "id": 510
 },
 {
  "name": "Syringe",
  "id": 511
 },
 {
  "name": "Table",
  "id": 512
 },
 {
  "name": "Table tennis racket",
  "id": 513
 },
 {
  "name": "Tablet computer",
  "id": 514
 },
 {
  "name": "Tableware",
  "id": 515
 },
 {
  "name": "Taco",
  "id": 516
 },
 {
  "name": "Tank",
  "id": 517
 },
 {
  "name": "Tap",
  "id": 518
 },
 {
  "name": "Tart",
  "id": 519
 },
 {
  "name": "Taxi",
  "id": 520
 },
 {
  "name": "Tea",
  "id": 521
 },
 {
  "name": "Teapot",
  "id": 522
 },
 {
  "name": "Teddy bear",
  "id": 523
 },
 {
  "name": "Telephone",
  "id": 524
 },
 {
  "name": "Television",
  "id": 525
 },
 {
  "name": "Tennis ball",
  "id": 526
 },
 {
  "name": "Tennis racket",
  "id": 527
 },
 {
  "name": "Tent",
  "id": 528
 },
 {
  "name": "Tiara",
  "id": 529
 },
 {
  "name": "Tick",
  "id": 530
 },
 {
  "name": "Tie",
  "id": 531
 },
 {
  "name": "Tiger",
  "id": 532
 },
 {
  "name": "Tin can",
  "id": 533
 },
 {
  "name": "Tire",
  "id": 534
 },
 {
  "name": "Toaster",
  "id": 535
 },
 {
  "name": "Toilet",
  "id": 536
 },
 {
  "name": "Toilet paper",
  "id": 537
 },
 {
  "name": "Tomato",
  "id": 538
 },
 {
  "name": "Tool",
  "id": 539
 },
 {
  "name": "Toothbrush",
  "id": 540
 },
 {
  "name": "Torch",
  "id": 541
 },
 {
  "name": "Tortoise",
  "id": 542
 },
 {
  "name": "Towel",
  "id": 543
 },
 {
  "name": "Tower",
  "id": 544
 },
 {
  "name": "Toy",
  "id": 545
 },
 {
  "name": "Traffic light",
  "id": 546
 },
 {
  "name": "Traffic sign",
  "id": 547
 },
 {
  "name": "Train",
  "id": 548
 },
 {
  "name": "Training bench",
  "id": 549
 },
 {
  "name": "Treadmill",
  "id": 550
 },
 {
  "name": "Tree",
  "id": 551
 },
 {
  "name": "Tree house",
  "id": 552
 },
 {
  "name": "Tripod",
  "id": 553
 },
 {
  "name": "Trombone",
  "id": 554
 },
 {
  "name": "Trousers",
  "id": 555
 },
 {
  "name": "Truck",
  "id": 556
 },
 {
  "name": "Trumpet",
  "id": 557
 },
 {
  "name": "Turkey",
  "id": 558
 },
 {
  "name": "Turtle",
  "id": 559
 },
 {
  "name": "Umbrella",
  "id": 560
 },
 {
  "name": "Unicycle",
  "id": 561
 },
 {
  "name": "Van",
  "id": 562
 },
 {
  "name": "Vase",
  "id": 563
 },
 {
  "name": "Vegetable",
  "id": 564
 },
 {
  "name": "Vehicle",
  "id": 565
 },
 {
  "name": "Vehicle registration plate",
  "id": 566
 },
 {
  "name": "Violin",
  "id": 567
 },
 {
  "name": "Volleyball (Ball)",
  "id": 568
 },
 {
  "name": "Waffle",
  "id": 569
 },
 {
  "name": "Waffle iron",
  "id": 570
 },
 {
  "name": "Wall clock",
  "id": 571
 },
 {
  "name": "Wardrobe",
  "id": 572
 },
 {
  "name": "Washing machine",
  "id": 573
 },
 {
  "name": "Waste container",
  "id": 574
 },
 {
  "name": "Watch",
  "id": 575
 },
 {
  "name": "Watercraft",
  "id": 576
 },
 {
  "name": "Watermelon",
  "id": 577
 },
 {
  "name": "Weapon",
  "id": 578
 },
 {
  "name": "Whale",
  "id": 579
 },
 {
  "name": "Wheel",
  "id": 580
 },
 {
  "name": "Wheelchair",
  "id": 581
 },
 {
  "name": "Whisk",
  "id": 582
 },
 {
  "name": "Whiteboard",
  "id": 583
 },
 {
  "name": "Willow",
  "id": 584
 },
 {
  "name": "Window",
  "id": 585
 },
 {
  "name": "Window blind",
  "id": 586
 },
 {
  "name": "Wine",
  "id": 587
 },
 {
  "name": "Wine glass",
  "id": 588
 },
 {
  "name": "Wine rack",
  "id": 589
 },
 {
  "name": "Winter melon",
  "id": 590
 },
 {
  "name": "Wok",
  "id": 591
 },
 {
  "name": "Woman",
  "id": 592
 },
 {
  "name": "Wood-burning stove",
  "id": 593
 },
 {
  "name": "Woodpecker",
  "id": 594
 },
 {
  "name": "Worm",
  "id": 595
 },
 {
  "name": "Wrench",
  "id": 596
 },
 {
  "name": "Zebra",
  "id": 597
 },
 {
  "name": "Zucchini",
  "id": 598
 }
]

NUM_CATEGORY = len(CUSTOM_CATEGORIES)


def _get_custom_instances_meta():
    thing_ids = [k["id"] for k in CUSTOM_CATEGORIES]
    assert len(thing_ids) == NUM_CATEGORY, len(thing_ids)
    # Mapping from the incontiguous category id to contiguous id.
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CUSTOM_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


_PREDEFINED_SPLITS = {
    # image_root, json_file, num_sampled_classes, template
    "custom_train_ovd_unipro": (
        "OpenImages/data",
        "OpenImages/labels.json",
        NUM_CATEGORY,
        "full",
    ),
    "custom_train_ovd": (
        "OpenImages/data",
        "OpenImages/labels.json",
        NUM_CATEGORY,
        "full",
    ),
    "custom_val_ovd_unipro": (
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
        80,
        "full",
    ),
    "custom_val_ovd": (
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
        80,
        "identity",
    ),
    # "custom_test_ovd": (
    #     "custom/test",
    #     "custom/annotations/test.json",
    #     NUM_CATEGORY,
    #     "identity",
    # ),
}


def register_all_custom_instances(root):
    for key, (
        image_root,
        json_file,
        num_sampled_classes,
        template,
    ) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datas`.
        if 'coco' not in key:
            register_custom_ovd_instances(
                key,
                _get_custom_instances_meta(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                num_sampled_classes,
                template=template,
                test_mode=True if "val" in key else False,
            )
        else:
            register_coco_ovd_instances(
                key,
                _get_coco_instances_meta(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                num_sampled_classes,
                template=template,
                test_mode=True if "val" in key else False,
            )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_custom_instances(_root)
