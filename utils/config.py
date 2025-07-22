# Central configuration
DEFICIT_MAP = {
    1: "endurance",
    2: "bas_du_corps_equilibre",
    3: "fatigue",
    4: "force_bas_du_corps",
    5: "force_main",
    6: "masse_maigre_appendiculaire",
    7: "mobilite",
    8: "puissance_bas_du_corps",
    9: "souplesse",
    10: "vitesse_de_marche"
}

CLASSIF_COLUMNS = [f"{name}_classif" for name in DEFICIT_MAP.values()]

PHONE_CATEGORIES = {
    'vowels': ['a', 'e', 'E', 'i', 'o', 'u', 'y', '2', '9', 'ã', 'õ', 'ũ', 'ɛ', 'œ', 'ə', 'ɔ', 'ø', 'ɑ', 'y', 'ɪ', 'ʊ', 'eː', 'oː'],
    'plosives': ['p', 'b', 't', 'd', 'k', 'g'],
    'fricatives': ['f', 'v', 's', 'z', 'ʃ', 'ʒ', 'ʁ', 'h'],
    'nasals': ['m', 'n', 'ɲ', 'ŋ'],
    'liquids': ['l', 'ʁ'],
    'glides': ['j', 'w', 'ɥ'],
    'silences': ['', 'sp', 'sil']
}

SEMANTIC_CATEGORIES = {
    'time': [
        "heure", "temps", "jour", "maintenant", "alors", "moment", "fois", "début", "fin", 
        "année", "mois", "semaine", "lundi", "mardi", "mercredi", "jeudi", "vendredi", 
        "samedi", "dimanche", "matin", "soir", "nuit", "instant", "siècle", "décennie", 
        "minute", "seconde", "futur", "passé", "présent", "tôt", "tard", "bientôt", 
        "hier", "aujourd'hui", "demain", "après", "avant", "pendant", "depuis"
    ],
    'narrator': [
        "je", "moi", "mon", "ma", "mes", "notre", "nos", "mien", "mienne", "miens", 
        "miennes", "personnellement", "j'", "m'", "me", "monde", "ma personne"
    ],
    'pos_emotion': [
        "bon", "bien", "heureux", "joie", "content", "satisfait", "excellent", "super", 
        "génial", "merveilleux", "positif", "agréable", "plaisir", "sourire", "amour", 
        "adorer", "apprécier", "fier", "calme", "sérénité", "paix", "enthousiasme", 
        "optimiste", "réussi", "victoire", "gai", "ravissant", "sublime", "fantastique",
        "extraordinaire", "magnifique", "parfait", "émerveillement", "félicité"
    ],
    'neg_emotion': [
        "mauvais", "mal", "triste", "colère", "désolé", "peur", "anxiété", "déçu", 
        "négatif", "désagréable", "douleur", "pleurer", "détester", "horrible", "terrible", 
        "effrayant", "stress", "inquiet", "échec", "solitude", "jalousie", "honte", 
        "culpabilité", "désespoir", "désastre", "catastrophe", "mélancolie", "dépression",
        "agonie", "souffrance", "amer", "rancoeur", "frustration", "regret"
    ],
    'location': [
        "lieu", "endroit", "ville", "maison", "rue", "pays", "monde", "région", "quartier", 
        "adresse", "chez", "intérieur", "extérieur", "gauche", "droite", "nord", "sud", 
        "est", "ouest", "ciel", "terre", "mer", "montagne", "forêt", "bâtiment", "salle", 
        "pièce", "parc", "jardin", "place", "route", "chemin", "pont", "aéroport", "gare", 
        "station", "métro", "arrêt", "ici", "là", "là-bas", "partout", "quelque part", 
        "ailleurs", "localisation", "position", "site", "zone", "continent", "océan", 
        "fleuve", "lac", "île", "péninsule", "désert", "vallée", "plage", "frontière"
    ]
}