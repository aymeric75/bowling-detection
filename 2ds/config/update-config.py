import json
import sys
import io
import argparse
import logging
#------------------------------------------------------------------------
#VARIABLES
path_2ds            = "/home/pi/Desktop/2ds/"
path_config         = path_2ds + "config/config.json"
path_log            = path_2ds + "log/log.txt"
#------------------------------------------------------------------------
#FUNCTIONS
# Force l'encodage du stdout à UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def setup_logger():
    # Configure le logging avec l'encodage UTF-8
    logging.basicConfig(filename=path_log, 
                        filemode='a', 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        encoding='utf-8')  # Spécifiez l'encodage ici

    # Ajoute un message dans le log
    logging.info('Le log a été configuré.')

# Appel de la fonction pour mettre en place le logger
setup_logger()

logging.info('Requête de mise à jour de configuration')
#------------------------------------------------------------------------
#PARAMETRES
parser              = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--config_key', type=str, default="null")
parser.add_argument('--config_value', type=int, default=0)
#
args                = parser.parse_args()
#------------------------------------------------------------------------
#HANDLE CONFIG KEY / VALUE
config_key          = args.config_key
config_value        = args.config_value

config_key_list     = [
    "line_l_x1", "line_l_x2", "line_r_x1", "line_r_x2", "line_y1", "line_y2", "pins_l_x1", "pins_l_x2", 
    "pins_r_x1", "pins_r_x2", "pins_y1", "pins_y2"
] 

if not config_key in config_key_list:
    sys.exit(0)

if config_value is None:
    print(f"L'index '{config_key}' n'existe pas dans le switch.")
    sys.exit(0)
#------------------------------------------------------------------------
#UPDATE CONFIG
def update_config(key, value):
    # Charge le fichier de configuration JSON
    try:
        with open(path_config, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print("Le fichier de configuration n'a pas été trouvé.")
        return False
    except json.JSONDecodeError:
        print("Erreur lors de la lecture du fichier JSON.")
        return False

    # Modifie la clé avec la nouvelle valeur
    config[key] = value

    # Enregistre les modifications dans le fichier JSON
    try:
        with open(path_config, 'w') as file:
            json.dump(config, file, indent=4)
        print("Fichier de configuration mis à jour.")
        return True
    except IOError:
        print("Erreur lors de l'écriture dans le fichier de configuration.")
        return False

# Tentative de mise à jour de la configuration
success         = update_config(config_key, config_value)
sys.exit(0 if success else 1)