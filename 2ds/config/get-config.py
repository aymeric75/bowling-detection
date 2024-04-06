import json
#------------------------------------------------------------------------
#VARIABLES
path_2ds            = "/home/pi/Desktop/2ds/"
path_config         = path_2ds + "config/config.json"
#------------------------------------------------------------------------
def read_config():
    try:
        with open(path_config, 'r') as config_file:
            config_data = json.load(config_file)
        return config_data
    except FileNotFoundError:
        return "Le fichier de configuration n'a pas été trouvé."
    except json.JSONDecodeError:
        return "Erreur lors de l'analyse du fichier de configuration JSON."
    except Exception as e:
        return f"Une erreur inattendue est survenue: {str(e)}"

# Appel de la fonction et impression du résultat
config_contents     = read_config()
print(config_contents)