from picamera2 import Picamera2
import time

picam2 = Picamera2()

# Attendre quelques secondes pour que la caméra se stabilise (facultatif)
time.sleep(2)

# Définir le nom du fichier pour la photo
file_name = "test2.jpg"

# Prendre une photo et l'enregistrer
picam2.capture(file_name)

# Fermer la caméra
picam2.close()

print("La photo a été enregistrée sous le nom", file_name)
