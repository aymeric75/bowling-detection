import RPi.GPIO as GPIO
import time

# Définir le numéro de la broche GPIO que vous utilisez pour contrôler le ventilateur
GPIO_PIN = 4

# Configuration de la bibliothèque GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_PIN, GPIO.OUT)

try:
    # Démarrer le ventilateur
    GPIO.output(GPIO_PIN, GPIO.HIGH)
    print("Ventilateur démarré. Pour arrêter, appuyez sur Ctrl+C.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    # Arrêter le ventilateur et nettoyer la configuration GPIO lors de l'interruption
    GPIO.output(GPIO_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("Ventilateur arrêté. Configuration GPIO nettoyée.")
