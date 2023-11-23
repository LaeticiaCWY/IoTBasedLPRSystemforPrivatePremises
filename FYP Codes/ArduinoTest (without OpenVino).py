# Initialize and import the necessary libraries
import serial
import subprocess
import time

# Initialize the port that is connected to Arduino
ser = serial.Serial('COM3', 9600)
car_detected = False

while True:
    # Read the data sent from Arduino
    data = ser.readline().decode('utf-8').strip()

    if data == "Waiting":
        car_detected = False
        print("False")
    if data == "Car Detected":
        car_detected = True
        print("Car detected! Running Python command.")
        # Run the command that will turn on the Python script (e.g., camera, LPR script, etc.)
        subprocess.run(["python", "wpodnet_script_L1.py", "--weights", "runs/train/exp12/weights/best.pt", "--source", "1"])
        print("LPR Process Complete.")
    car_detected = False
    print("Waiting.")
