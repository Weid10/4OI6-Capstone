import serial

PORT_NAME = '/dev/ttyUSB0'  # Update this to serial port
BAUD_RATE = 9600
TRIGGER_CHAR = 'V'

# Global serial port object
ser = None

def init_serial(port=PORT_NAME, baudrate=BAUD_RATE):
    """
    Open serial connection
    """
    global ser
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Serial connection established on {port} at {baudrate} baud")
        return True
    except serial.SerialException as e:
        print(f"Error opening serial port {port}: {e}")
        return False

def wait_for_trigger():
    """
    Block until trigger character is received
    """
    
    while True:
        char = ser.read(1).decode('utf-8', errors='ignore')
        if char == TRIGGER_CHAR:
            print(f"Trigger received: {TRIGGER_CHAR}")
            return True

def send_volume(volume_ml):
    """
    Send volume measurement to other device
    Format: <volume>\n (e.g., "250\n")
    """

    if ser.is_open:
        try:
            message = f"{int(volume_ml)}\n"
            ser.write(message.encode('utf-8'))
            print(f"Sent volume: {message.strip()}")
            return True
        except Exception as e:
            print(f"Error sending volume: {e}")
            return False
    return False

def close_serial():
    """Close serial connection."""
    global ser
    if ser and ser.is_open:
        ser.close()
        print("Serial connection closed")

