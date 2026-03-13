import serial
import numpy as np

PORT = "COM10"


def read_exact(serial_port, size):
    """Read exactly `size` bytes or return partial data on timeout."""
    buffer = bytearray()
    while len(buffer) < size:
        chunk = serial_port.read(size - len(buffer))
        if not chunk:
            break
        buffer.extend(chunk)
    return bytes(buffer)

def synchronise_UART(serial_port):
    """
    Synchronizes the UART communication by sending a byte and waiting for a response.

    Args:
        serial_port (serial.Serial): The serial port object to use for communication.

    Returns:
        None
    """
    while (1):
        serial_port.write(b"\xAB")
        ret = serial_port.read(1)
        if (ret == b"\xCD"):
            serial_port.read(1)
            break


def send_inputs_to_STM32(inputs, serial_port):
    """
    Sends a numpy array of inputs to the STM32 microcontroller via a serial port.

    Args:
        inputs (numpy.ndarray): The inputs to send to the STM32.
        serial_port (serial.Serial): The serial port to use for communication.

    Returns:
        None
    """
    inputs = np.asarray(inputs, dtype=np.float32).ravel()
    serial_port.write(inputs.tobytes())


def read_output_from_STM32(serial_port, output_size):
    """
    Reads output_size bytes from the given serial port and returns float values in [0, 1].

    Args:
    serial_port: A serial port object.

    Returns:
    A list of float values obtained by dividing each byte by 255.
    """
    output = read_exact(serial_port, output_size)
    if len(output) != output_size:
        raise TimeoutError(f"Expected {output_size} output bytes, got {len(output)}")

    float_values = [int(out) / 255 for out in output]
    return float_values


def evaluate_model_on_STM32(iterations, serial_port, output_size):
    """
    Evaluates the accuracy of a machine learning model on an STM32 device.

    Args:
        iterations (int): The number of iterations to run the evaluation for.
        serial_port (Serial): The serial port object used to communicate with the STM32 device.

    Returns:
        float: The accuracy of the model, as a percentage.
    """
    accuracy = 0
    for i in range(iterations):
        print(f"----- Iteration {i+1} -----")
        send_inputs_to_STM32(X_test[i], serial_port)
        output = read_output_from_STM32(serial_port, output_size)
        if (np.argmax(output) == np.argmax(Y_test[i])):
            accuracy += 1 / iterations
        print(f"   Expected output: {Y_test[i]}")
        print(f"   Received output: {output}")
        print(f"----------------------- Accuracy: {accuracy:.2f}\n")
    return accuracy


if __name__ == '__main__':
    X_test = np.load("./model/X_test.npy")
    Y_test = np.load("./model/y_test.npy")
    output_size = Y_test.shape[-1] if Y_test.ndim > 1 else 1

    with serial.Serial(PORT, 115200, timeout=2) as ser:
        print("Synchronising...")
        synchronise_UART(ser)
        print("Synchronised")

        print("Evaluating model on STM32...")
        iterations = min(100, len(X_test))
        error = evaluate_model_on_STM32(iterations, ser, output_size)