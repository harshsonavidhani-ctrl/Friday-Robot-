"""Listen to a serial device and report what it sends. Read-only.

This NEVER writes to the port. Some devices only stream after a command, so
silence here means "didn't volunteer data", not "broken".

    python scripts/probe_serial.py /dev/ttyACM0 [baud] [seconds]
"""

import sys
import time

import serial


def probe(port: str, baud: int, seconds: float) -> None:
    print(f"Listening to {port} at {baud} baud for {seconds:g}s (no writes)...\n")

    # dsrdtr/rtscts off so opening the port can't toggle a reset line.
    with serial.Serial(port, baud, timeout=0.2, dsrdtr=False, rtscts=False) as ser:
        chunks: list[bytes] = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            data = ser.read(4096)
            if data:
                chunks.append(data)

    raw = b"".join(chunks)
    if not raw:
        print("Nothing received. The device may need a command first,")
        print("or may be using a different baud rate.")
        return

    print(f"Received {len(raw)} bytes in {seconds:g}s "
          f"(~{len(raw) / seconds:.0f} B/s)\n")

    print("--- first 256 bytes, hex ---")
    head = raw[:256]
    for i in range(0, len(head), 16):
        row = head[i:i + 16]
        hexed = " ".join(f"{b:02x}" for b in row)
        ascii_ = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        print(f"{i:04x}  {hexed:<47}  |{ascii_}|")

    printable = sum(1 for b in raw if 32 <= b < 127 or b in (10, 13))
    ratio = printable / len(raw)
    print(f"\nPrintable: {ratio:.0%} -> looks like "
          f"{'TEXT' if ratio > 0.85 else 'BINARY'}")

    if ratio > 0.85:
        print("\n--- as text, first 20 lines ---")
        text = raw.decode("utf-8", errors="replace")
        for line in text.splitlines()[:20]:
            print(f"  {line}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    probe(
        sys.argv[1],
        int(sys.argv[2]) if len(sys.argv) > 2 else 115200,
        float(sys.argv[3]) if len(sys.argv) > 3 else 5.0,
    )
