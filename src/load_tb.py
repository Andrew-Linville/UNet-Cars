#!/usr/bin/env python3
import os
import subprocess
import getpass
import socket

def select_logdir(base_dir):
    """List available TensorBoard log directories and return user selection."""
    logdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not logdirs:
        print(f"No log directories found in {base_dir}")
        exit(1)

    print("Available log directories:")
    for i, d in enumerate(sorted(logdirs)):
        print(f"[{i}] {d}")

    choice = int(input("Select logdir index: "))
    selected = os.path.join(base_dir, sorted(logdirs)[choice])
    print(f"\nSelected logdir: {selected}")
    return selected

def main():
    base_dir = input("Enter base directory for TensorBoard logs (e.g. ./runs): ").strip()
    if not base_dir:
        base_dir = "./runs"

    selected_logdir = select_logdir(base_dir)

    port = input("Enter port for TensorBoard (default 6006): ").strip() or "6006"

    # Get username and hostname
    user = getpass.getuser()
    hostname = socket.gethostname()

    print("\n============================================")
    print("💡 To view TensorBoard on your local machine:")
    print("--------------------------------------------")
    print(f"Run this on your LOCAL terminal (not ARC):\n")
    print(f"ssh -L {port}:localhost:{port} {user}@arc.vt.edu")
    print(f"\nThen open in your browser: http://localhost:{port}")
    print("============================================\n")

    print(f"Launching TensorBoard on {hostname} (ARC node)...")
    subprocess.run(["tensorboard", "--logdir", selected_logdir, "--port", port])

if __name__ == "__main__":
    main()
