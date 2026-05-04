#!/usr/bin/env python3
"""
Myo Armband Power Off Script
============================

Sends a deep sleep command to the Myo armband via Bluetooth LE.

Usage:
    python myo_power_off.py              # Auto-discover and power off
    python myo_power_off.py --address XX:XX:XX:XX:XX:XX  # Specific device

Requirements:
    pip install bleak

Author: BioRobotics Course
"""

import asyncio
import argparse
import sys

try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    print("Error: bleak not installed")
    print("Install with: pip install bleak")
    sys.exit(1)


# Myo BLE UUIDs - prefix for matching
MYO_SERVICE_PREFIX = "d5060001"  # Myo Control Service
MYO_COMMAND_PREFIX = "d5060401"  # Command Characteristic

# Myo Commands (from myohw.h)
CMD_VIBRATE = 0x03      # Vibrate command
CMD_DEEP_SLEEP = 0x04   # Deep sleep / power off

# Vibration types
VIBRATE_SHORT = 0x01
VIBRATE_MEDIUM = 0x02
VIBRATE_LONG = 0x03


async def find_myo(timeout: float = 10.0) -> str:
    """Scan for Myo armband and return its address."""
    print(f"Scanning for Myo armband ({timeout}s timeout)...")
    
    devices = await BleakScanner.discover(timeout=timeout)
    
    for device in devices:
        name = device.name or ""
        if "Myo" in name:
            print(f"Found: {device.name} [{device.address}]")
            return device.address
    
    return None


async def power_off_myo(address: str) -> bool:
    """Send deep sleep command to Myo."""
    print(f"Connecting to Myo at {address}...")
    
    try:
        async with BleakClient(address, timeout=15.0) as client:
            if not client.is_connected:
                print("Failed to connect")
                return False
            
            print("Connected!")
            
            # Find the command characteristic
            print("\nDiscovering services...")
            command_char = None
            
            for service in client.services:
                service_uuid = str(service.uuid).lower()
                
                if MYO_SERVICE_PREFIX in service_uuid:
                    print(f"\n  [Myo Control Service] {service.uuid}")
                    
                    for char in service.characteristics:
                        char_uuid = str(char.uuid).lower()
                        props = list(char.properties)
                        print(f"    Char: {char.uuid}")
                        print(f"          handle={char.handle}, props={props}")
                        
                        if MYO_COMMAND_PREFIX in char_uuid:
                            command_char = char
                            print(f"          ^ This is the COMMAND characteristic")
            
            if not command_char:
                print("\n✗ Could not find Myo command characteristic")
                return False
            
            print(f"\n--- Using characteristic ---")
            print(f"  UUID: {command_char.uuid}")
            print(f"  Handle: {command_char.handle}")
            print(f"  Properties: {list(command_char.properties)}")
            
            # Determine write mode based on properties
            use_response = "write" in command_char.properties
            print(f"  Write with response: {use_response}")
            
            # Try vibrate first
            print(f"\n--- Sending VIBRATE command ---")
            vibrate_cmd = bytes([CMD_VIBRATE, 0x01, VIBRATE_MEDIUM])
            print(f"  Bytes: {vibrate_cmd.hex()} = {list(vibrate_cmd)}")
            
            try:
                await client.write_gatt_char(command_char, vibrate_cmd, response=use_response)
                print("  ✓ Vibrate sent (with response={})".format(use_response))
            except Exception as e:
                print(f"  ✗ Failed with response={use_response}: {e}")
                # Try opposite
                try:
                    await client.write_gatt_char(command_char, vibrate_cmd, response=not use_response)
                    print("  ✓ Vibrate sent (with response={})".format(not use_response))
                    use_response = not use_response  # Update for deep sleep
                except Exception as e2:
                    print(f"  ✗ Also failed with response={not use_response}: {e2}")
            
            await asyncio.sleep(1.0)
            
            # Send deep sleep
            print(f"\n--- Sending DEEP SLEEP command ---")
            deep_sleep_cmd = bytes([CMD_DEEP_SLEEP, 0x00])
            print(f"  Bytes: {deep_sleep_cmd.hex()} = {list(deep_sleep_cmd)}")
            
            try:
                await client.write_gatt_char(command_char, deep_sleep_cmd, response=use_response)
                print("  ✓ Deep sleep sent (with response={})".format(use_response))
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                # Try opposite
                try:
                    await client.write_gatt_char(command_char, deep_sleep_cmd, response=not use_response)
                    print("  ✓ Deep sleep sent (with response={})".format(not use_response))
                except Exception as e2:
                    print(f"  ✗ Also failed: {e2}")
                    return False
            
            await asyncio.sleep(0.5)
            
            print("\n" + "="*50)
            print("Commands sent! If Myo didn't respond:")
            print("  - Make sure MyoConnect is not running")
            print("  - Try disconnecting/reconnecting the Myo")
            print("  - The Myo may need firmware that supports BLE commands")
            print("="*50)
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main(address: str = None, timeout: float = 10.0):
    """Main function."""
    if not address:
        address = await find_myo(timeout)
        if not address:
            print("\n✗ No Myo armband found!")
            print("  Make sure it's powered on and not connected to another app.")
            return False
    
    success = await power_off_myo(address)
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Power off Myo armband (deep sleep)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python myo_power_off.py                        # Auto-discover Myo
  python myo_power_off.py -a E4:B3:C2:A1:00:11   # Specific address
  
Note: 
  - The Myo must not be connected to MyoConnect or another app.
  - Deep sleep is the Myo's "off" state (lowest power consumption).
  - To wake: tap the Myo logo or plug in the USB charger.
        """
    )
    parser.add_argument(
        "-a", "--address",
        help="Myo Bluetooth address (e.g., E4:B3:C2:A1:00:11)"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=10.0,
        help="Scan timeout in seconds (default: 10)"
    )
    
    args = parser.parse_args()
    
    try:
        success = asyncio.run(main(args.address, args.timeout))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)