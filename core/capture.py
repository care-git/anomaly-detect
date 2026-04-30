# core/capture.py

from tqdm import tqdm
from scapy.all import sniff, Scapy_Exception

from utils.packet_utils import print_packet_summary
from utils.file_saver import safe_save_path, save_pcap
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


def capture_packets(interface: str, duration: int, packet_count: int, output_path: str) -> None:
    """
    Uses Scapy to sniff packets from the specified network interface until a max packet count
    or duration timeout is reached, and saves result to a PCAP file.

    Parameters:
        interface (str): Name of the network interface to capture packets from.
        duration (int): Time in seconds to continue the capture.
        packet_count (int): Maximum number of packets to capture.
        output_path (str): File path to save the captured packets in PCAP format.
    """

    logger.info(f"Capturing packets on interface '%s' for %ds or %d packets...", interface, duration, packet_count)
    captured = []
    bar = tqdm(total=packet_count, desc="Capturing Packets", unit="pkt", leave=True)
    try:
        def handle(pkt):
            captured.append(pkt)
            bar.update(1)

        sniff(
            iface=interface,
            timeout=duration,
            count=packet_count,
            prn=handle,
        )

        logger.info("Packet capture complete, saving to: %s", output_path)
        save_pcap(captured, output_path)
        logger.info("Saved %d packets.", len(captured))

    except Scapy_Exception as e:
        logger.error("Packet capture failed: %s", e)
        raise
    finally:
        bar.close()


def live_packet_monitor(interface: str, packet_callback: callable, count: int, timeout: int) -> None:
    """
    Uses Scapy to sniff live packets from an interface and forward them to a callback function
    that prints basic packet data to the terminal.

    Parameters:
        interface (str): Name of the network interface to monitor packets on.
        timeout (int): Time in seconds to monitor for (None = unlimited).
        count (int): Maximum number of packets to monitor (0 = unlimited).
        packet_callback (callable): Function that prints basic packet information to screen.
    """

    logger.info("Starting live capture on interface: %s", interface)
    sniff(iface=interface, prn=packet_callback, store=False, count=count, timeout=timeout)


def run_capture(args) -> None:
    """
    Command-line interface handler for network traffic capture.
    
    Parameters:
        args: Parsed command-line arguments containing 'live', 'interface', 'duration', 'packet_count', and 'output' options.
    
    This dispatcher handles the flow of the following capture operations:
        - Network capture and save to PCAP file.
        - Live network monitoring with basic packet data printed to screen.

    Uses config defaults and safe file naming if needed.
    """
    interface = args.interface or config['capture']['interface']
    if args.live:
        packet_count = 0
        duration = None
    else:
        packet_count = args.packet_count or config['capture']['packet_count']
        duration = args.duration or config['capture']['duration']

    if args.live:
        live_packet_monitor(interface, print_packet_summary, packet_count, duration)
    
    else:
        default_output = args.output or config['capture']['output_path']
        safe_output = safe_save_path(default_output)

        capture_packets(interface, duration, packet_count, safe_output)
