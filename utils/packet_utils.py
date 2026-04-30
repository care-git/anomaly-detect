# utils/packet_utils.py

import decimal
from scapy.layers.inet import IP
from scapy.data import IP_PROTOS

from utils.logger import get_logger

logger = get_logger(__name__, "INFO")


def drop_columns(df, columns):
    """
    Drops specified columns from a DataFrame if they exist.

    Parameters:
        df (pd.DataFrame): The DataFrame to modify.
        columns (list): List of column names to drop.

    Returns:
        pd.DataFrame: DataFrame with columns removed.
    """
    for col in columns:
        if col in df.columns:
            logger.debug("Dropping column: %s", col)

    return df.drop(columns=[c for c in columns if c in df.columns], errors='ignore')


def print_packet_summary(pkt):
    """
    Logs basic packet information to the screen.
    
    Parameters:
        pkt (scapy.Packet): Packet to display.
    """
    if IP in pkt:
        pkt_src = pkt[IP].src
        pkt_dst = pkt[IP].dst
        proto_num = int(pkt[IP].proto)
        proto_lookup = dict(IP_PROTOS)
        pkt_proto = proto_lookup.get(proto_num, f"IP#{proto_num}")
    else:
        pkt_src = getattr(pkt, 'src', 'unknown')
        pkt_dst = getattr(pkt, 'dst', 'unknown')
        pkt_proto = pkt.name

    logger.info("Packet: %s → %s | Protocol: %s | Length: %s",
                pkt_src, pkt_dst, pkt_proto, len(pkt))


def safe_numeric_cast(x):
    """
    Attempts to safely convert various numeric types to float.
    
    Parameters:
        x: Input value to cast.
        
    Returns: 
        float or original value if cast fails.
    """
    if isinstance(x, (float, int)):
        return x
    
    if isinstance(x, decimal.Decimal):
        return float(x)
    
    if hasattr(x, '__float__'):
        try:
            return float(x)
        except (TypeError, ValueError, OverflowError):
            return x
    return x
