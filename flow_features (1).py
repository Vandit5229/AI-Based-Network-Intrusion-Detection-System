from collections import OrderedDict
import numpy as np

def process_packet(pkt, local_ip=None):
    try:
        packet_dict = {}

        packet_dict["length"] = int(pkt.length)
        packet_dict["ts"] = float(pkt.sniff_timestamp)

        if hasattr(pkt, "ip"):
            packet_dict["protocol"] = pkt.ip.proto
        else:
            packet_dict["protocol"] = 0

        if hasattr(pkt, "tcp") or hasattr(pkt, "udp"):
            if hasattr(pkt, "tcp"):
                packet_dict["src_port"] = int(pkt.tcp.srcport)
                packet_dict["dst_port"] = int(pkt.tcp.dstport)
                packet_dict["flags"] = pkt.tcp.flags
                packet_dict["win_bytes"] = int(pkt.tcp.window_size)
                packet_dict["hdr_len"] = int(pkt.tcp.hdr_len)
            else:
                packet_dict["src_port"] = int(pkt.udp.srcport)
                packet_dict["dst_port"] = int(pkt.udp.dstport)
                packet_dict["flags"] = ""
                packet_dict["win_bytes"] = 0
                packet_dict["hdr_len"] = 0
        else:
            packet_dict["src_port"] = 0
            packet_dict["dst_port"] = 0
            packet_dict["flags"] = ""
            packet_dict["win_bytes"] = 0
            packet_dict["hdr_len"] = 0

        if local_ip and hasattr(pkt, "ip"):
            packet_dict["dir"] = "fwd" if pkt.ip.src == local_ip else "bwd"
        else:
            packet_dict["dir"] = "fwd"

        return packet_dict

    except Exception as e:
        print(f"[!] process_packet error: {e}")
        return None


def compute_flow_features_78(flow_packets):
    try:
        if len(flow_packets) < 2:
            return None

        pkt_lengths = np.array([p["length"] for p in flow_packets])
        pkt_times = np.array([p["ts"] for p in flow_packets])

        flow_duration = (pkt_times.max() - pkt_times.min()) * 1e6

        fwd_pkts = [p for p in flow_packets if p["dir"] == "fwd"]
        bwd_pkts = [p for p in flow_packets if p["dir"] == "bwd"]

        fwd_lengths = np.array([p["length"] for p in fwd_pkts]) if fwd_pkts else np.array([0])
        bwd_lengths = np.array([p["length"] for p in bwd_pkts]) if bwd_pkts else np.array([0])

        pkt_times_sorted = np.sort(pkt_times)
        flow_iats = np.diff(pkt_times_sorted) if len(pkt_times_sorted) > 1 else np.array([0])

        fwd_times = np.array([p["ts"] for p in fwd_pkts]) if fwd_pkts else np.array([0])
        bwd_times = np.array([p["ts"] for p in bwd_pkts]) if bwd_pkts else np.array([0])

        fwd_iats = np.diff(np.sort(fwd_times)) if len(fwd_times) > 1 else np.array([0])
        bwd_iats = np.diff(np.sort(bwd_times)) if len(bwd_times) > 1 else np.array([0])

        flags = [p.get("flags", "") for p in flow_packets]

        fin_count = sum('F' in f for f in flags)
        syn_count = sum('S' in f for f in flags)
        rst_count = sum('R' in f for f in flags)
        psh_count = sum('P' in f for f in flags)
        ack_count = sum('A' in f for f in flags)
        urg_count = sum('U' in f for f in flags)
        ece_count = sum('E' in f for f in flags)
        cwr_count = sum('C' in f for f in flags)

        # CIC-IDS definition:
        # Active state = packets arrive within <= 1 second gap
        diffs = np.diff(pkt_times_sorted)
        active_times = diffs[diffs <= 1.0] if len(diffs) > 0 else np.array([0])
        idle_times = diffs[diffs > 1.0] if len(diffs) > 0 else np.array([0])

        total_fwd = len(fwd_lengths)
        total_bwd = len(bwd_lengths)

        features = OrderedDict({

            "Source Port": flow_packets[0].get("src_port", 0),
            "Destination Port": flow_packets[0].get("dst_port", 0),
            "Protocol": flow_packets[0].get("protocol", 0),

            "Flow Duration": flow_duration,

            "Total Fwd Packets": total_fwd,
            "Total Backward Packets": total_bwd,

            "Total Length of Fwd Packets": np.sum(fwd_lengths),
            "Total Length of Bwd Packets": np.sum(bwd_lengths),

            "Fwd Packet Length Max": fwd_lengths.max(),
            "Fwd Packet Length Min": fwd_lengths.min(),
            "Fwd Packet Length Mean": fwd_lengths.mean(),
            "Fwd Packet Length Std": fwd_lengths.std(),

            "Bwd Packet Length Max": bwd_lengths.max(),
            "Bwd Packet Length Min": bwd_lengths.min(),
            "Bwd Packet Length Mean": bwd_lengths.mean(),
            "Bwd Packet Length Std": bwd_lengths.std(),

            "Flow Bytes/s": (np.sum(fwd_lengths) + np.sum(bwd_lengths)) / (flow_duration/1e6 + 1e-6),
            "Flow Packets/s": len(pkt_lengths) / (flow_duration/1e6 + 1e-6),

            "Flow IAT Mean": flow_iats.mean(),
            "Flow IAT Std": flow_iats.std(),
            "Flow IAT Max": flow_iats.max(),
            "Flow IAT Min": flow_iats.min(),

            "FIN Flag Count": fin_count,
            "SYN Flag Count": syn_count,
            "RST Flag Count": rst_count,
            "PSH Flag Count": psh_count,
            "ACK Flag Count": ack_count,
            "URG Flag Count": urg_count,
            "CWE Flag Count": cwr_count,
            "ECE Flag Count": ece_count,

            "Active Mean": active_times.mean() if len(active_times) else 0,
            "Active Max": active_times.max() if len(active_times) else 0,
            "Idle Mean": idle_times.mean() if len(idle_times) else 0,
            "Idle Max": idle_times.max() if len(idle_times) else 0,
        })

        return features

    except Exception as e:
        print(f"[!] Feature extraction error: {e}")
        return None
