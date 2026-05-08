| Attack     | Top_3_features                                          |
| ---------- | ------------------------------------------------------- |
| Benign     | LONGEST_FLOW_PKT, L4_SRC_PORT, SRC_TO_DST_SECOND_BYTES  |
| backdoor   | OUT_PKTS, L7_PROTO, MIN_TTL                             |
| ddos       | TCP_WIN_MAX_IN, L4_DST_PORT, FLOW_DURATION_MILLISECONDS |
| dos        | TCP_WIN_MAX_IN, DNS_QUERY_TYPE, DNS_QUERY_ID            |
| injection  | TCP_WIN_MAX_IN, DST_TO_SRC_AVG_THROUGHPUT, L4_DST_PORT  |
| mitm       | MIN_IP_PKT_LEN, SHORTEST_FLOW_PKT, DNS_QUERY_ID         |
| password   | L7_PROTO, L4_DST_PORT, TCP_WIN_MAX_OUT                  |
| ransomware | NUM_PKTS_128_TO_256_BYTES, L4_DST_PORT, L4_SRC_PORT     |
| scanning   | IN_BYTES, L7_PROTO, SRC_TO_DST_SECOND_BYTES             |
| xss        | L4_DST_PORT, LONGEST_FLOW_PKT, MAX_IP_PKT_LEN           |
