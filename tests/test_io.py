import polars as pl

from vcaml.io.pcap_to_parquet import pcap_to_parquet


def test_pcap_to_parquet(tmp_path):
    # Use a tiny demo PCAP (include in repo under demo/)
    pcap = 'demo/sample.pcap'
    out = tmp_path / 'out.parquet'
    pcap_to_parquet(pcap, out)
    df = pl.read_parquet(out)
    assert {'timestamp_ns', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'length'}.issubset(
        df.columns
    )
    assert df.height > 0
