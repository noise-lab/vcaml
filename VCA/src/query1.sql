-- To be executed on MLab BigQuery
-- Selects 300 samples satisfying the network condition criterion

SELECT
id,
AVG(cum_tput) as mean_throughput, 
STDDEV(inst_tput) as throughput_jitter,
AVG(rtt)/1000 as mean_rtt,
STDDEV(rtt)/1000 as rtt_jitter,
AVG(loss) as loss_rate
FROM(
  SELECT
  id,
  cum_bytes,
  loss,
  abs_rel_time,
  ((8*cum_bytes)/abs_rel_time) as cum_tput,
  (cum_bytes - bytes_prev) as bytes_diff,
  (abs_rel_time - time_lag) as rel_time_gap,
  (8*(cum_bytes - bytes_prev) / (abs_rel_time - time_lag)) as inst_tput,
  rtt
  FROM
  (
    SELECT
    ndt.id as id,
    a.LossRate as loss,
    client.TCPInfo.RTT as rtt,
    client.TCPInfo.BytesReceived AS cum_bytes, 
    client.TCPInfo.ElapsedTime AS abs_rel_time,
    LAG(client.TCPInfo.BytesReceived) OVER (ORDER BY ndt.id, client.TCPInfo.ElapsedTime) AS bytes_prev,
    LAG(client.TCPInfo.ElapsedTime) OVER (ORDER BY ndt.id, client.TCPInfo.ElapsedTime) AS time_lag,
    FROM
      `measurement-lab.ndt.ndt7` ndt
      CROSS JOIN
      UNNEST(ndt.raw.Upload.ClientMeasurements) as client
    WHERE
      date BETWEEN '2019-01-01' AND '2023-01-31'
      AND array_length(ndt.raw.Upload.ClientMeasurements) > 0
  )
  ORDER BY id, abs_rel_time
)
GROUP BY id
HAVING mean_throughput >= 0.25 and mean_throughput <= 10 AND mean_rtt < 400;