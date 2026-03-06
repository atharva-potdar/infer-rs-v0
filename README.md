# Infer-rs V0

This serves an MNIST model over gRPC. Each logical core on the system gets a dedicated OS thread with its own `ort` session, and the load is distributed with Round-Robin channel dispatch.

The following results are taken on localhost (load-tester and server running on same system) on a Ryzen 5 4600H laptop (6 cores, 12 threads). During the throughput test, the CPU was almost fully saturated (95% reported CPU usage).

## Throughput test
```
ghz --insecure \
    --proto proto/inference/v1/service.proto \
    --call inference.v1.RpcInferenceService/RpcInference \
    -D payload.json \
    -c 200 \
    --connections 10 \
    -z 30s \
    0.0.0.0:50051
```

Summary:
  Count:	894743
  Total:	30.00 s
  Slowest:	33.51 ms
  Fastest:	0.19 ms
  Average:	4.13 ms
  **Requests/sec:	29823.27**

Latency distribution:
  10 % in 1.30 ms 
  25 % in 2.25 ms 
  50 % in 3.68 ms 
  75 % in 5.47 ms 
  90 % in 7.46 ms 
  95 % in 8.88 ms 
  99 % in 12.14 ms

## Latency test
```
ghz --insecure \
    --proto proto/inference/v1/service.proto \
    --call inference.v1.RpcInferenceService/RpcInference \
    -D payload.json \
    -c 1 \
    -n 10000 \
    0.0.0.0:50051
```

Summary:
  Count:	10000
  Total:	5.68 s
  Slowest:	2.42 ms
  Fastest:	0.13 ms
  Average:	0.40 ms
  Requests/sec:	1761.37

Latency distribution:
  10 % in 0.24 ms 
  25 % in 0.36 ms 
  **50 % in 0.41 ms**
  75 % in 0.43 ms 
  90 % in 0.48 ms 
  95 % in 0.56 ms 
  **99 % in 1.08 ms**
