use anyhow::Result;
use ndarray::IxDyn;
use ort::{session::Session, session::builder::GraphOptimizationLevel, value::TensorRef};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tonic::{Request, Response, Status};

tonic::include_proto!("inference.v1");

const MODEL_PATH: &str = "model_artifacts/mnist_cnn.onnx";

struct InferenceJob {
    arr: ndarray::Array<f32, IxDyn>,
    reply: oneshot::Sender<Result<Vec<f32>, String>>,
}

fn spawn_workers(num_workers: usize) -> Result<Vec<mpsc::Sender<InferenceJob>>> {
    let mut senders = Vec::with_capacity(num_workers);

    for i in 0..num_workers {
        let mut session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(MODEL_PATH)?;
        println!("Session {} initialized.", i + 1);

        let (tx, mut rx) = mpsc::channel::<InferenceJob>(256);

        std::thread::spawn(move || {
            while let Some(job) = rx.blocking_recv() {
                let result = (|| {
                    let input_value = TensorRef::from_array_view(&job.arr)
                        .map_err(|e| format!("ort tensor error: {}", e))?;

                    let outputs = session
                        .run(ort::inputs![input_value])
                        .map_err(|e| format!("session run error: {}", e))?;

                    let (_shape, out_slice) = outputs[0]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| format!("extract error: {}", e))?;

                    Ok(out_slice.to_vec())
                })();

                let _ = job.reply.send(result);
            }
        });

        senders.push(tx);
    }

    Ok(senders)
}

#[derive(Clone)]
struct InferenceService {
    workers: Arc<Vec<mpsc::Sender<InferenceJob>>>,
    counter: Arc<AtomicUsize>,
}

#[tonic::async_trait]
impl rpc_inference_service_server::RpcInferenceService for InferenceService {
    async fn rpc_inference(
        &self,
        req: Request<RpcInferenceRequest>,
    ) -> Result<Response<RpcInferenceResponse>, Status> {
        let req = req.into_inner();

        if req.tensor.is_empty() || req.shape.is_empty() {
            return Err(Status::invalid_argument("tensor and shape required"));
        }

        let shape_usize: Vec<usize> = req.shape.iter().map(|&d| d as usize).collect();
        let expected = shape_usize.iter().product::<usize>();
        if expected != req.tensor.len() {
            return Err(Status::invalid_argument(
                "shape does not match tensor length",
            ));
        }

        let arr = ndarray::Array::from_shape_vec(shape_usize, req.tensor)
            .map_err(|e| Status::internal(format!("ndarray error: {}", e)))?;

        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        let (tx, rx) = oneshot::channel();

        self.workers[idx]
            .send(InferenceJob { arr, reply: tx })
            .await
            .map_err(|_| Status::internal("Worker thread died"))?;

        let output_vec = rx
            .await
            .map_err(|_| Status::internal("Worker dropped response"))?
            .map_err(|e| Status::internal(e))?;

        Ok(Response::new(RpcInferenceResponse { output: output_vec }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let num_cores = std::thread::available_parallelism()?.get();
    println!(
        "Detected {} cores. Spawning {} inference workers...",
        num_cores, num_cores
    );

    let workers = spawn_workers(num_cores)?;
    println!("Worker pool ready.");

    let svc = InferenceService {
        workers: Arc::new(workers),
        counter: Arc::new(AtomicUsize::new(0)),
    };

    let addr = "0.0.0.0:50051".parse().expect("invalid addr");
    println!("Serving gRPC on {}", addr);

    tonic::transport::Server::builder()
        .tcp_nodelay(true)
        .add_service(rpc_inference_service_server::RpcInferenceServiceServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
