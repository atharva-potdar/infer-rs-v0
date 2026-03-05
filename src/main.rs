use anyhow::Result;
use ort::{session::Session, session::builder::GraphOptimizationLevel, value::TensorRef};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tonic::{Request, Response, Status};

tonic::include_proto!("inference.v1");

const MODEL_PATH: &str = "model_artifacts/mnist_cnn.onnx";

/// Creates one single-threaded ORT session per logical core.
///
/// `Session::run` requires `&mut self`, so a single shared session serializes
/// all inference behind one lock. Multiple sessions with `intra_threads=1`
/// allow true parallelism without oversubscribing OS threads.
async fn load_sessions() -> Result<Vec<Mutex<Session>>> {
    let num_cores = std::thread::available_parallelism()?.get();
    println!(
        "Detected {} cores. Creating {} inference sessions...",
        num_cores, num_cores
    );

    let mut sessions = Vec::with_capacity(num_cores);
    for i in 0..num_cores {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(MODEL_PATH)?;
        sessions.push(Mutex::new(session));
        println!("Session {} initialized.", i + 1);
    }
    Ok(sessions)
}

#[derive(Clone)]
struct InferenceService {
    sessions: Arc<Vec<Mutex<Session>>>,
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

        // Round-robin across sessions.
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        let sessions_arc = self.sessions.clone();

        // Offload CPU-bound inference to avoid blocking the async executor.
        let output_vec = tokio::task::spawn_blocking(move || {
            let input_value = TensorRef::from_array_view(&arr)
                .map_err(|e| anyhow::anyhow!("ort tensor error: {}", e))?;

            let mut session = sessions_arc[idx]
                .lock()
                .map_err(|_| anyhow::anyhow!("Mutex poisoned"))?;

            let outputs = session
                .run(ort::inputs![input_value])
                .map_err(|e| anyhow::anyhow!("session run error: {}", e))?;

            let (_shape, out_slice) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("extract error: {}", e))?;

            Ok::<Vec<f32>, anyhow::Error>(out_slice.to_vec())
        })
        .await
        .map_err(|e| Status::internal(format!("Tokio Join error: {}", e)))?
        .map_err(|e| Status::internal(format!("Inference error: {}", e)))?;

        Ok(Response::new(RpcInferenceResponse { output: output_vec }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let sessions = load_sessions().await?;
    println!("Session pool ready.");

    let svc = InferenceService {
        sessions: Arc::new(sessions),
        counter: Arc::new(AtomicUsize::new(0)),
    };

    let addr = "0.0.0.0:50051".parse().expect("invalid addr");
    println!("Serving gRPC on {}", addr);

    tonic::transport::Server::builder()
        .add_service(rpc_inference_service_server::RpcInferenceServiceServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
