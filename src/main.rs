use anyhow::Result;
use ort::{session::Session, session::builder::GraphOptimizationLevel, value::TensorRef};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tonic::{Request, Response, Status};

// Ensure your proto package name matches here
tonic::include_proto!("inference.v1");

const MODEL_PATH: &str = "model_artifacts/mnist_cnn.onnx";

/// Loads one session per logical CPU core to enable true parallelism.
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
            // CRITICAL: Set intra-threads to 1 so sessions don't fight for resources.
            // We achieve parallelism by running multiple sessions, not multi-threaded sessions.
            .with_intra_threads(1)?
            .commit_from_file(MODEL_PATH)?;
        sessions.push(Mutex::new(session));
        println!("Session {} initialized.", i + 1);
    }
    Ok(sessions)
}

#[derive(Clone)]
struct InferenceService {
    // A list of protected sessions (one per core)
    sessions: Arc<Vec<Mutex<Session>>>,
    // Atomic counter for Round-Robin scheduling
    counter: Arc<AtomicUsize>,
}

#[tonic::async_trait]
impl rpc_inference_service_server::RpcInferenceService for InferenceService {
    async fn rpc_inference(
        &self,
        req: Request<RpcInferenceRequest>,
    ) -> Result<Response<RpcInferenceResponse>, Status> {
        let req = req.into_inner();

        // 1. Validation
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

        // 2. Data Preparation (ndarray)
        // We do this on the async thread because it's cheap memory moving
        let arr = ndarray::Array::from_shape_vec(shape_usize, req.tensor)
            .map_err(|e| Status::internal(format!("ndarray error: {}", e)))?;

        // 3. Round-Robin Selection
        // Select the next session index atomically
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.sessions.len();

        // Clone the Arc to pass into the blocking thread
        let sessions_arc = self.sessions.clone();

        // 4. Offload heavy compute to a blocking thread
        let output_vec = tokio::task::spawn_blocking(move || {
            // -- BLOCKING THREAD START --

            // Create ORT Tensor View (Zero-copy from ndarray)
            let input_value = TensorRef::from_array_view(&arr)
                .map_err(|e| anyhow::anyhow!("ort tensor error: {}", e))?;

            // Lock the SPECIFIC session for this core
            let mut session = sessions_arc[idx]
                .lock()
                .map_err(|_| anyhow::anyhow!("Mutex poisoned"))?;

            // Run Inference
            let outputs = session
                .run(ort::inputs![input_value])
                .map_err(|e| anyhow::anyhow!("session run error: {}", e))?;

            // Extract Output
            let (_shape, out_slice) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("extract error: {}", e))?;

            Ok::<Vec<f32>, anyhow::Error>(out_slice.to_vec())
            // -- BLOCKING THREAD END --
        })
        .await
        .map_err(|e| Status::internal(format!("Tokio Join error: {}", e)))?
        .map_err(|e| Status::internal(format!("Inference error: {}", e)))?;

        Ok(Response::new(RpcInferenceResponse { output: output_vec }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Initialize Thread Pool
    let sessions = load_sessions().await?;
    println!("Session pool ready.");

    // 2. Initialize Service
    let svc = InferenceService {
        sessions: Arc::new(sessions),
        counter: Arc::new(AtomicUsize::new(0)),
    };

    // 3. Start Server
    let addr = "0.0.0.0:50051".parse().expect("invalid addr");
    println!("Serving gRPC on {}", addr);

    tonic::transport::Server::builder()
        .add_service(rpc_inference_service_server::RpcInferenceServiceServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
