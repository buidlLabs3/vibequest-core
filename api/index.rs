use tower::ServiceBuilder;
use vercel_runtime::{Error, axum::VercelLayer};
use vibequest_core::{app_state, build_router};

#[tokio::main]
async fn main() -> Result<(), Error> {
    let app = ServiceBuilder::new()
        .layer(VercelLayer::new())
        .service(build_router(app_state()));

    vercel_runtime::run(app).await
}
