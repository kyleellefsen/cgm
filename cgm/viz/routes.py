"""FastAPI routes for the visualization module."""
from typing import Dict
import fastapi
from fastapi import Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .state import VizState
import pathlib
import time

from . import models as m
from ..inference.approximate import sampling
from ..inference.approximate.sampling import ForwardSamplingCertificate


async def get_viz_state(request: fastapi.Request) -> VizState:
    """Dependency to get the visualization state."""
    return request.app.state.viz_state

def setup_routes(app: fastapi.FastAPI, static_dir: pathlib.Path) -> None:
    """Setup all routes for the application."""
    
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the visualization page."""

        cache_buster = f"?v={int(time.time())}"
        html_content = (static_dir / "index.html").read_text()
        # Replace all instances of "?nocache" with the current timestamp
        html_content = html_content.replace("?nocache", cache_buster)
        return HTMLResponse(content=html_content)

    @app.get("/state")
    async def get_state(viz_state: VizState = Depends(get_viz_state)) -> Dict:
        """Return the current graph state as JSON."""
        current_graph = viz_state.current_graph
        if not current_graph:
            return {"nodes": [], "links": []}

        current_state = viz_state.current_graph_state
        nodes = [{
            "id": node.name,
            "states": node.num_states,
            "type": "effect",
            "conditioned_state": -1,
            "cpd": node.cpd.table.html() if node.cpd else None,
            **({"conditioned_state": int(current_state.values[current_state.schema.var_to_idx[node.name]])}
               if current_state and current_state.mask[current_state.schema.var_to_idx[node.name]] else {})
        } for node in current_graph.nodes]

        links = [{"source": parent.name, "target": node.name}
                for node in current_graph.nodes
                for parent in node.parents]

        return {"nodes": nodes, "links": links}

    @app.post("/condition/{node_id}/{state_value}")
    async def condition_node(node_id: str, state_value: int,
                             viz_state: VizState = Depends(get_viz_state)):
        """Update node's conditioned state"""
        if not viz_state.current_graph:
            return {"status": "no graph loaded"}
        
        if viz_state.condition(node_id, None if state_value == -1 else state_value):
            return {"status": "updated"}
        return {"status": "failed"}

    @app.post("/api/sample", response_model=m.SamplingResponse)
    async def generate_samples(request: m.SamplingRequest,
                               viz_state: VizState = Depends(get_viz_state)) -> m.SamplingResponse:
        """Generate samples from the current graph state."""
        current_graph = viz_state.current_graph
        if not current_graph:
            return m.SamplingResponse(
                success=False,
                error=m.SamplingResponseError(
                    error_type="NO_GRAPH",
                    message="No graph loaded"
                )
            )
            
        try:
            # Set random seed and get the seed used
            seed_used = viz_state.set_seed(request.options.random_seed)
            
            # Start with existing graph state if it exists, otherwise create new state with request conditions
            current_state = viz_state.current_graph_state
            if current_state is None:
                return m.SamplingResponse(
                    success=False,
                    error=m.SamplingResponseError(
                        error_type="NO_STATE",
                        message="No graph state available"
                    )
                )

            # Create the sampling certificate
            try:
                cert = ForwardSamplingCertificate(current_state)
            except Exception as e:
                return m.SamplingResponse(
                    success=False,
                    error=m.SamplingResponseError(
                        error_type="INVALID_CONDITIONS",
                        message="Cannot use forward sampling with these conditions",
                        details=(
                            f"{str(e)}. Forward sampling requires that if a node is conditioned, "
                            "all of its ancestors must also be conditioned. Consider using a "
                            "different sampling method or adjusting your conditions to satisfy "
                            "this requirement."
                        )
                    )
                )
                
            # Generate samples
            try:
                sample_array, _ = sampling.get_n_samples(
                    current_graph,
                    viz_state.rng,
                    num_samples=request.num_samples,
                    state=current_state,
                    certificate=cert,
                    return_array=True
                )
            except Exception as e: # pylint: disable=broad-exception-caught
                return m.SamplingResponse(
                    success=False,
                    error=m.SamplingResponseError(
                        error_type="SAMPLING_FAILED",
                        message="Sampling operation failed",
                        details=str(e)
                    )
                )

            if not isinstance(sample_array, sampling.SampleArray):
                return m.SamplingResponse(
                    success=False,
                    error=m.SamplingResponseError(
                        error_type="INVALID_SAMPLE_FORMAT",
                        message=f"Unexpected sample format: {type(sample_array)}"
                    )
                )

            # Store samples and metadata
            metadata = {
                'seed': seed_used,
                'timestamp': time.time(),
                'num_samples': request.num_samples,
            }
            viz_state.store_samples(sample_array, metadata)

            n_samples: int = viz_state.n_samples
            
            return m.SamplingResponse(
                success=True,
                result=m.SamplingResponseSuccess(
                    total_samples=n_samples,
                    accepted_samples=n_samples,
                    rejected_samples=0,
                    seed_used=seed_used
                )
            )

        except Exception as e: # pylint: disable=broad-exception-caught
            import traceback
            error_detail = f"Unexpected error during sampling: {str(e)}\n{traceback.format_exc()}"
            return m.SamplingResponse(
                success=False,
                error=m.SamplingResponseError(
                    error_type="UNEXPECTED_ERROR",
                    message="An unexpected error occurred during sampling",
                    details=error_detail
                )
            )

    @app.post("/api/node_distribution", response_model=m.NodeDistributionResponse)
    async def get_node_distribution(request: m.NodeDistributionRequest,
                                    viz_state: VizState = Depends(get_viz_state)) -> m.NodeDistributionResponse:
        """Get the distribution for a specific node from cached samples."""
        if not viz_state.current_graph:
            return m.NodeDistributionResponse(
                success=False,
                error=m.NodeDistributionError(
                    error_type="NO_GRAPH",
                    message="No graph loaded"
                )
            )

        if viz_state.current_samples is None:
            return m.NodeDistributionResponse(
                success=False,
                error=m.NodeDistributionError(
                    error_type="NO_SAMPLES",
                    message="No samples available. Generate samples first."
                )
            )
        
        samples = viz_state.get_node_samples(request.node_name)
        if samples is None:
            return m.NodeDistributionResponse(
                success=False,
                error=m.NodeDistributionError(
                    error_type="UNKNOWN_NODE",
                    message=f"Unknown node: {request.node_name}"
                )
            )

        node_distribution = viz_state.get_node_distribution(request.node_name,
        request.codomain)
        if node_distribution is None:
            return m.NodeDistributionResponse(
                success=False,
                error=m.NodeDistributionError(
                    error_type="NO_DISTRIBUTION",
                    message="No distribution available"
                )
            )
        x, y = node_distribution
        return m.NodeDistributionResponse(
            success=True,
            result=m.NodeDistributionSuccess(
                node_name=request.node_name,
                codomain=request.codomain,
                x_values=x,
                y_values=y
            )
        ) 